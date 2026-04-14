"""Multimodal frame analyzer for smart frame selection.

Uses a vision-language model served by Ollama Cloud (qwen3-vl:8b by default) to
produce rich natural-language descriptions of key frames.  Falls back to the
existing CLIP emotion classifier on timeout or request error.
"""

from __future__ import annotations

import base64
import io
import json
import logging
from typing import Any

import httpx
import numpy as np
from PIL import Image

from app.core.audio.data_types import MomentType
from app.core.vision.config import VisionConfig
from app.core.vision.data_types import MultimodalFrameAnalysis

logger = logging.getLogger(__name__)

ANALYSIS_PROMPT_TEMPLATE = """Analyze this image of a classroom participant.
The person is in moment type: {moment_type}.
Context: {context_text}

Respond ONLY with valid JSON matching this exact schema:
{{
  "emotion_primary": "<one of: happy, sad, angry, surprised, fearful, disgusted, neutral, attentive, confused>",
  "emotion_confidence": <float 0.0-1.0>,
  "attention_direction": "<one of: camera, other_person, downward, away, unclear>",
  "engagement_level": "<one of: high, medium, low, unclear>",
  "description": "<1-2 sentence natural language description of the person's expression and posture>"
}}"""


_MAX_FRAME_DIM = 512  # resize before sending to Ollama to avoid oversized payloads


def _resize_frame(frame_rgb: np.ndarray) -> np.ndarray:
    """Proportionally resize *frame_rgb* so neither side exceeds _MAX_FRAME_DIM."""
    h, w = frame_rgb.shape[:2]
    if max(h, w) <= _MAX_FRAME_DIM:
        return frame_rgb
    if w >= h:
        new_w = _MAX_FRAME_DIM
        new_h = int(h * _MAX_FRAME_DIM / w)
    else:
        new_h = _MAX_FRAME_DIM
        new_w = int(w * _MAX_FRAME_DIM / h)
    pil = Image.fromarray(frame_rgb)
    return np.asarray(pil.resize((new_w, new_h), Image.LANCZOS))


def _frame_to_base64(frame_rgb: np.ndarray) -> str:
    """Encode an RGB numpy array as a base64 JPEG string (after resizing)."""
    resized = _resize_frame(frame_rgb)
    pil_image = Image.fromarray(resized)
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


class MultimodalAnalyzer:
    """Analyze key frames using a vision-language model served by Ollama.

    Parameters
    ----------
    config : VisionConfig
        Pipeline configuration.  ``multimodal_*`` fields control model
        selection, device, and timeout.
    """

    def __init__(self, config: VisionConfig) -> None:
        self.config = config
        self._clip_fallback: Any = None

        # Resolve Ollama Cloud settings from app settings when available, otherwise
        # fall back to sensible defaults.
        try:
            from app.utils.config import settings as _app_settings
            self._ollama_base_url: str = _app_settings.ollama_base_url
            self._ollama_api_key: str = _app_settings.ollama_api_key
            self._ollama_model: str = _app_settings.ollama_vision_model
        except Exception:
            self._ollama_base_url = "https://api.ollama.com"
            self._ollama_api_key = ""
            self._ollama_model = "qwen3-vl:8b"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Validate connectivity to the Ollama Cloud server.  Called once per worker."""
        logger.info(
            "Connecting to Ollama Cloud at %s (model=%s)",
            self._ollama_base_url,
            self._ollama_model,
        )
        try:
            headers = self._auth_headers()
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(f"{self._ollama_base_url}/v1/models", headers=headers)
                resp.raise_for_status()
            logger.info("Ollama Cloud connection verified.")
        except Exception as exc:
            logger.warning(
                "Could not reach Ollama Cloud at %s: %s. "
                "Inference calls will proceed anyway and fail individually if needed.",
                self._ollama_base_url,
                exc,
            )

    def unload(self) -> None:
        """No-op: Ollama Cloud manages model lifecycle on its own side."""
        logger.info("MultimodalAnalyzer unloaded (Ollama Cloud backend — nothing to free).")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_frame(
        self,
        frame_rgb: np.ndarray,
        moment_type: MomentType,
        context_text: str,
        speaker_id: str,
        person_id: str,
        frame_number: int,
        timestamp_ms: float,
    ) -> MultimodalFrameAnalysis:
        """Analyze a key frame using the Ollama vision model with fallback to CLIP.

        Parameters
        ----------
        frame_rgb : np.ndarray
            RGB frame as a numpy array.
        moment_type : MomentType
            The type of conversational moment this frame captures.
        context_text : str
            Short textual context (e.g. the question that was asked).
        speaker_id : str
            Diarised speaker identifier.
        person_id : str
            Tracked face/person identifier.
        frame_number : int
            Original frame index in the video.
        timestamp_ms : float
            Frame timestamp in milliseconds.

        Returns
        -------
        MultimodalFrameAnalysis
        """
        prompt = ANALYSIS_PROMPT_TEMPLATE.format(
            moment_type=moment_type.value,
            context_text=context_text or "(none)",
        )

        face_detected = frame_rgb is not None and frame_rgb.size > 0
        fallback_model = self._ollama_model
        fallback_reason: str | None = None
        parsed: dict = {}

        try:
            raw = self._invoke_multimodal(frame_rgb, prompt)
            parsed = self._parse_response(raw)
            fallback_model = self._ollama_model
        except TimeoutError:
            fallback_reason = "timeout"
            logger.warning(
                "Ollama Cloud inference timed out for frame %d, using CLIP fallback.",
                frame_number,
            )
            parsed = self._clip_fallback_analyze(frame_rgb)
            fallback_model = "clip"
        except Exception as exc:
            fallback_reason = str(exc)
            logger.warning(
                "Ollama Cloud inference failed for frame %d (%s), using CLIP fallback.",
                frame_number,
                exc,
            )
            parsed = self._clip_fallback_analyze(frame_rgb)
            fallback_model = "clip"

        return self._build_frame_analysis(
            parsed=parsed,
            frame_number=frame_number,
            timestamp_ms=timestamp_ms,
            speaker_id=speaker_id,
            person_id=person_id,
            moment_type=moment_type,
            face_detected=face_detected,
            fallback_model=fallback_model,
            fallback_reason=fallback_reason,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _auth_headers(self) -> dict[str, str]:
        """Return Authorization header dict if an API key is configured."""
        if self._ollama_api_key:
            return {"Authorization": f"Bearer {self._ollama_api_key}"}
        return {}

    def _invoke_multimodal(self, frame_rgb: np.ndarray, prompt: str) -> str:
        """Send a chat request to the Ollama Cloud OpenAI-compatible endpoint.

        Ollama Cloud exposes an OpenAI-compatible API at ``/v1/chat/completions``
        rather than the local Ollama ``/api/chat`` endpoint.  Vision input is
        passed via the standard OpenAI image_url content part using a
        ``data:image/jpeg;base64,`` URI.

        Raises
        ------
        TimeoutError
            If the request exceeds ``config.multimodal_timeout_seconds``.
        httpx.HTTPError
            On HTTP-level errors from the Ollama Cloud server.
        """
        timeout = float(self.config.multimodal_timeout_seconds)
        image_b64 = _frame_to_base64(frame_rgb)

        # OpenAI-compatible format required by Ollama Cloud /v1/chat/completions
        payload = {
            "model": self._ollama_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}",
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
            "stream": False,
        }

        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(
                    f"{self._ollama_base_url}/v1/chat/completions",
                    json=payload,
                    headers=self._auth_headers(),
                )
                if resp.status_code >= 400:
                    logger.warning(
                        "Ollama Cloud error %d body: %s", resp.status_code, resp.text[:500]
                    )
                resp.raise_for_status()
        except httpx.TimeoutException as exc:
            raise TimeoutError("Ollama Cloud inference timed out") from exc

        # OpenAI-compatible response: {"choices": [{"message": {"content": "..."}}]}
        data = resp.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "")

    def _parse_response(self, raw: str) -> dict:
        """Parse JSON from model output.  One retry with basic cleanup."""
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # Try to extract a JSON object from the response
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(raw[start:end])
            except json.JSONDecodeError:
                pass

        logger.warning("Could not parse multimodal response as JSON: %s", raw[:200])
        return {}

    def _clip_fallback_analyze(self, frame_rgb: np.ndarray) -> dict:
        """Use the CLIP emotion classifier as a fallback."""
        if self._clip_fallback is None:
            try:
                from app.core.vision.emotion_classifier import EmotionClassifier
                from app.core.vision.config import VisionConfig as VC
                self._clip_fallback = EmotionClassifier(config=VC())
                self._clip_fallback.load()
            except Exception as exc:
                logger.warning("Could not load CLIP fallback: %s", exc)
                return {
                    "emotion_primary": "neutral",
                    "emotion_confidence": 0.0,
                    "attention_direction": "unclear",
                    "engagement_level": "unclear",
                    "description": "Analysis unavailable.",
                }

        try:
            import cv2
            bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            # process() expects a single np.ndarray, NOT a list
            emotion_data = self._clip_fallback.process(bgr)
            if emotion_data is not None:
                return {
                    "emotion_primary": emotion_data.primary_emotion,
                    "emotion_confidence": emotion_data.confidence,
                    "attention_direction": "unclear",
                    "engagement_level": "unclear",
                    "description": f"CLIP fallback: {emotion_data.primary_emotion}",
                }
        except Exception as exc:
            logger.warning("CLIP fallback analysis failed: %s", exc)

        return {
            "emotion_primary": "neutral",
            "emotion_confidence": 0.0,
            "attention_direction": "unclear",
            "engagement_level": "unclear",
            "description": "Analysis unavailable.",
        }

    def _build_frame_analysis(
        self,
        parsed: dict,
        frame_number: int,
        timestamp_ms: float,
        speaker_id: str,
        person_id: str,
        moment_type: MomentType,
        face_detected: bool,
        fallback_model: str,
        fallback_reason: str | None,
    ) -> MultimodalFrameAnalysis:
        """Assemble the final MultimodalFrameAnalysis result object."""
        return MultimodalFrameAnalysis(
            frame_number=frame_number,
            timestamp_ms=timestamp_ms,
            speaker_id=speaker_id,
            person_id=person_id,
            moment_type=moment_type,
            multimodal_description=parsed.get("description", ""),
            emotion_primary=parsed.get("emotion_primary", "neutral"),
            emotion_confidence=float(parsed.get("emotion_confidence", 0.0)),
            attention_direction=parsed.get("attention_direction", "unclear"),
            engagement_level=parsed.get("engagement_level", "unclear"),
            face_detected=face_detected,
            fallback_model=fallback_model,
            fallback_reason=fallback_reason,
        )
