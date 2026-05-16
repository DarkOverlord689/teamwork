"""Speaker diarization using Deepgram Nova-2.

Uses the Deepgram ``listen.v1.media.transcribe_file`` API with ``diarize=True``
to produce a list of :class:`~app.core.audio.data_types.SpeakerSegment`
objects sorted by start time with normalized speaker IDs
(``speaker_0``, ``speaker_1``, … assigned in order of first appearance).
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path
from typing import List

from app.core.audio.base_processor import AudioBaseProcessor
from app.core.audio.config import AudioConfig
from app.core.audio.data_types import SpeakerSegment

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class DiarizationError(Exception):
    """Raised when speaker diarization fails."""


# ---------------------------------------------------------------------------
# Deepgram import (lazy, for testability)
# ---------------------------------------------------------------------------

try:
    from deepgram import DeepgramClient  # type: ignore
except ImportError:  # pragma: no cover
    DeepgramClient = None  # type: ignore


# ---------------------------------------------------------------------------
# Diarizer
# ---------------------------------------------------------------------------


class Diarizer(AudioBaseProcessor):
    """Run Deepgram speaker diarization on an audio file.

    Parameters
    ----------
    config : AudioConfig
        Pipeline configuration including Deepgram API key, speaker bounds, and
        minimum segment duration.
    """

    processor_name = "diarizer"

    def __init__(self, config: AudioConfig) -> None:
        super().__init__(device=config.audio_device)
        self.config = config
        self._client = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Validate that the Deepgram API key is available.

        Raises
        ------
        DiarizationError
            If ``deepgram_api_key`` is empty or the SDK is not installed.
        """
        if not self.config.deepgram_api_key:
            raise DiarizationError(
                "deepgram_api_key is required for diarization. "
                "Obtain a key from https://console.deepgram.com/signup and "
                "set AudioConfig.deepgram_api_key."
            )

        if DeepgramClient is None:
            raise DiarizationError(
                "deepgram-sdk is not installed. "
                "Install it with: pip install deepgram-sdk"
            )

        self._client = DeepgramClient(api_key=self.config.deepgram_api_key)
        self._loaded = True
        logger.info("Deepgram client initialised for diarization")

    def unload(self) -> None:
        """Release the Deepgram client reference."""
        self._client = None
        self._loaded = False
        logger.info("Deepgram client unloaded")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def process(self, audio_path: str) -> list[SpeakerSegment]:
        """Run diarization on *audio_path*.

        Parameters
        ----------
        audio_path : str
            Path to a WAV (or compatible) audio file.

        Returns
        -------
        list[SpeakerSegment]
            Segments sorted by ``start`` time with normalized speaker IDs.

        Raises
        ------
        DiarizationError
            If the client has not been initialised or the API call fails.
        """
        if not self._loaded or self._client is None:
            raise DiarizationError(
                "Diarizer is not loaded. Call load() or use as a context manager."
            )

        path = Path(audio_path)
        if not path.is_file():
            raise DiarizationError(f"Audio file not found: {audio_path}")

        try:
            logger.info("Running Deepgram diarization on %s", audio_path)
            with open(audio_path, "rb") as audio:
                response = self._client.listen.v1.media.transcribe_file(
                    request=audio.read(),
                    model=self.config.deepgram_model,
                    diarize=True,
                    punctuate=True,
                    language=self.config.whisper_language,
                    smart_format=True,
                    utterances=True,
                )
        except Exception as exc:
            raise DiarizationError(
                f"Deepgram diarization failed on '{audio_path}': {exc}"
            ) from exc

        # Extract utterances with speaker labels
        utterances = (
            response.results.utterances
            if response.results and response.results.utterances
            else []
        )
        if not utterances:
            logger.warning("Deepgram returned no utterances for %s", audio_path)
            return []

        raw_tracks: list[tuple[float, float, str]] = []
        for u in utterances:
            speaker = str(getattr(u, "speaker", 0))
            raw_tracks.append((float(u.start), float(u.end), speaker))

        # Normalize labels by order of first appearance
        label_map: dict[str, str] = OrderedDict()
        all_labels = [label for _, _, label in raw_tracks]
        normalized_labels = self._normalize_labels(all_labels, label_map=label_map)

        # Build SpeakerSegment list, filtering short segments
        segments: list[SpeakerSegment] = []
        for (start, end, _), norm_label in zip(raw_tracks, normalized_labels):
            duration = end - start
            if duration < self.config.diarize_min_duration:
                logger.debug(
                    "Skipping short segment %.2f–%.2f (%s, %.3fs < min %.3fs)",
                    start,
                    end,
                    norm_label,
                    duration,
                    self.config.diarize_min_duration,
                )
                continue
            segments.append(SpeakerSegment(start=start, end=end, speaker_id=norm_label))

        segments.sort(key=lambda s: s.start)
        logger.info(
            "Diarization complete: %d segments, %d speakers",
            len(segments),
            len(label_map),
        )
        return segments

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _normalize_labels(
        self,
        raw_labels: list[str],
        label_map: dict[str, str] | None = None,
    ) -> list[str]:
        """Normalize raw speaker labels to ``speaker_0``, ``speaker_1``, …

        Labels are assigned in order of **first appearance** in *raw_labels*,
        not alphabetically.

        Parameters
        ----------
        raw_labels : list[str]
            Raw labels as returned by Deepgram (e.g. ``"0"``, ``"1"``).
        label_map : dict, optional
            If provided, the mapping is written into this dict (useful for
            inspecting the final mapping after the call).

        Returns
        -------
        list[str]
            Normalized labels in the same order as *raw_labels*.
        """
        mapping: dict[str, str] = {}
        normalized: list[str] = []
        for raw in raw_labels:
            if raw not in mapping:
                mapping[raw] = f"speaker_{len(mapping)}"
            normalized.append(mapping[raw])
        if label_map is not None:
            label_map.update(mapping)
        return normalized
