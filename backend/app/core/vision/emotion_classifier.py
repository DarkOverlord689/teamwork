"""CLIP-based zero-shot emotion classifier.

Uses ``openai/clip-vit-base-patch32`` to classify facial expressions into a
configurable set of emotion labels via zero-shot text-image similarity.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import cv2
import numpy as np
import torch
from PIL import Image

from .base_processor import BaseProcessor
from .config import VisionConfig
from .data_types import EmotionData

logger = logging.getLogger(__name__)


class EmotionClassifier(BaseProcessor):
    """Zero-shot emotion classification using CLIP.

    Parameters
    ----------
    config : VisionConfig
        Pipeline configuration (supplies ``clip_model``, ``emotion_labels``,
        and ``device``).
    """

    def __init__(self, config: Optional[VisionConfig] = None) -> None:
        self.config = config or VisionConfig()
        super().__init__(device=self.config.device)
        self._model = None
        self._processor = None
        self._text_prompts: list[str] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load CLIP model and processor onto the target device."""
        from transformers import CLIPModel, CLIPProcessor

        logger.info("Loading CLIP model %s on %s", self.config.clip_model, self.device)
        self._model = CLIPModel.from_pretrained(self.config.clip_model)
        self._processor = CLIPProcessor.from_pretrained(self.config.clip_model)
        self._model.to(self.device)
        self._model.eval()

        # Pre-build the text prompts once
        self._text_prompts = [
            f"a photo of a person with {emotion} expression"
            for emotion in self.config.emotion_labels
        ]

        super().load()

    def unload(self) -> None:
        """Release CLIP model and free GPU memory."""
        del self._model
        del self._processor
        self._model = None
        self._processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        super().unload()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def process(self, face_crop: np.ndarray) -> EmotionData:
        """Classify the dominant emotion in a face crop.

        Parameters
        ----------
        face_crop : np.ndarray
            BGR face crop (as returned by OpenCV).

        Returns
        -------
        EmotionData
            Primary emotion label, confidence, and full probability dict.
        """
        if not self.is_loaded:
            raise RuntimeError("EmotionClassifier is not loaded. Call load() first.")

        # Validate input
        if face_crop is None or face_crop.size == 0:
            return EmotionData(
                primary_emotion="neutral",
                confidence=0.0,
                all_emotions={e: 0.0 for e in self.config.emotion_labels},
            )

        # Convert BGR → RGB → PIL
        rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)

        # Tokenize text + image
        inputs = self._processor(
            text=self._text_prompts,
            images=image,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            logits_per_image = outputs.logits_per_image  # (1, num_labels)
            probs = logits_per_image.softmax(dim=1)

        # Build results
        prob_list = probs[0].cpu().tolist()
        all_emotions = {
            emotion: prob
            for emotion, prob in zip(self.config.emotion_labels, prob_list)
        }

        primary_emotion = max(all_emotions, key=all_emotions.get)  # type: ignore[arg-type]
        confidence = all_emotions[primary_emotion]

        return EmotionData(
            primary_emotion=primary_emotion,
            confidence=confidence,
            all_emotions=all_emotions,
        )

    def process_batch(self, face_crops: List[np.ndarray]) -> List[EmotionData]:
        """Classify emotions for multiple face crops in one forward pass.

        Parameters
        ----------
        face_crops : list of np.ndarray
            List of BGR face crops (as returned by OpenCV).

        Returns
        -------
        list of EmotionData
            One result per input crop, in the same order.
        """
        if not self.is_loaded:
            raise RuntimeError("EmotionClassifier is not loaded. Call load() first.")

        if not face_crops:
            return []

        # Filter out invalid crops, keeping track of indices
        valid_indices: List[int] = []
        pil_images: List[Image.Image] = []
        results: List[EmotionData] = [
            EmotionData(
                primary_emotion="neutral",
                confidence=0.0,
                all_emotions={e: 0.0 for e in self.config.emotion_labels},
            )
            for _ in face_crops
        ]

        for i, crop in enumerate(face_crops):
            if crop is not None and crop.size > 0:
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                pil_images.append(Image.fromarray(rgb))
                valid_indices.append(i)

        if not pil_images:
            return results

        # Process all valid images in a single forward pass
        inputs = self._processor(
            text=self._text_prompts,
            images=pil_images,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            logits_per_image = outputs.logits_per_image  # (N, num_labels)
            probs = logits_per_image.softmax(dim=1)

        # Build results for each valid image
        for batch_idx, orig_idx in enumerate(valid_indices):
            prob_list = probs[batch_idx].cpu().tolist()
            all_emotions = {
                emotion: prob
                for emotion, prob in zip(self.config.emotion_labels, prob_list)
            }
            primary_emotion = max(all_emotions, key=all_emotions.get)  # type: ignore[arg-type]
            confidence = all_emotions[primary_emotion]
            results[orig_idx] = EmotionData(
                primary_emotion=primary_emotion,
                confidence=confidence,
                all_emotions=all_emotions,
            )

        return results
