"""Person tracking via face-embedding matching and spatial IoU fallback.

Uses ``facenet-pytorch`` ``InceptionResnetV1`` (pretrained on VGGFace2) to
compute 512-d face embeddings and matches them against a gallery of known
embeddings using cosine similarity.  When embedding extraction fails (e.g.
low-quality crop), an IoU-based spatial fallback is used.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from .base_processor import BaseProcessor
from .config import VisionConfig
from .data_types import PersonFrame

logger = logging.getLogger(__name__)


def _iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    """Compute Intersection-over-Union for two ``(x, y, w, h)`` boxes."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0

    area_a = aw * ah
    area_b = bw * bh
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0

    return inter / union


class PersonTracker(BaseProcessor):
    """Assign stable ``person_id`` values across frames using face embeddings.

    Parameters
    ----------
    config : VisionConfig, optional
        Pipeline configuration.  ``embedding_similarity_threshold`` controls
        the cosine-similarity cut-off for matching.
    device : str
        ``"auto"`` (default) selects CUDA when available.
    """

    def __init__(
        self,
        config: Optional[VisionConfig] = None,
        device: str = "auto",
    ) -> None:
        super().__init__(device=device)
        self.config = config or VisionConfig()

        # Face-embedding model (loaded lazily via ``load()``)
        self._model: Any = None

        # Gallery of known person embeddings
        self.known_embeddings: Dict[str, np.ndarray] = {}
        self.next_id: int = 0

        # Previous frame bboxes for IoU fallback
        self._prev_bboxes: Dict[str, Tuple[int, int, int, int]] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load InceptionResnetV1 (VGGFace2) onto the configured device."""
        from facenet_pytorch import InceptionResnetV1

        self._model = InceptionResnetV1(pretrained="vggface2").eval()
        self._model = self._model.to(self.device)
        logger.info("PersonTracker model loaded on %s", self.device)
        super().load()

    def unload(self) -> None:
        """Release model weights and clear CUDA cache if applicable."""
        if self._model is not None:
            del self._model
            self._model = None

        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.known_embeddings.clear()
        self.next_id = 0
        self._prev_bboxes.clear()
        logger.info("PersonTracker unloaded")
        super().unload()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def process(
        self,
        frame: np.ndarray,
        persons: List[PersonFrame],
    ) -> List[PersonFrame]:
        """Assign ``person_id`` and ``face_embedding`` to each *PersonFrame*.

        Parameters
        ----------
        frame : np.ndarray
            The full BGR frame (used for cropping faces).
        persons : list of PersonFrame
            Detected faces for the current frame (typically from
            ``FaceDetector.process``).

        Returns
        -------
        list of PersonFrame
            The same list, mutated in-place with ``person_id`` and
            ``face_embedding`` populated.
        """
        if self._model is None:
            raise RuntimeError(
                "PersonTracker has not been loaded. Call load() first."
            )

        current_bboxes: Dict[str, Tuple[int, int, int, int]] = {}

        for person in persons:
            embedding = self._get_embedding(frame, person.bbox)

            if embedding is not None:
                person_id = self._match_by_embedding(embedding)
                person.face_embedding = embedding
            else:
                # Fallback: spatial IoU against previous frame
                person_id = self._match_by_iou(person.bbox)

            person.person_id = person_id
            current_bboxes[person_id] = person.bbox

        # Update previous-frame bboxes for the next call
        self._prev_bboxes = current_bboxes
        return persons

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_embedding(
        self, frame: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> Optional[np.ndarray]:
        """Crop, preprocess, and compute a 512-d face embedding.

        Returns ``None`` if the crop is invalid or inference fails.
        """
        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            return None

        h_frame, w_frame = frame.shape[:2]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w_frame, x + w)
        y2 = min(h_frame, y + h)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        try:
            # Convert BGR → RGB, resize to 160x160 (InceptionResnetV1 input)
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop_resized = cv2.resize(crop_rgb, (160, 160))

            # Normalise to [-1, 1] and convert to tensor (1, 3, 160, 160)
            tensor = (
                torch.from_numpy(crop_resized)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float()
                .div(255.0)
                .sub(0.5)
                .div(0.5)
                .to(self.device)
            )

            with torch.no_grad():
                emb = self._model(tensor)

            return emb.cpu().numpy().flatten()
        except Exception:
            logger.debug("Embedding extraction failed for bbox %s", bbox, exc_info=True)
            return None

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two 1-D vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _match_by_embedding(self, embedding: np.ndarray) -> str:
        """Find best matching known person or create a new one."""
        best_id: Optional[str] = None
        best_sim: float = -1.0
        threshold = self.config.embedding_similarity_threshold

        for pid, known_emb in self.known_embeddings.items():
            sim = self._cosine_similarity(embedding, known_emb)
            if sim > best_sim:
                best_sim = sim
                best_id = pid

        if best_sim >= threshold and best_id is not None:
            # Update gallery with exponential moving average
            self.known_embeddings[best_id] = (
                0.8 * self.known_embeddings[best_id] + 0.2 * embedding
            )
            return best_id

        # New person
        new_id = f"person_{self.next_id}"
        self.next_id += 1
        self.known_embeddings[new_id] = embedding
        return new_id

    def _match_by_iou(self, bbox: Tuple[int, int, int, int]) -> str:
        """Spatial fallback: match against previous-frame bounding boxes."""
        best_id: Optional[str] = None
        best_iou: float = 0.0
        iou_threshold = 0.3

        for pid, prev_bbox in self._prev_bboxes.items():
            score = _iou(bbox, prev_bbox)
            if score > best_iou:
                best_iou = score
                best_id = pid

        if best_iou >= iou_threshold and best_id is not None:
            return best_id

        # No spatial match – assign new id
        new_id = f"person_{self.next_id}"
        self.next_id += 1
        return new_id
