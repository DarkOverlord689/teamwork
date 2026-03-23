"""Face detection and landmark extraction using MediaPipe Face Mesh.

Detects multiple faces in a single frame and returns a list of
``PersonFrame`` instances with 468 3-D landmarks and a bounding box.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

import numpy as np

from .base_processor import BaseProcessor
from .config import VisionConfig
from .data_types import PersonFrame

logger = logging.getLogger(__name__)


class FaceDetector(BaseProcessor):
    """Detect faces and extract landmarks via MediaPipe Face Mesh.

    Parameters
    ----------
    config : VisionConfig, optional
        Pipeline configuration.  ``max_faces`` and ``min_face_confidence``
        control the underlying Face Mesh behaviour.
    device : str
        Passed through to ``BaseProcessor``.  MediaPipe runs on CPU
        regardless, but the value is kept for interface consistency.
    """

    def __init__(
        self,
        config: Optional[VisionConfig] = None,
        device: str = "auto",
    ) -> None:
        super().__init__(device=device)
        self.config = config or VisionConfig()
        self._face_mesh: Any = None  # mediapipe FaceMesh instance

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Initialise the MediaPipe Face Mesh model."""
        import mediapipe as mp  # deferred import to keep module importable without mediapipe

        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=self.config.max_faces,
            min_detection_confidence=self.config.min_face_confidence,
            refine_landmarks=True,
        )
        logger.info(
            "FaceDetector loaded (max_faces=%d, min_confidence=%.2f)",
            self.config.max_faces,
            self.config.min_face_confidence,
        )
        super().load()

    def unload(self) -> None:
        """Close the Face Mesh instance and release resources."""
        if self._face_mesh is not None:
            self._face_mesh.close()
            self._face_mesh = None
            logger.info("FaceDetector unloaded")
        super().unload()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def process(self, frame: np.ndarray) -> List[PersonFrame]:
        """Detect faces in *frame* and return per-face ``PersonFrame`` objects.

        Parameters
        ----------
        frame : np.ndarray
            BGR image as returned by OpenCV (H x W x 3, uint8).

        Returns
        -------
        list of PersonFrame
            One entry per detected face.  ``person_id`` is set to ``-1``
            (unassigned) – the ``PersonTracker`` is responsible for
            assigning stable identities.
        """
        if self._face_mesh is None:
            raise RuntimeError(
                "FaceDetector has not been loaded. Call load() first."
            )

        # MediaPipe expects RGB input
        import cv2

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return []

        h, w, _ = frame.shape
        persons: List[PersonFrame] = []

        for face_landmarks in results.multi_face_landmarks:
            # Extract 468 landmarks as (468, 3) numpy array
            landmarks = np.array(
                [
                    [lm.x * w, lm.y * h, lm.z * w]
                    for lm in face_landmarks.landmark
                ],
                dtype=np.float32,
            )

            # Derive bounding box from landmark extremes
            x_coords = landmarks[:, 0]
            y_coords = landmarks[:, 1]

            x_min = int(max(0, np.min(x_coords)))
            y_min = int(max(0, np.min(y_coords)))
            x_max = int(min(w, np.max(x_coords)))
            y_max = int(min(h, np.max(y_coords)))
            bbox_w = x_max - x_min
            bbox_h = y_max - y_min

            # Skip degenerate detections
            if bbox_w <= 0 or bbox_h <= 0:
                logger.debug("Skipping degenerate face detection (zero-area bbox)")
                continue

            persons.append(
                PersonFrame(
                    person_id="unassigned",  # reassigned when tracker runs
                    bbox=(x_min, y_min, bbox_w, bbox_h),
                    landmarks=landmarks,
                )
            )

        logger.debug("Detected %d face(s) in frame", len(persons))
        return persons
