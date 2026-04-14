"""Geometric gaze estimator from MediaPipe face-mesh iris landmarks.

Computes gaze direction purely from landmark geometry — no additional ML
model is required.
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

import numpy as np

from .base_processor import BaseProcessor
from .config import VisionConfig
from .data_types import GazeCategory, GazeData

logger = logging.getLogger(__name__)

# MediaPipe landmark indices
_LEFT_IRIS = list(range(468, 473))   # 468-472
_RIGHT_IRIS = list(range(473, 478))  # 473-477
_LEFT_EYE_OUTER = 33
_LEFT_EYE_INNER = 133
_RIGHT_EYE_INNER = 362
_RIGHT_EYE_OUTER = 263


class GazeEstimator(BaseProcessor):
    """Pure-geometry gaze estimation from MediaPipe iris landmarks.

    Parameters
    ----------
    config : VisionConfig
        Pipeline configuration (supplies ``gaze_camera_threshold`` and
        ``device``).
    """

    def __init__(self, config: Optional[VisionConfig] = None) -> None:
        self.config = config or VisionConfig()
        super().__init__(device=self.config.device)

    # ------------------------------------------------------------------
    # Lifecycle (no-ops — no model to manage)
    # ------------------------------------------------------------------

    def load(self) -> None:
        """No model to load; marks the processor as ready."""
        super().load()

    def unload(self) -> None:
        """No resources to release."""
        super().unload()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _iris_centre(landmarks: np.ndarray, indices: list[int]) -> np.ndarray:
        """Return the mean (x, y, z) position of the given iris landmarks."""
        return landmarks[indices].mean(axis=0)

    @staticmethod
    def _gaze_ratio(iris: np.ndarray, outer: np.ndarray, inner: np.ndarray) -> float:
        """Compute normalised iris position between outer and inner eye corner.

        Returns a value in roughly [0, 1]: 0.5 means centred, <0.5 means
        looking towards the outer corner, >0.5 towards the inner corner.
        """
        eye_width = np.linalg.norm(inner[:2] - outer[:2])
        if eye_width < 1e-6:
            return 0.5
        iris_offset = np.linalg.norm(iris[:2] - outer[:2])
        return float(iris_offset / eye_width)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def process(
        self,
        landmarks: np.ndarray,
        frame_shape: Tuple[int, int],
    ) -> GazeData:
        """Estimate gaze direction from face-mesh landmarks.

        Parameters
        ----------
        landmarks : np.ndarray
            Array of shape ``(478, 3)`` with MediaPipe face-mesh landmarks
            (including iris landmarks 468-477).  Coordinates are normalised
            [0, 1] relative to the frame.
        frame_shape : tuple[int, int]
            ``(height, width)`` of the original frame, used for scaling.

        Returns
        -------
        GazeData
        """
        if not self.is_loaded:
            raise RuntimeError("GazeEstimator is not loaded. Call load() first.")

        # Validate that we have iris landmarks
        if landmarks is None or len(landmarks) < 478:
            return GazeData(
                direction=(0.0, 0.0, 0.0),
                is_looking_at_camera=False,
                confidence=0.0,
                category=GazeCategory.UNKNOWN.value,
            )

        try:
            # Scale landmarks to pixel coordinates for distance calculations
            h, w = frame_shape
            scaled = landmarks.copy()
            scaled[:, 0] *= w
            scaled[:, 1] *= h

            # Iris centres
            left_iris = self._iris_centre(scaled, _LEFT_IRIS)
            right_iris = self._iris_centre(scaled, _RIGHT_IRIS)

            # Eye corners
            left_outer = scaled[_LEFT_EYE_OUTER]
            left_inner = scaled[_LEFT_EYE_INNER]
            right_inner = scaled[_RIGHT_EYE_INNER]
            right_outer = scaled[_RIGHT_EYE_OUTER]

            # Gaze ratios (horizontal)
            left_ratio = self._gaze_ratio(left_iris, left_outer, left_inner)
            right_ratio = self._gaze_ratio(right_iris, right_inner, right_outer)
            avg_h_ratio = (left_ratio + right_ratio) / 2.0

            # Vertical gaze ratio (iris y relative to eye top/bottom)
            left_eye_height = abs(scaled[159, 1] - scaled[145, 1])  # top-bottom of left eye
            right_eye_height = abs(scaled[386, 1] - scaled[374, 1])
            avg_eye_height = (left_eye_height + right_eye_height) / 2.0

            left_v_offset = left_iris[1] - scaled[159, 1]
            right_v_offset = right_iris[1] - scaled[386, 1]
            if avg_eye_height > 1e-6:
                avg_v_ratio = ((left_v_offset + right_v_offset) / 2.0) / avg_eye_height
            else:
                avg_v_ratio = 0.5

            # Convert ratios to approximate yaw / pitch (degrees)
            # ratio 0.5 = centre → 0°; deviation of 0.1 ≈ 15-20°
            yaw = (avg_h_ratio - 0.5) * 120.0   # rough mapping to degrees
            pitch = (avg_v_ratio - 0.5) * 100.0
            roll = 0.0  # not estimated from iris alone

            # Classification
            threshold = self.config.gaze_camera_threshold
            abs_yaw = abs(yaw)
            abs_pitch = abs(pitch)

            if abs_yaw <= threshold and abs_pitch <= threshold:
                category = GazeCategory.CAMERA.value
                is_looking = True
            elif abs_pitch > threshold and pitch > 0:
                # Looking down — likely at screen
                category = GazeCategory.SCREEN.value
                is_looking = False
            elif abs_yaw > threshold:
                # Looking sideways — peer or away
                category = GazeCategory.PEER.value if abs_yaw < 45.0 else GazeCategory.AWAY.value
                is_looking = False
            else:
                category = GazeCategory.AWAY.value
                is_looking = False

            # Confidence: higher when closer to the decision boundary centre
            deviation = math.sqrt(yaw ** 2 + pitch ** 2)
            confidence = max(0.0, min(1.0, 1.0 - deviation / 90.0))

            return GazeData(
                direction=(round(yaw, 2), round(pitch, 2), round(roll, 2)),
                is_looking_at_camera=is_looking,
                confidence=round(confidence, 4),
                category=category,
            )

        except Exception:
            logger.exception("Gaze estimation failed")
            return GazeData(
                direction=(0.0, 0.0, 0.0),
                is_looking_at_camera=False,
                confidence=0.0,
                category=GazeCategory.UNKNOWN.value,
            )
