"""MediaPipe Pose-based body orientation estimator.

Estimates how a person's torso is oriented relative to the camera by
analysing shoulder landmark positions from MediaPipe Pose.
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .base_processor import BaseProcessor
from .config import VisionConfig
from .data_types import PersonFrame, PoseData

logger = logging.getLogger(__name__)

# MediaPipe Pose landmark indices
_LEFT_SHOULDER = 11
_RIGHT_SHOULDER = 12


class PoseEstimator(BaseProcessor):
    """Body orientation estimation via MediaPipe Pose.

    Parameters
    ----------
    config : VisionConfig
        Pipeline configuration (supplies ``min_pose_confidence`` and
        ``device``).
    """

    def __init__(self, config: Optional[VisionConfig] = None) -> None:
        self.config = config or VisionConfig()
        super().__init__(device=self.config.device)
        self._pose = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Initialise the MediaPipe Pose solution."""
        import mediapipe as mp

        self._pose = mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=self.config.min_pose_confidence,
        )
        super().load()

    def unload(self) -> None:
        """Close the MediaPipe Pose solution and release resources."""
        if self._pose is not None:
            self._pose.close()
            self._pose = None
        super().unload()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pad_bbox(
        bbox: Tuple[int, int, int, int],
        frame_shape: Tuple[int, ...],
        padding: float = 0.2,
    ) -> Tuple[int, int, int, int]:
        """Expand a bounding box by *padding* fraction, clamped to frame bounds.

        Parameters
        ----------
        bbox : (x, y, w, h)
        frame_shape : (height, width, ...)
        padding : fraction to expand on each side

        Returns
        -------
        (x1, y1, x2, y2) — pixel coordinates of the padded region.
        """
        x, y, w, h = bbox
        fh, fw = frame_shape[:2]
        pad_w = int(w * padding)
        pad_h = int(h * padding)
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(fw, x + w + pad_w)
        y2 = min(fh, y + h + pad_h)
        return x1, y1, x2, y2

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def process(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> PoseData:
        """Estimate body orientation for one person.

        Parameters
        ----------
        frame : np.ndarray
            Full BGR video frame.
        bbox : (x, y, w, h)
            Bounding box of the detected person/face.

        Returns
        -------
        PoseData
        """
        if not self.is_loaded:
            raise RuntimeError("PoseEstimator is not loaded. Call load() first.")

        low_conf = PoseData(body_orientation=0.0, shoulder_angle=0.0, confidence=0.0)

        if frame is None or frame.size == 0:
            return low_conf

        # Crop person region with padding
        x1, y1, x2, y2 = self._pad_bbox(bbox, frame.shape)
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return low_conf

        # MediaPipe expects RGB
        rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        try:
            results = self._pose.process(rgb_crop)
        except Exception:
            logger.exception("MediaPipe Pose inference failed")
            return low_conf

        if results.pose_landmarks is None:
            return low_conf

        landmarks = results.pose_landmarks.landmark
        left_sh = landmarks[_LEFT_SHOULDER]
        right_sh = landmarks[_RIGHT_SHOULDER]

        # Both shoulders must be reasonably visible
        min_vis = self.config.min_pose_confidence
        if left_sh.visibility < min_vis or right_sh.visibility < min_vis:
            return PoseData(
                body_orientation=0.0,
                shoulder_angle=0.0,
                confidence=min(left_sh.visibility, right_sh.visibility),
            )

        # Shoulder vector in pixel space of the crop
        crop_h, crop_w = crop.shape[:2]
        lx, ly = left_sh.x * crop_w, left_sh.y * crop_h
        rx, ry = right_sh.x * crop_w, right_sh.y * crop_h

        dx = rx - lx
        dy = ry - ly

        # Shoulder angle relative to horizontal (degrees)
        shoulder_angle = math.degrees(math.atan2(dy, dx))

        # Body orientation: approximate from the ratio of apparent shoulder
        # width to expected width.  When facing the camera the shoulders
        # span their full width; turning reduces the apparent width.
        apparent_width = math.sqrt(dx ** 2 + dy ** 2)
        # Normalise by crop width as a rough proxy of expected full width
        ratio = apparent_width / crop_w if crop_w > 0 else 0.0
        # ratio ~0.35-0.45 when facing camera; smaller when turned
        # Map to orientation: 0° = facing, 90° = profile
        body_orientation = max(0.0, (1.0 - ratio / 0.45)) * 90.0
        body_orientation = min(90.0, body_orientation)

        confidence = float(min(left_sh.visibility, right_sh.visibility))

        return PoseData(
            body_orientation=round(body_orientation, 2),
            shoulder_angle=round(shoulder_angle, 2),
            confidence=round(confidence, 4),
        )

    def estimate_all(
        self,
        frame: np.ndarray,
        persons: List[PersonFrame],
    ) -> List[PersonFrame]:
        """Estimate body orientation for all persons in a single frame.

        Parameters
        ----------
        frame : np.ndarray
            Full BGR video frame.
        persons : list of PersonFrame
            Detected persons with bounding boxes.

        Returns
        -------
        list of PersonFrame
            Same list, mutated in-place with ``pose`` populated.
        """
        if not self.is_loaded:
            raise RuntimeError("PoseEstimator is not loaded. Call load() first.")

        for person in persons:
            person.pose = self.process(frame, person.bbox)

        return persons
