"""Temporal landmark-based gesture analyser.

Detects head gestures (nodding, shaking) and facial micro-expressions
(frowning) by tracking landmark displacement over a sliding window.
"""

from __future__ import annotations

import collections
import logging
from typing import Dict, Optional, Union

import numpy as np

from .base_processor import BaseProcessor
from .config import VisionConfig
from .data_types import GestureData, GestureType

logger = logging.getLogger(__name__)

# MediaPipe landmark indices
_NOSE_TIP = 1
_LEFT_BROW = [65, 66, 67, 68]
_RIGHT_BROW = [295, 296, 297, 298]
# Upper-eye landmarks used as baseline for brow depression
_LEFT_EYE_UPPER = [159, 160]
_RIGHT_EYE_UPPER = [386, 387]


class GestureAnalyzer(BaseProcessor):
    """Temporal gesture detection from face-mesh landmark sequences.

    Parameters
    ----------
    config : VisionConfig
        Pipeline configuration (supplies ``gesture_window_size``,
        ``nod_threshold``, and ``device``).
    """

    def __init__(self, config: Optional[VisionConfig] = None) -> None:
        self.config = config or VisionConfig()
        super().__init__(device=self.config.device)
        self.history: Dict[Union[str, int], collections.deque] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Initialise internal history buffers."""
        self.history = {}
        super().load()

    def unload(self) -> None:
        """Clear history buffers."""
        self.history.clear()
        super().unload()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_oscillation(values: list[float], threshold: float) -> tuple[bool, float]:
        """Detect an oscillation (up-down-up or down-up-down) in *values*.

        Returns ``(detected, intensity)`` where *intensity* is the peak
        deviation normalised by *threshold*.
        """
        if len(values) < 4:
            return False, 0.0

        diffs = np.diff(values)
        sign_changes = np.diff(np.sign(diffs))
        num_sign_changes = int(np.count_nonzero(sign_changes))

        if num_sign_changes < 2:
            return False, 0.0

        amplitude = float(np.max(values) - np.min(values))
        if amplitude < threshold:
            return False, 0.0

        intensity = min(1.0, amplitude / (threshold * 5.0))
        return True, intensity

    @staticmethod
    def _brow_depression(landmarks: np.ndarray) -> float:
        """Measure how depressed (lowered) the brows are relative to the eyes.

        A larger *positive* value means the brows are closer to the eyes
        (frowning).  Returns 0.0 if landmarks are insufficient.
        """
        try:
            left_brow_y = np.mean([landmarks[i, 1] for i in _LEFT_BROW])
            right_brow_y = np.mean([landmarks[i, 1] for i in _RIGHT_BROW])
            left_eye_y = np.mean([landmarks[i, 1] for i in _LEFT_EYE_UPPER])
            right_eye_y = np.mean([landmarks[i, 1] for i in _RIGHT_EYE_UPPER])

            # In normalised coords y increases downward, so brow < eye when
            # brows are raised.  Depression = small gap.
            left_gap = left_eye_y - left_brow_y
            right_gap = right_eye_y - right_brow_y
            avg_gap = (left_gap + right_gap) / 2.0
            return float(avg_gap)
        except (IndexError, ValueError):
            return 0.0

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def process(self, person_id: Union[str, int], landmarks: np.ndarray) -> GestureData:
        """Analyse gestures for a tracked person.

        Parameters
        ----------
        person_id : int
            Unique identifier for the person being tracked (so that each
            person gets their own sliding window).
        landmarks : np.ndarray
            Face-mesh landmarks of shape ``(468, 3)`` or ``(478, 3)``
            with normalised coordinates.

        Returns
        -------
        GestureData
        """
        if not self.is_loaded:
            raise RuntimeError("GestureAnalyzer is not loaded. Call load() first.")

        neutral = GestureData(
            gesture_type=GestureType.NEUTRAL.value,
            confidence=0.0,
            intensity=0.0,
        )

        if landmarks is None or len(landmarks) < 300:
            return neutral

        # Ensure a history buffer exists for this person
        window = self.config.gesture_window_size
        if person_id not in self.history:
            self.history[person_id] = collections.deque(maxlen=window)

        # Store current nose-tip coordinates and brow depression
        nose = landmarks[_NOSE_TIP]
        brow_dep = self._brow_depression(landmarks)
        self.history[person_id].append((nose[0], nose[1], brow_dep))

        buf = self.history[person_id]
        if len(buf) < 4:
            return neutral

        # Extract time-series from buffer
        xs = [entry[0] for entry in buf]
        ys = [entry[1] for entry in buf]
        brow_deps = [entry[2] for entry in buf]

        threshold = self.config.nod_threshold

        # --- Nodding: y-axis oscillation of nose tip ---
        nod_detected, nod_intensity = self._detect_oscillation(ys, threshold)
        if nod_detected:
            return GestureData(
                gesture_type=GestureType.NOD.value,
                confidence=min(1.0, nod_intensity + 0.3),
                intensity=nod_intensity,
            )

        # --- Head shake: x-axis oscillation of nose tip ---
        shake_detected, shake_intensity = self._detect_oscillation(xs, threshold)
        if shake_detected:
            return GestureData(
                gesture_type=GestureType.SHAKE.value,
                confidence=min(1.0, shake_intensity + 0.3),
                intensity=shake_intensity,
            )

        # --- Frowning: sustained brow depression ---
        mean_brow = float(np.mean(brow_deps))
        baseline_gap = 0.04  # approximate relaxed brow-eye gap in normalised coords
        depression = max(0.0, baseline_gap - mean_brow)
        if depression > threshold * 0.5:
            frown_intensity = min(1.0, depression / (baseline_gap * 0.5))
            return GestureData(
                gesture_type=GestureType.FROWN.value,
                confidence=min(1.0, frown_intensity * 0.8),
                intensity=frown_intensity,
            )

        return neutral
