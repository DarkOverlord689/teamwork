"""Tests for the GestureAnalyzer module (T5.5)."""

import numpy as np
import pytest

from backend.app.core.vision.config import VisionConfig
from backend.app.core.vision.data_types import GestureData, GestureType
from backend.app.core.vision.gesture_analyzer import GestureAnalyzer


@pytest.fixture
def config():
    return VisionConfig(gesture_window_size=10, nod_threshold=0.03)


@pytest.fixture
def analyzer(config):
    ga = GestureAnalyzer(config=config)
    ga.load()
    yield ga
    ga.unload()


def _make_landmarks(nose_x=0.5, nose_y=0.5, brow_gap=0.04):
    """Create synthetic 478x3 landmarks with configurable nose position and brow gap.

    brow_gap: distance between brow and eye upper landmarks (larger = more relaxed).
    """
    landmarks = np.random.rand(478, 3).astype(np.float32) * 0.01 + 0.45

    # Nose tip (index 1)
    landmarks[1] = [nose_x, nose_y, 0.0]

    # Left brow (65, 66, 67, 68) — placed above left eye upper (159, 160)
    eye_upper_y = 0.40
    brow_y = eye_upper_y - brow_gap
    for idx in [65, 66, 67, 68]:
        landmarks[idx] = [0.40, brow_y, 0.0]
    for idx in [159, 160]:
        landmarks[idx] = [0.40, eye_upper_y, 0.0]

    # Right brow (295, 296, 297, 298) — placed above right eye upper (386, 387)
    for idx in [295, 296, 297, 298]:
        landmarks[idx] = [0.60, brow_y, 0.0]
    for idx in [386, 387]:
        landmarks[idx] = [0.60, eye_upper_y, 0.0]

    return landmarks


class TestGestureAnalyzerLifecycle:
    def test_load_unload(self, config):
        ga = GestureAnalyzer(config=config)
        assert not ga.is_loaded
        ga.load()
        assert ga.is_loaded
        ga.unload()
        assert not ga.is_loaded

    def test_process_without_load_raises(self, config):
        ga = GestureAnalyzer(config=config)
        lm = _make_landmarks()
        with pytest.raises(RuntimeError, match="not loaded"):
            ga.process(0, lm)


class TestNeutralGesture:
    def test_insufficient_history(self, analyzer):
        """With only one frame of history, should return neutral."""
        lm = _make_landmarks()
        result = analyzer.process(person_id=0, landmarks=lm)
        assert isinstance(result, GestureData)
        assert result.gesture_type == GestureType.NEUTRAL.value

    def test_insufficient_landmarks(self, analyzer):
        """Fewer than 300 landmarks should return neutral."""
        short_lm = np.random.rand(100, 3).astype(np.float32)
        result = analyzer.process(person_id=0, landmarks=short_lm)
        assert result.gesture_type == GestureType.NEUTRAL.value

    def test_none_landmarks(self, analyzer):
        result = analyzer.process(person_id=0, landmarks=None)
        assert result.gesture_type == GestureType.NEUTRAL.value

    def test_static_head_neutral(self, analyzer):
        """Head staying perfectly still should remain neutral."""
        for _ in range(10):
            lm = _make_landmarks(nose_x=0.5, nose_y=0.5)
            result = analyzer.process(person_id=0, landmarks=lm)
        assert result.gesture_type == GestureType.NEUTRAL.value


class TestNoddingDetection:
    def test_oscillating_y_values(self, analyzer):
        """Oscillating nose-tip y-values should trigger nod detection."""
        person_id = 1
        # Create a clear up-down-up oscillation with sufficient amplitude
        y_values = [0.5, 0.55, 0.5, 0.45, 0.5, 0.55, 0.5, 0.45, 0.5, 0.55]
        for y in y_values:
            lm = _make_landmarks(nose_x=0.5, nose_y=y)
            result = analyzer.process(person_id=person_id, landmarks=lm)

        assert result.gesture_type == GestureType.NOD.value
        assert result.confidence > 0.0
        assert result.intensity > 0.0


class TestHeadShakeDetection:
    def test_oscillating_x_values(self, analyzer):
        """Oscillating nose-tip x-values (with stable y) should trigger shake."""
        person_id = 2
        x_values = [0.5, 0.55, 0.5, 0.45, 0.5, 0.55, 0.5, 0.45, 0.5, 0.55]
        for x in x_values:
            # Keep y stable so nod does not trigger first
            lm = _make_landmarks(nose_x=x, nose_y=0.5)
            result = analyzer.process(person_id=person_id, landmarks=lm)

        assert result.gesture_type == GestureType.SHAKE.value
        assert result.confidence > 0.0


class TestFrowningDetection:
    def test_depressed_brows(self, analyzer):
        """Brows close to eyes (small gap) should trigger frown detection."""
        person_id = 3
        # Provide enough frames with depressed brows (very small brow-eye gap)
        for _ in range(10):
            lm = _make_landmarks(nose_x=0.5, nose_y=0.5, brow_gap=0.005)
            result = analyzer.process(person_id=person_id, landmarks=lm)

        assert result.gesture_type == GestureType.FROWN.value
        assert result.confidence > 0.0
        assert result.intensity > 0.0


class TestPerPersonHistory:
    def test_separate_histories(self, analyzer):
        """Different person_ids should maintain separate sliding windows."""
        # Feed one person steady data
        for _ in range(5):
            lm = _make_landmarks(nose_x=0.5, nose_y=0.5)
            analyzer.process(person_id=10, landmarks=lm)

        # Feed another person oscillating data
        y_values = [0.5, 0.55, 0.5, 0.45, 0.5, 0.55, 0.5, 0.45, 0.5, 0.55]
        for y in y_values:
            lm = _make_landmarks(nose_x=0.5, nose_y=y)
            result = analyzer.process(person_id=20, landmarks=lm)

        # Person 20 should have nod, person 10 should still be neutral
        assert result.gesture_type == GestureType.NOD.value

        lm_static = _make_landmarks(nose_x=0.5, nose_y=0.5)
        result_10 = analyzer.process(person_id=10, landmarks=lm_static)
        assert result_10.gesture_type == GestureType.NEUTRAL.value
