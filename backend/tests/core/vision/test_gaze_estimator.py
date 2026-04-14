"""Tests for the GazeEstimator module (T5.3)."""

import numpy as np
import pytest

from backend.app.core.vision.config import VisionConfig
from backend.app.core.vision.data_types import GazeCategory, GazeData
from backend.app.core.vision.gaze_estimator import GazeEstimator


@pytest.fixture
def config():
    return VisionConfig(gaze_camera_threshold=15.0)


@pytest.fixture
def estimator(config):
    est = GazeEstimator(config=config)
    est.load()
    yield est
    est.unload()


def _make_landmarks(num=478):
    """Create synthetic 478x3 landmarks with iris at centre of eye sockets."""
    landmarks = np.random.rand(num, 3).astype(np.float32) * 0.1 + 0.45

    # Eye corners (normalised coords ~0.3-0.7 range)
    # Left eye: outer=33, inner=133
    landmarks[33] = [0.35, 0.45, 0.0]   # left eye outer
    landmarks[133] = [0.45, 0.45, 0.0]  # left eye inner
    # Right eye: inner=362, outer=263
    landmarks[362] = [0.55, 0.45, 0.0]  # right eye inner
    landmarks[263] = [0.65, 0.45, 0.0]  # right eye outer

    # Eye upper/lower for vertical ratio
    landmarks[159] = [0.40, 0.43, 0.0]  # left eye top
    landmarks[145] = [0.40, 0.47, 0.0]  # left eye bottom
    landmarks[386] = [0.60, 0.43, 0.0]  # right eye top
    landmarks[374] = [0.60, 0.47, 0.0]  # right eye bottom

    return landmarks


def _place_iris(landmarks, h_ratio=0.5, v_ratio=0.5):
    """Place iris landmarks based on position within the eye socket.

    h_ratio=0.5 means centred horizontally (looking at camera).
    v_ratio=0.5 means centred vertically.
    """
    lm = landmarks.copy()

    # Left iris (468-472): interpolate between outer(33) and inner(133)
    left_outer = lm[33]
    left_inner = lm[133]
    left_iris_pos = left_outer + h_ratio * (left_inner - left_outer)
    # Vertical: interpolate between top(159) and bottom(145)
    left_top = lm[159]
    left_bot = lm[145]
    left_iris_pos[1] = left_top[1] + v_ratio * (left_bot[1] - left_top[1])
    for i in range(468, 473):
        lm[i] = left_iris_pos + np.random.rand(3) * 0.001

    # Right iris (473-477): interpolate between inner(362) and outer(263)
    right_inner = lm[362]
    right_outer = lm[263]
    right_iris_pos = right_inner + h_ratio * (right_outer - right_inner)
    right_top = lm[386]
    right_bot = lm[374]
    right_iris_pos[1] = right_top[1] + v_ratio * (right_bot[1] - right_top[1])
    for i in range(473, 478):
        lm[i] = right_iris_pos + np.random.rand(3) * 0.001

    return lm


class TestGazeEstimatorLifecycle:
    def test_load_unload(self, config):
        est = GazeEstimator(config=config)
        assert not est.is_loaded
        est.load()
        assert est.is_loaded
        est.unload()
        assert not est.is_loaded

    def test_process_without_load_raises(self, config):
        est = GazeEstimator(config=config)
        lm = _make_landmarks()
        with pytest.raises(RuntimeError, match="not loaded"):
            est.process(lm, (480, 640))


class TestGazeEstimation:
    def test_camera_gaze_centred_iris(self, estimator):
        """Iris centred in the eye socket should classify as CAMERA."""
        landmarks = _make_landmarks()
        landmarks = _place_iris(landmarks, h_ratio=0.5, v_ratio=0.5)
        result = estimator.process(landmarks, (480, 640))

        assert isinstance(result, GazeData)
        assert result.category == GazeCategory.CAMERA.value
        assert result.is_looking_at_camera is True
        assert result.confidence > 0.0

    def test_away_gaze_iris_at_edge(self, estimator):
        """Iris positioned at extreme outer edge -> not looking at camera."""
        landmarks = _make_landmarks()
        # Push iris to extreme outer position
        landmarks = _place_iris(landmarks, h_ratio=0.05, v_ratio=0.5)
        result = estimator.process(landmarks, (480, 640))

        assert isinstance(result, GazeData)
        assert result.is_looking_at_camera is False
        assert result.category != GazeCategory.CAMERA.value

    def test_insufficient_landmarks(self, estimator):
        """Fewer than 478 landmarks should return UNKNOWN with zero confidence."""
        short_landmarks = np.random.rand(100, 3).astype(np.float32)
        result = estimator.process(short_landmarks, (480, 640))

        assert result.category == GazeCategory.UNKNOWN.value
        assert result.confidence == 0.0
        assert result.is_looking_at_camera is False

    def test_none_landmarks(self, estimator):
        result = estimator.process(None, (480, 640))
        assert result.category == GazeCategory.UNKNOWN.value
        assert result.confidence == 0.0

    def test_direction_tuple_format(self, estimator):
        landmarks = _make_landmarks()
        landmarks = _place_iris(landmarks, h_ratio=0.5, v_ratio=0.5)
        result = estimator.process(landmarks, (480, 640))

        assert isinstance(result.direction, tuple)
        assert len(result.direction) == 3
        yaw, pitch, roll = result.direction
        assert isinstance(yaw, (float, int, np.floating))
        assert isinstance(pitch, (float, int, np.floating))
        assert isinstance(roll, (float, int, np.floating))

    def test_downward_gaze_screen(self, estimator):
        """Iris positioned low in the eye -> looking down (screen)."""
        landmarks = _make_landmarks()
        landmarks = _place_iris(landmarks, h_ratio=0.5, v_ratio=0.95)
        result = estimator.process(landmarks, (480, 640))

        assert result.is_looking_at_camera is False
