"""Tests for the PoseEstimator module (T5.6)."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from backend.app.core.vision.config import VisionConfig
from backend.app.core.vision.data_types import PoseData
from backend.app.core.vision.pose_estimator import PoseEstimator


@pytest.fixture
def config():
    return VisionConfig(min_pose_confidence=0.5)


def _make_mock_landmark(x, y, z, visibility):
    lm = MagicMock()
    lm.x = x
    lm.y = y
    lm.z = z
    lm.visibility = visibility
    return lm


class TestPoseEstimatorLifecycle:
    def test_load_unload(self, config):
        import sys

        mock_mp = MagicMock()
        mock_pose_instance = MagicMock()
        mock_mp.solutions.pose.Pose.return_value = mock_pose_instance

        with patch.dict(sys.modules, {"mediapipe": mock_mp}):
            estimator = PoseEstimator(config=config)
            assert not estimator.is_loaded

            estimator.load()
            assert estimator.is_loaded

            estimator.unload()
            assert not estimator.is_loaded
            mock_pose_instance.close.assert_called_once()

    def test_process_without_load_raises(self, config):
        estimator = PoseEstimator(config=config)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.raises(RuntimeError, match="not loaded"):
            estimator.process(frame, (10, 10, 50, 50))


class TestPoseEstimatorProcess:
    def _create_estimator_with_mock(self, config, pose_landmarks):
        """Create PoseEstimator with mocked MediaPipe Pose."""
        estimator = PoseEstimator(config=config)

        mock_pose = MagicMock()
        mock_results = MagicMock()
        mock_results.pose_landmarks = pose_landmarks
        mock_pose.process.return_value = mock_results

        estimator._pose = mock_pose
        estimator._loaded = True
        return estimator

    def test_no_pose_detected(self, config):
        estimator = self._create_estimator_with_mock(config, None)
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        result = estimator.process(frame, (20, 20, 100, 100))

        assert isinstance(result, PoseData)
        assert result.confidence == 0.0

    def test_valid_pose_facing_camera(self, config):
        """Shoulders spread wide should indicate facing camera (low orientation)."""
        mock_landmarks = MagicMock()
        landmarks_list = [MagicMock() for _ in range(33)]

        # Left shoulder (index 11): left side of crop
        landmarks_list[11] = _make_mock_landmark(0.2, 0.5, 0.0, 0.9)
        # Right shoulder (index 12): right side of crop
        landmarks_list[12] = _make_mock_landmark(0.8, 0.5, 0.0, 0.9)

        mock_landmarks.landmark = landmarks_list
        estimator = self._create_estimator_with_mock(config, mock_landmarks)

        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        result = estimator.process(frame, (10, 10, 150, 150))

        assert isinstance(result, PoseData)
        assert result.confidence > 0.5
        assert 0.0 <= result.body_orientation <= 90.0

    def test_low_visibility_shoulders(self, config):
        """Shoulders below min_pose_confidence return low-confidence result."""
        mock_landmarks = MagicMock()
        landmarks_list = [MagicMock() for _ in range(33)]

        landmarks_list[11] = _make_mock_landmark(0.3, 0.5, 0.0, 0.2)
        landmarks_list[12] = _make_mock_landmark(0.7, 0.5, 0.0, 0.2)

        mock_landmarks.landmark = landmarks_list
        estimator = self._create_estimator_with_mock(config, mock_landmarks)

        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        result = estimator.process(frame, (10, 10, 150, 150))

        assert result.confidence < 0.5
        assert result.body_orientation == 0.0

    def test_empty_frame(self, config):
        estimator = self._create_estimator_with_mock(config, None)
        empty_frame = np.array([], dtype=np.uint8)
        result = estimator.process(empty_frame, (0, 0, 10, 10))

        assert result.confidence == 0.0

    def test_none_frame(self, config):
        estimator = self._create_estimator_with_mock(config, None)
        result = estimator.process(None, (0, 0, 10, 10))
        assert result.confidence == 0.0

    def test_shoulder_angle_computed(self, config):
        """Tilted shoulders should produce non-zero shoulder_angle."""
        mock_landmarks = MagicMock()
        landmarks_list = [MagicMock() for _ in range(33)]

        # Left shoulder higher than right (tilted)
        landmarks_list[11] = _make_mock_landmark(0.2, 0.3, 0.0, 0.9)
        landmarks_list[12] = _make_mock_landmark(0.8, 0.6, 0.0, 0.9)

        mock_landmarks.landmark = landmarks_list
        estimator = self._create_estimator_with_mock(config, mock_landmarks)

        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        result = estimator.process(frame, (10, 10, 150, 150))

        assert result.shoulder_angle != 0.0


class TestPadBbox:
    def test_basic_padding(self):
        result = PoseEstimator._pad_bbox((50, 50, 100, 100), (300, 300), padding=0.2)
        x1, y1, x2, y2 = result
        assert x1 < 50
        assert y1 < 50
        assert x2 > 150
        assert y2 > 150

    def test_clamped_to_frame(self):
        result = PoseEstimator._pad_bbox((0, 0, 100, 100), (100, 100), padding=0.5)
        x1, y1, x2, y2 = result
        assert x1 >= 0
        assert y1 >= 0
        assert x2 <= 100
        assert y2 <= 100
