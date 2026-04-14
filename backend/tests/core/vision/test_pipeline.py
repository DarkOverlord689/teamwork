"""Tests for the VisionPipeline module (T5.8)."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from backend.app.core.vision.config import VisionConfig
from backend.app.core.vision.data_types import (
    EmotionData,
    FrameResult,
    GazeData,
    GestureData,
    PersonFrame,
    PoseData,
    VisionResult,
)
from backend.app.core.vision.pipeline import VisionPipeline


@pytest.fixture
def minimal_config():
    """Config with all optional modules disabled."""
    return VisionConfig(
        enable_emotion=False,
        enable_gaze=False,
        enable_gesture=False,
        enable_pose=False,
        enable_tracking=False,
    )


@pytest.fixture
def full_config():
    """Config with all modules enabled."""
    return VisionConfig()


class TestPipelineInit:
    def test_minimal_config_no_optional_processors(self, minimal_config):
        pipeline = VisionPipeline(config=minimal_config)
        assert pipeline._emotion_classifier is None
        assert pipeline._gaze_estimator is None
        assert pipeline._gesture_analyzer is None
        assert pipeline._pose_estimator is None
        assert pipeline._person_tracker is None

    def test_full_config_all_processors(self, full_config):
        pipeline = VisionPipeline(config=full_config)
        assert pipeline._emotion_classifier is not None
        assert pipeline._gaze_estimator is not None
        assert pipeline._gesture_analyzer is not None
        assert pipeline._pose_estimator is not None
        assert pipeline._person_tracker is not None

    def test_selective_modules(self):
        config = VisionConfig(
            enable_emotion=True,
            enable_gaze=False,
            enable_gesture=True,
            enable_pose=False,
            enable_tracking=False,
        )
        pipeline = VisionPipeline(config=config)
        assert pipeline._emotion_classifier is not None
        assert pipeline._gaze_estimator is None
        assert pipeline._gesture_analyzer is not None
        assert pipeline._pose_estimator is None
        assert pipeline._person_tracker is None


class TestPipelineLifecycle:
    def test_context_manager(self, minimal_config):
        """Context manager should call load_all and unload_all."""
        pipeline = VisionPipeline(config=minimal_config)

        with patch.object(pipeline, "load_all") as mock_load, \
             patch.object(pipeline, "unload_all") as mock_unload:
            with pipeline:
                mock_load.assert_called_once()
            mock_unload.assert_called_once()

    def test_load_all_unload_all(self, minimal_config):
        """load_all loads frame_extractor and face_detector at minimum."""
        pipeline = VisionPipeline(config=minimal_config)

        with patch.object(pipeline._frame_extractor, "load") as fe_load, \
             patch.object(pipeline._face_detector, "load") as fd_load, \
             patch.object(pipeline._frame_extractor, "unload") as fe_unload, \
             patch.object(pipeline._face_detector, "unload") as fd_unload:

            # Mark as loaded for unload_all to call unload
            pipeline._frame_extractor._loaded = True
            pipeline._face_detector._loaded = True

            pipeline.load_all()
            fe_load.assert_called_once()
            fd_load.assert_called_once()

            pipeline.unload_all()
            fe_unload.assert_called_once()
            fd_unload.assert_called_once()


class TestPipelineProcessVideo:
    def _create_mocked_pipeline(self, config):
        """Create a pipeline with all sub-processors mocked."""
        pipeline = VisionPipeline(config=config)

        # Mock frame extractor
        pipeline._frame_extractor = MagicMock()
        pipeline._frame_extractor.is_loaded = True
        pipeline._frame_extractor.get_video_properties.return_value = (
            30, 3.0, 10.0  # total_frames, duration, native_fps
        )
        # Return 3 frames
        frames = [
            (0, 0.0, np.zeros((100, 100, 3), dtype=np.uint8)),
            (10, 1.0, np.zeros((100, 100, 3), dtype=np.uint8)),
            (20, 2.0, np.zeros((100, 100, 3), dtype=np.uint8)),
        ]
        pipeline._frame_extractor.process.return_value = frames

        # Mock face detector
        pipeline._face_detector = MagicMock()
        pipeline._face_detector.is_loaded = True
        pipeline._face_detector.process.return_value = [
            PersonFrame(
                person_id="unassigned",
                bbox=(10, 10, 30, 30),
                landmarks=np.random.rand(478, 3).astype(np.float32),
            )
        ]

        return pipeline

    def test_basic_flow_no_optional_modules(self, minimal_config):
        pipeline = self._create_mocked_pipeline(minimal_config)

        result = pipeline.process_video("fake_video.mp4")

        assert isinstance(result, VisionResult)
        assert result.video_path == "fake_video.mp4"
        assert result.total_frames == 30
        assert result.duration_seconds == 3.0
        assert result.fps_processed == minimal_config.fps
        assert len(result.frames) == 3
        assert result.processing_time_seconds >= 0

    def test_frame_results_structure(self, minimal_config):
        pipeline = self._create_mocked_pipeline(minimal_config)
        result = pipeline.process_video("fake_video.mp4")

        for fr in result.frames:
            assert isinstance(fr, FrameResult)
            assert isinstance(fr.frame_number, int)
            assert isinstance(fr.timestamp, float)
            assert isinstance(fr.persons, list)

    def test_no_faces_detected(self, minimal_config):
        pipeline = self._create_mocked_pipeline(minimal_config)
        pipeline._face_detector.process.return_value = []

        result = pipeline.process_video("fake_video.mp4")

        assert len(result.frames) == 3
        for fr in result.frames:
            assert len(fr.persons) == 0

    def test_with_tracker_enabled(self):
        config = VisionConfig(
            enable_emotion=False, enable_gaze=False,
            enable_gesture=False, enable_pose=False,
            enable_tracking=True,
        )
        pipeline = self._create_mocked_pipeline(config)

        # Mock tracker
        mock_tracker = MagicMock()
        mock_tracker.is_loaded = True
        mock_tracker.process.side_effect = lambda frame, persons: persons
        pipeline._person_tracker = mock_tracker

        result = pipeline.process_video("fake_video.mp4")

        assert mock_tracker.process.call_count == 3  # once per frame

    def test_with_emotion_enabled(self):
        config = VisionConfig(
            enable_emotion=True, enable_gaze=False,
            enable_gesture=False, enable_pose=False,
            enable_tracking=False,
        )
        pipeline = self._create_mocked_pipeline(config)

        mock_emotion = MagicMock()
        mock_emotion.is_loaded = True
        mock_emotion.process.return_value = EmotionData(
            primary_emotion="happy", confidence=0.9,
            all_emotions={"happy": 0.9, "neutral": 0.1},
        )
        pipeline._emotion_classifier = mock_emotion

        result = pipeline.process_video("fake_video.mp4")

        assert mock_emotion.process.call_count == 3
        for fr in result.frames:
            for person in fr.persons:
                assert person.emotion is not None
                assert person.emotion.primary_emotion == "happy"

    def test_with_gaze_enabled(self):
        config = VisionConfig(
            enable_emotion=False, enable_gaze=True,
            enable_gesture=False, enable_pose=False,
            enable_tracking=False,
        )
        pipeline = self._create_mocked_pipeline(config)

        mock_gaze = MagicMock()
        mock_gaze.is_loaded = True
        mock_gaze.process.return_value = GazeData(
            direction=(0.0, 0.0, 0.0), is_looking_at_camera=True,
            confidence=0.9, category="camera",
        )
        pipeline._gaze_estimator = mock_gaze

        result = pipeline.process_video("fake_video.mp4")
        assert mock_gaze.process.call_count == 3

    def test_with_gesture_enabled(self):
        config = VisionConfig(
            enable_emotion=False, enable_gaze=False,
            enable_gesture=True, enable_pose=False,
            enable_tracking=False,
        )
        pipeline = self._create_mocked_pipeline(config)

        mock_gesture = MagicMock()
        mock_gesture.is_loaded = True
        mock_gesture.process.return_value = GestureData(
            gesture_type="neutral", confidence=0.0, intensity=0.0,
        )
        pipeline._gesture_analyzer = mock_gesture

        result = pipeline.process_video("fake_video.mp4")
        assert mock_gesture.process.call_count == 3

    def test_with_pose_enabled(self):
        config = VisionConfig(
            enable_emotion=False, enable_gaze=False,
            enable_gesture=False, enable_pose=True,
            enable_tracking=False,
        )
        pipeline = self._create_mocked_pipeline(config)

        mock_pose = MagicMock()
        mock_pose.is_loaded = True
        mock_pose.process.return_value = PoseData(
            body_orientation=10.0, shoulder_angle=5.0, confidence=0.8,
        )
        pipeline._pose_estimator = mock_pose

        result = pipeline.process_video("fake_video.mp4")
        assert mock_pose.process.call_count == 3


class TestExtractFaceCrop:
    def test_valid_crop(self):
        frame = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        crop = VisionPipeline._extract_face_crop(frame, (10, 10, 50, 50))
        assert crop is not None
        assert crop.shape == (50, 50, 3)

    def test_crop_clamped_to_frame(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        crop = VisionPipeline._extract_face_crop(frame, (80, 80, 50, 50))
        assert crop is not None
        assert crop.shape[0] <= 100
        assert crop.shape[1] <= 100

    def test_empty_crop_returns_none(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        crop = VisionPipeline._extract_face_crop(frame, (200, 200, 10, 10))
        assert crop is None
