"""Tests for the VisionConfig module (T5.10)."""

import pytest

from backend.app.core.vision.config import VisionConfig


class TestDefaultConfig:
    def test_default_values(self):
        config = VisionConfig()

        # Frame extraction
        assert config.fps == 3.0
        assert config.max_frames == 500

        # Face detection
        assert config.max_faces == 10
        assert config.min_face_confidence == 0.5

        # Emotion classification
        assert isinstance(config.emotion_labels, list)
        assert len(config.emotion_labels) > 0
        assert "happy" in config.emotion_labels
        assert "neutral" in config.emotion_labels
        assert config.clip_model == "openai/clip-vit-base-patch32"

        # Gaze estimation
        assert config.gaze_camera_threshold == 15.0

        # Gesture detection
        assert config.gesture_window_size == 10
        assert config.nod_threshold == 0.03

        # Pose estimation
        assert config.min_pose_confidence == 0.5

        # Person tracking
        assert config.embedding_similarity_threshold == 0.6

        # Device
        assert config.device == "auto"

        # Module toggles
        assert config.enable_emotion is True
        assert config.enable_gaze is True
        assert config.enable_gesture is True
        assert config.enable_pose is True
        assert config.enable_tracking is True


class TestCustomConfig:
    def test_override_all_fields(self):
        config = VisionConfig(
            fps=5.0,
            max_frames=100,
            max_faces=3,
            min_face_confidence=0.8,
            emotion_labels=["happy", "sad"],
            clip_model="custom/model",
            gaze_camera_threshold=20.0,
            gesture_window_size=15,
            nod_threshold=0.05,
            min_pose_confidence=0.7,
            embedding_similarity_threshold=0.8,
            device="cpu",
            enable_emotion=False,
            enable_gaze=False,
            enable_gesture=False,
            enable_pose=False,
            enable_tracking=False,
        )

        assert config.fps == 5.0
        assert config.max_frames == 100
        assert config.max_faces == 3
        assert config.min_face_confidence == 0.8
        assert config.emotion_labels == ["happy", "sad"]
        assert config.clip_model == "custom/model"
        assert config.gaze_camera_threshold == 20.0
        assert config.gesture_window_size == 15
        assert config.nod_threshold == 0.05
        assert config.min_pose_confidence == 0.7
        assert config.embedding_similarity_threshold == 0.8
        assert config.device == "cpu"
        assert config.enable_emotion is False
        assert config.enable_gaze is False
        assert config.enable_gesture is False
        assert config.enable_pose is False
        assert config.enable_tracking is False

    def test_partial_override(self):
        config = VisionConfig(fps=10.0, enable_emotion=False)
        assert config.fps == 10.0
        assert config.enable_emotion is False
        # Rest should be defaults
        assert config.max_frames == 500
        assert config.enable_gaze is True


class TestEmotionLabelsDefault:
    def test_default_labels_not_shared(self):
        """Each VisionConfig instance should get its own emotion_labels list."""
        config1 = VisionConfig()
        config2 = VisionConfig()
        config1.emotion_labels.append("custom_emotion")
        assert "custom_emotion" not in config2.emotion_labels

    def test_default_labels_content(self):
        config = VisionConfig()
        expected = [
            "happy", "sad", "angry", "surprised", "fearful",
            "disgusted", "contemptuous", "neutral", "attentive",
        ]
        assert config.emotion_labels == expected
