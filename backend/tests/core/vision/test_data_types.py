"""Tests for the data_types module (T5.9)."""

import numpy as np
import pytest

from backend.app.core.vision.data_types import (
    EmotionData,
    FrameResult,
    GazeCategory,
    GazeData,
    GestureData,
    GestureType,
    PersonFrame,
    PoseData,
    VisionResult,
)


class TestEnums:
    def test_gaze_category_values(self):
        assert GazeCategory.CAMERA.value == "camera"
        assert GazeCategory.SCREEN.value == "screen"
        assert GazeCategory.PEER.value == "peer"
        assert GazeCategory.AWAY.value == "away"
        assert GazeCategory.UNKNOWN.value == "unknown"

    def test_gaze_category_is_str(self):
        assert isinstance(GazeCategory.CAMERA, str)
        assert GazeCategory.CAMERA == "camera"

    def test_gesture_type_values(self):
        assert GestureType.NOD.value == "nod"
        assert GestureType.SHAKE.value == "shake"
        assert GestureType.FROWN.value == "frown"
        assert GestureType.NEUTRAL.value == "neutral"

    def test_gesture_type_is_str(self):
        assert isinstance(GestureType.NOD, str)
        assert GestureType.NOD == "nod"


class TestGazeData:
    def test_creation(self):
        gaze = GazeData(
            direction=(10.0, 5.0, 0.0),
            is_looking_at_camera=True,
            confidence=0.95,
            category="camera",
        )
        assert gaze.direction == (10.0, 5.0, 0.0)
        assert gaze.is_looking_at_camera is True
        assert gaze.confidence == 0.95
        assert gaze.category == "camera"

    def test_to_dict(self):
        gaze = GazeData(
            direction=(10.0, 5.0, 0.0),
            is_looking_at_camera=True,
            confidence=0.95,
            category="camera",
        )
        d = gaze.to_dict()
        assert d["direction"] == [10.0, 5.0, 0.0]
        assert d["is_looking_at_camera"] is True
        assert d["confidence"] == 0.95
        assert d["category"] == "camera"


class TestGestureData:
    def test_creation(self):
        gesture = GestureData(gesture_type="nod", confidence=0.8, intensity=0.6)
        assert gesture.gesture_type == "nod"
        assert gesture.confidence == 0.8
        assert gesture.intensity == 0.6

    def test_to_dict(self):
        gesture = GestureData(gesture_type="shake", confidence=0.7, intensity=0.5)
        d = gesture.to_dict()
        assert d == {"gesture_type": "shake", "confidence": 0.7, "intensity": 0.5}


class TestPoseData:
    def test_creation(self):
        pose = PoseData(body_orientation=15.0, shoulder_angle=3.0, confidence=0.9)
        assert pose.body_orientation == 15.0
        assert pose.shoulder_angle == 3.0
        assert pose.confidence == 0.9

    def test_to_dict(self):
        pose = PoseData(body_orientation=15.0, shoulder_angle=3.0, confidence=0.9)
        d = pose.to_dict()
        assert d == {"body_orientation": 15.0, "shoulder_angle": 3.0, "confidence": 0.9}


class TestEmotionData:
    def test_creation(self):
        emotion = EmotionData(
            primary_emotion="happy",
            confidence=0.85,
            all_emotions={"happy": 0.85, "sad": 0.1, "neutral": 0.05},
        )
        assert emotion.primary_emotion == "happy"
        assert emotion.confidence == 0.85
        assert len(emotion.all_emotions) == 3

    def test_to_dict(self):
        emotion = EmotionData(
            primary_emotion="angry",
            confidence=0.6,
            all_emotions={"angry": 0.6, "neutral": 0.4},
        )
        d = emotion.to_dict()
        assert d["primary_emotion"] == "angry"
        assert d["confidence"] == 0.6
        assert d["all_emotions"] == {"angry": 0.6, "neutral": 0.4}


class TestPersonFrame:
    def test_creation_minimal(self):
        person = PersonFrame(person_id="p1", bbox=(10, 20, 100, 150))
        assert person.person_id == "p1"
        assert person.bbox == (10, 20, 100, 150)
        assert person.landmarks is None
        assert person.gaze is None
        assert person.gesture is None
        assert person.pose is None
        assert person.emotion is None
        assert person.face_embedding is None

    def test_creation_full(self):
        landmarks = np.random.rand(478, 3).astype(np.float32)
        embedding = np.random.rand(512).astype(np.float32)
        gaze = GazeData(direction=(0, 0, 0), is_looking_at_camera=True,
                        confidence=0.9, category="camera")
        gesture = GestureData(gesture_type="nod", confidence=0.8, intensity=0.5)
        pose = PoseData(body_orientation=10.0, shoulder_angle=2.0, confidence=0.85)
        emotion = EmotionData(primary_emotion="happy", confidence=0.9,
                              all_emotions={"happy": 0.9, "neutral": 0.1})

        person = PersonFrame(
            person_id="p2",
            bbox=(50, 50, 80, 120),
            landmarks=landmarks,
            gaze=gaze,
            gesture=gesture,
            pose=pose,
            emotion=emotion,
            face_embedding=embedding,
        )
        assert person.landmarks.shape == (478, 3)
        assert person.face_embedding.shape == (512,)

    def test_to_dict_minimal(self):
        person = PersonFrame(person_id="p1", bbox=(10, 20, 30, 40))
        d = person.to_dict()
        assert d["person_id"] == "p1"
        assert d["bbox"] == [10, 20, 30, 40]
        assert "landmarks" not in d
        assert "gaze" not in d

    def test_to_dict_full(self):
        landmarks = np.ones((3, 3), dtype=np.float32)
        embedding = np.ones(5, dtype=np.float32)
        gaze = GazeData(direction=(1, 2, 3), is_looking_at_camera=False,
                        confidence=0.5, category="away")

        person = PersonFrame(
            person_id="p3",
            bbox=(0, 0, 10, 10),
            landmarks=landmarks,
            gaze=gaze,
            face_embedding=embedding,
        )
        d = person.to_dict()
        assert "landmarks" in d
        assert "gaze" in d
        assert "face_embedding" in d
        assert d["gaze"]["category"] == "away"


class TestFrameResult:
    def test_creation(self):
        fr = FrameResult(frame_number=5, timestamp=1.5)
        assert fr.frame_number == 5
        assert fr.timestamp == 1.5
        assert fr.persons == []

    def test_to_dict(self):
        person = PersonFrame(person_id="p1", bbox=(0, 0, 10, 10))
        fr = FrameResult(frame_number=10, timestamp=3.0, persons=[person])
        d = fr.to_dict()
        assert d["frame_number"] == 10
        assert d["timestamp"] == 3.0
        assert len(d["persons"]) == 1


class TestVisionResult:
    def test_creation(self):
        vr = VisionResult(
            video_path="test.mp4",
            total_frames=100,
            fps_processed=3.0,
            duration_seconds=10.0,
        )
        assert vr.video_path == "test.mp4"
        assert vr.total_frames == 100
        assert vr.frames == []
        assert vr.processing_time_seconds == 0.0

    def test_to_dict(self):
        person = PersonFrame(person_id="p1", bbox=(0, 0, 10, 10))
        frame = FrameResult(frame_number=0, timestamp=0.0, persons=[person])
        vr = VisionResult(
            video_path="test.mp4",
            total_frames=50,
            fps_processed=2.0,
            duration_seconds=5.0,
            frames=[frame],
            processing_time_seconds=1.23,
        )
        d = vr.to_dict()
        assert d["video_path"] == "test.mp4"
        assert d["total_frames"] == 50
        assert d["fps_processed"] == 2.0
        assert d["duration_seconds"] == 5.0
        assert d["processing_time_seconds"] == 1.23
        assert len(d["frames"]) == 1

    def test_to_dict_serializable(self):
        """Verify to_dict output is JSON-serializable (no numpy arrays)."""
        import json

        landmarks = np.random.rand(5, 3).astype(np.float32)
        embedding = np.random.rand(10).astype(np.float32)
        person = PersonFrame(
            person_id="p1",
            bbox=(0, 0, 10, 10),
            landmarks=landmarks,
            face_embedding=embedding,
        )
        frame = FrameResult(frame_number=0, timestamp=0.0, persons=[person])
        vr = VisionResult(
            video_path="test.mp4",
            total_frames=10,
            fps_processed=1.0,
            duration_seconds=1.0,
            frames=[frame],
        )
        d = vr.to_dict()
        # This should not raise
        json_str = json.dumps(d)
        assert isinstance(json_str, str)
