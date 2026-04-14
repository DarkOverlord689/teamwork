"""Tests for the FaceDetector module (T5.2)."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from backend.app.core.vision.config import VisionConfig
from backend.app.core.vision.data_types import PersonFrame
from backend.app.core.vision.face_detector import FaceDetector


@pytest.fixture
def config():
    return VisionConfig(max_faces=5, min_face_confidence=0.5)


class TestFaceDetectorLifecycle:
    def test_load_unload(self, config):
        import sys

        mock_mp = MagicMock()
        mock_mesh_instance = MagicMock()
        mock_mp.solutions.face_mesh.FaceMesh.return_value = mock_mesh_instance

        with patch.dict(sys.modules, {"mediapipe": mock_mp}):
            detector = FaceDetector(config=config, device="cpu")
            assert not detector.is_loaded

            detector.load()
            assert detector.is_loaded
            mock_mp.solutions.face_mesh.FaceMesh.assert_called_once()

            detector.unload()
            assert not detector.is_loaded
            mock_mesh_instance.close.assert_called_once()

    def test_process_without_load_raises(self, config):
        detector = FaceDetector(config=config, device="cpu")
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.raises(RuntimeError, match="not been loaded"):
            detector.process(frame)


class TestFaceDetectorProcess:
    def _make_mock_landmark(self, x, y, z):
        lm = MagicMock()
        lm.x = x
        lm.y = y
        lm.z = z
        return lm

    def _create_detector_with_mock(self, config, face_landmarks_list):
        """Create a FaceDetector with a mocked MediaPipe FaceMesh."""
        detector = FaceDetector(config=config, device="cpu")

        mock_mesh = MagicMock()
        mock_results = MagicMock()
        mock_results.multi_face_landmarks = face_landmarks_list
        mock_mesh.process.return_value = mock_results

        detector._face_mesh = mock_mesh
        detector._loaded = True
        return detector

    def test_no_faces_blank_image(self, config):
        detector = self._create_detector_with_mock(config, None)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = detector.process(frame)
        assert result == []

    def test_single_face_detected(self, config):
        # Create 478 landmarks forming a face-like region
        mock_face = MagicMock()
        landmarks = []
        for i in range(478):
            # Place landmarks in the centre of a 200x200 image
            lm = self._make_mock_landmark(
                x=0.3 + (i % 20) * 0.02,
                y=0.3 + (i // 20) * 0.02,
                z=0.0,
            )
            landmarks.append(lm)
        mock_face.landmark = landmarks

        detector = self._create_detector_with_mock(config, [mock_face])
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        result = detector.process(frame)

        assert len(result) == 1
        person = result[0]
        assert isinstance(person, PersonFrame)
        assert person.person_id == "unassigned"  # unassigned
        assert person.landmarks is not None
        assert person.landmarks.shape == (478, 3)
        # bbox should be a 4-tuple
        assert len(person.bbox) == 4
        x, y, w, h = person.bbox
        assert w > 0 and h > 0

    def test_multiple_faces_detected(self, config):
        faces = []
        for offset in [0.2, 0.6]:
            mock_face = MagicMock()
            landmarks = []
            for i in range(478):
                lm = self._make_mock_landmark(
                    x=offset + (i % 10) * 0.01,
                    y=offset + (i // 10) * 0.005,
                    z=0.0,
                )
                landmarks.append(lm)
            mock_face.landmark = landmarks
            faces.append(mock_face)

        detector = self._create_detector_with_mock(config, faces)
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        result = detector.process(frame)

        assert len(result) == 2
        for person in result:
            assert isinstance(person, PersonFrame)
            assert person.landmarks.shape == (478, 3)

    def test_degenerate_bbox_skipped(self, config):
        """Face where all landmarks are at the same point -> zero-area bbox."""
        mock_face = MagicMock()
        landmarks = [self._make_mock_landmark(0.5, 0.5, 0.0) for _ in range(478)]
        mock_face.landmark = landmarks

        detector = self._create_detector_with_mock(config, [mock_face])
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        result = detector.process(frame)

        assert result == []
