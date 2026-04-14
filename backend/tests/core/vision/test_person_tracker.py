"""Tests for the PersonTracker module (T5.7)."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from backend.app.core.vision.config import VisionConfig
from backend.app.core.vision.data_types import PersonFrame
from backend.app.core.vision.person_tracker import PersonTracker, _iou


@pytest.fixture
def config():
    return VisionConfig(embedding_similarity_threshold=0.6)


def _create_tracker_with_mock(config):
    """Create a PersonTracker with a mocked FaceNet model."""
    tracker = PersonTracker(config=config, device="cpu")

    mock_model = MagicMock()
    # Model returns a random 512-d embedding when called
    def mock_forward(tensor):
        # Return a deterministic embedding based on the input mean
        # so the same face crop produces the same embedding
        seed = int(tensor.mean().item() * 1000) % 2**31
        rng = np.random.RandomState(seed)
        emb = rng.randn(1, 512).astype(np.float32)
        return torch.from_numpy(emb)

    mock_model.side_effect = mock_forward
    mock_model.__call__ = mock_forward
    mock_model.to = MagicMock(return_value=mock_model)
    mock_model.eval = MagicMock(return_value=mock_model)

    tracker._model = mock_model
    tracker._loaded = True

    return tracker


class TestIoU:
    def test_perfect_overlap(self):
        assert _iou((10, 10, 50, 50), (10, 10, 50, 50)) == pytest.approx(1.0)

    def test_no_overlap(self):
        assert _iou((0, 0, 10, 10), (100, 100, 10, 10)) == 0.0

    def test_partial_overlap(self):
        iou_val = _iou((0, 0, 20, 20), (10, 10, 20, 20))
        assert 0.0 < iou_val < 1.0

    def test_zero_area_box(self):
        assert _iou((10, 10, 0, 0), (10, 10, 50, 50)) == 0.0


class TestPersonTrackerLifecycle:
    def test_process_without_load_raises(self, config):
        tracker = PersonTracker(config=config, device="cpu")
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        persons = [PersonFrame(person_id="unassigned", bbox=(10, 10, 30, 30))]
        with pytest.raises(RuntimeError, match="not been loaded"):
            tracker.process(frame, persons)

    def test_load_unload(self, config):
        import sys

        mock_facenet = MagicMock()
        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_facenet.InceptionResnetV1.return_value = mock_model

        with patch.dict(sys.modules, {"facenet_pytorch": mock_facenet}):
            tracker = PersonTracker(config=config, device="cpu")
            assert not tracker.is_loaded

            tracker.load()
            assert tracker.is_loaded

            tracker.unload()
            assert not tracker.is_loaded
            assert tracker.next_id == 0
            assert len(tracker.known_embeddings) == 0


class TestPersonTrackerProcess:
    def test_new_person_assignment(self, config):
        """First person seen should get person_id 0."""
        tracker = _create_tracker_with_mock(config)
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        persons = [PersonFrame(person_id="unassigned", bbox=(10, 10, 30, 30))]

        result = tracker.process(frame, persons)

        assert len(result) == 1
        assert result[0].person_id.startswith("person_")

    def test_multiple_persons(self, config):
        """Multiple persons in the same frame should get different IDs."""
        tracker = _create_tracker_with_mock(config)

        # Create two persons at different locations with different pixel values
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        frame[10:50, 10:50] = 200  # bright patch for person 1
        frame[100:150, 100:150] = 50  # dim patch for person 2

        persons = [
            PersonFrame(person_id="unassigned", bbox=(10, 10, 40, 40)),
            PersonFrame(person_id="unassigned", bbox=(100, 100, 50, 50)),
        ]

        result = tracker.process(frame, persons)

        assert len(result) == 2
        ids = {p.person_id for p in result}
        assert len(ids) == 2  # different IDs

    def test_reid_same_person_across_frames(self, config):
        """Same face crop across two frames should get the same person_id."""
        tracker = _create_tracker_with_mock(config)

        # Use identical frame and bbox to produce same embedding
        frame = np.full((100, 100, 3), fill_value=128, dtype=np.uint8)
        bbox = (10, 10, 40, 40)

        persons_f1 = [PersonFrame(person_id="unassigned", bbox=bbox)]
        result_f1 = tracker.process(frame, persons_f1)
        id_f1 = result_f1[0].person_id

        persons_f2 = [PersonFrame(person_id="unassigned", bbox=bbox)]
        result_f2 = tracker.process(frame, persons_f2)
        id_f2 = result_f2[0].person_id

        assert id_f1 == id_f2

    def test_iou_fallback(self, config):
        """When embedding fails, tracker should fall back to IoU matching."""
        tracker = _create_tracker_with_mock(config)

        # Override _get_embedding to return None (simulating failure)
        tracker._get_embedding = MagicMock(return_value=None)

        frame = np.zeros((200, 200, 3), dtype=np.uint8)

        # First frame: person at position A
        persons_f1 = [PersonFrame(person_id="unassigned", bbox=(10, 10, 50, 50))]
        result_f1 = tracker.process(frame, persons_f1)
        id_f1 = result_f1[0].person_id

        # Second frame: person at nearby position (high IoU)
        persons_f2 = [PersonFrame(person_id="unassigned", bbox=(12, 12, 50, 50))]
        result_f2 = tracker.process(frame, persons_f2)
        id_f2 = result_f2[0].person_id

        assert id_f1 == id_f2

    def test_embedding_stored(self, config):
        """After processing, face_embedding should be populated."""
        tracker = _create_tracker_with_mock(config)
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        persons = [PersonFrame(person_id="unassigned", bbox=(10, 10, 30, 30))]

        result = tracker.process(frame, persons)

        assert result[0].face_embedding is not None
        assert result[0].face_embedding.shape == (512,)


class TestCosineSimlarity:
    def test_identical_vectors(self, config):
        tracker = PersonTracker(config=config, device="cpu")
        v = np.array([1.0, 2.0, 3.0])
        assert tracker._cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self, config):
        tracker = PersonTracker(config=config, device="cpu")
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert tracker._cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_zero_vector(self, config):
        tracker = PersonTracker(config=config, device="cpu")
        a = np.array([1.0, 2.0])
        b = np.zeros(2)
        assert tracker._cosine_similarity(a, b) == 0.0
