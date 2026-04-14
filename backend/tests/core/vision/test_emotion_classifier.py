"""Tests for the EmotionClassifier module (T5.4)."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from backend.app.core.vision.config import VisionConfig
from backend.app.core.vision.data_types import EmotionData
from backend.app.core.vision.emotion_classifier import EmotionClassifier


@pytest.fixture
def config():
    return VisionConfig(
        emotion_labels=["happy", "sad", "angry", "neutral"],
        clip_model="openai/clip-vit-base-patch32",
    )


def _create_mock_classifier(config):
    """Create an EmotionClassifier with mocked CLIP model and processor."""
    classifier = EmotionClassifier(config=config)

    # Mock the CLIP model
    mock_model = MagicMock()
    # Simulate logits output: 4 emotion labels
    mock_outputs = MagicMock()
    # Create fake logits that make "happy" the dominant emotion
    mock_outputs.logits_per_image = torch.tensor([[2.5, 0.5, 0.3, 0.7]])
    mock_model.return_value = mock_outputs
    mock_model.__call__ = MagicMock(return_value=mock_outputs)

    # Mock the processor
    mock_processor = MagicMock()
    mock_processor.return_value = {
        "input_ids": torch.zeros(1, 10, dtype=torch.long),
        "attention_mask": torch.ones(1, 10, dtype=torch.long),
        "pixel_values": torch.zeros(1, 3, 224, 224),
    }

    classifier._model = mock_model
    classifier._processor = mock_processor
    classifier._text_prompts = [
        f"a photo of a person with {e} expression"
        for e in config.emotion_labels
    ]
    classifier._loaded = True

    return classifier


class TestEmotionClassifierLifecycle:
    def test_not_loaded_raises(self, config):
        classifier = EmotionClassifier(config=config)
        face_crop = np.zeros((64, 64, 3), dtype=np.uint8)
        with pytest.raises(RuntimeError, match="not loaded"):
            classifier.process(face_crop)

    @patch("transformers.CLIPModel")
    @patch("transformers.CLIPProcessor")
    def test_load_unload(self, mock_proc_cls, mock_model_cls, config):
        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_proc_cls.from_pretrained.return_value = MagicMock()

        classifier = EmotionClassifier(config=config)
        assert not classifier.is_loaded

        classifier.load()
        assert classifier.is_loaded
        mock_model_cls.from_pretrained.assert_called_once_with(config.clip_model)

        classifier.unload()
        assert not classifier.is_loaded


class TestEmotionClassifierProcess:
    def test_returns_emotion_data(self, config):
        classifier = _create_mock_classifier(config)
        face_crop = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = classifier.process(face_crop)

        assert isinstance(result, EmotionData)
        assert result.primary_emotion in config.emotion_labels
        assert 0.0 <= result.confidence <= 1.0

    def test_all_emotions_sum_to_one(self, config):
        classifier = _create_mock_classifier(config)
        face_crop = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = classifier.process(face_crop)

        total = sum(result.all_emotions.values())
        assert total == pytest.approx(1.0, abs=0.01)

    def test_all_emotion_labels_present(self, config):
        classifier = _create_mock_classifier(config)
        face_crop = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = classifier.process(face_crop)

        for label in config.emotion_labels:
            assert label in result.all_emotions

    def test_blank_image_returns_neutral(self, config):
        """Empty/null face crop should return neutral with zero confidence."""
        classifier = _create_mock_classifier(config)

        # Empty array
        empty_crop = np.array([], dtype=np.uint8)
        result = classifier.process(empty_crop)
        assert result.primary_emotion == "neutral"
        assert result.confidence == 0.0

    def test_none_image_returns_neutral(self, config):
        classifier = _create_mock_classifier(config)
        result = classifier.process(None)
        assert result.primary_emotion == "neutral"
        assert result.confidence == 0.0

    def test_primary_emotion_is_highest(self, config):
        classifier = _create_mock_classifier(config)
        face_crop = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = classifier.process(face_crop)

        max_emotion = max(result.all_emotions, key=result.all_emotions.get)
        assert result.primary_emotion == max_emotion
