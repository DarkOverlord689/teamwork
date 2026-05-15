"""Integration tests for AudioPipeline (T4.3)."""

import numpy as np
import pytest
from unittest.mock import MagicMock

from app.core.audio.config import AudioConfig
from app.core.audio.data_types import AudioResult, SpeakerSegment
from app.core.audio.pipeline import AudioPipeline


def test_pipeline_context_manager() -> None:
    """Pipeline should load and unload processors on enter/exit."""
    config = AudioConfig(deepgram_api_key="fake")
    pipeline = AudioPipeline(config)

    pipeline._extractor = MagicMock()
    pipeline._diarizer = MagicMock()
    pipeline._transcriber = MagicMock()

    with pipeline:
        pipeline._extractor.load.assert_called_once()
        pipeline._diarizer.load.assert_called_once()

    pipeline._extractor.unload.assert_called_once()
    pipeline._diarizer.unload.assert_called_once()


def test_pipeline_process_audio_returns_audio_result(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """process_audio() should return an AudioResult."""
    import soundfile as sf

    wav = tmp_path / "test.wav"
    sf.write(str(wav), np.zeros(32000, dtype=np.float32), 16000)

    config = AudioConfig(
        enable_diarization=False,
        enable_transcription=False,
        enable_interruption=False,
        enable_participation=False,
    )

    pipeline = AudioPipeline(config)
    pipeline._extractor = MagicMock()
    pipeline._extractor.process.return_value = (
        np.zeros(32000, dtype=np.float32),
        16000,
        2.0,
    )
    pipeline._extractor.write_temp_wav.return_value = str(wav)
    pipeline._extractor.is_loaded = True
    pipeline.load_all()  # won't fail since _diarizer/_transcriber are None

    result = pipeline.process_audio(str(wav))
    assert isinstance(result, AudioResult)
    assert result.duration_seconds == pytest.approx(2.0)
    assert result.segments == []


def test_pipeline_progress_callback_called(tmp_path: pytest.TempPathFactory) -> None:
    """progress_callback should be called with increasing values."""
    import soundfile as sf

    wav = tmp_path / "test.wav"
    sf.write(str(wav), np.zeros(32000, dtype=np.float32), 16000)

    config = AudioConfig(
        enable_diarization=False,
        enable_transcription=False,
        enable_interruption=False,
        enable_participation=False,
    )
    pipeline = AudioPipeline(config)
    pipeline._extractor = MagicMock()
    pipeline._extractor.process.return_value = (
        np.zeros(32000, dtype=np.float32),
        16000,
        2.0,
    )
    pipeline._extractor.write_temp_wav.return_value = str(wav)

    progress_values: list[float] = []

    def callback(p: float, msg: str) -> None:
        progress_values.append(p)

    pipeline.process_audio(str(wav), progress_callback=callback)
    assert len(progress_values) > 0
    assert progress_values[0] == 0.0
    assert progress_values[-1] == 1.0


def test_pipeline_disabled_components_are_none() -> None:
    """Feature flags should disable optional components."""
    config = AudioConfig(
        enable_diarization=False,
        enable_transcription=False,
        enable_interruption=False,
        enable_participation=False,
    )
    pipeline = AudioPipeline(config)
    assert pipeline._diarizer is None
    assert pipeline._transcriber is None
    assert pipeline._interruption_detector is None
    assert pipeline._aggregator is None


def test_pipeline_enabled_components_are_created() -> None:
    """When feature flags are enabled the components should be instantiated."""
    config = AudioConfig(
        enable_diarization=True,
        enable_transcription=True,
        enable_interruption=True,
        enable_participation=True,
        deepgram_api_key="dummy",
    )
    pipeline = AudioPipeline(config)
    assert pipeline._diarizer is not None
    assert pipeline._transcriber is not None
    assert pipeline._interruption_detector is not None
    assert pipeline._aggregator is not None
