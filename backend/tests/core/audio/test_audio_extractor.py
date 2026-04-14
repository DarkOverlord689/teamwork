"""Tests for AudioExtractor (T2.3).

Uses synthetic audio files written with soundfile — no real video files
or ML models are required.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import soundfile as sf

from app.core.audio.audio_extractor import AudioExtractionError, AudioExtractor
from app.core.audio.config import AudioConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_wav(path: str, duration_s: float = 1.0, sr: int = 16000) -> None:
    """Write a silent WAV file at *path*."""
    samples = np.zeros(int(duration_s * sr), dtype=np.float32)
    sf.write(path, samples, sr)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_audio_extractor_load():
    """load() should set is_loaded to True."""
    config = AudioConfig()
    extractor = AudioExtractor(config)
    assert not extractor.is_loaded
    extractor.load()
    assert extractor.is_loaded


def test_audio_extractor_unload():
    """unload() should set is_loaded to False."""
    config = AudioConfig()
    extractor = AudioExtractor(config)
    extractor.load()
    extractor.unload()
    assert not extractor.is_loaded


def test_process_returns_waveform_tuple(tmp_path):
    """process() should return (waveform, sr, duration) tuple."""
    wav_path = str(tmp_path / "test.wav")
    _make_wav(wav_path, duration_s=1.0, sr=16000)

    config = AudioConfig()
    extractor = AudioExtractor(config)
    extractor.load()

    waveform, sr, duration = extractor.process(wav_path)

    assert sr == 16000
    assert duration == pytest.approx(1.0, abs=0.1)
    assert isinstance(waveform, np.ndarray)
    assert waveform.dtype == np.float32


def test_process_resamples_to_config_rate(tmp_path):
    """process() should resample to the configured sample rate."""
    # Write a 44100 Hz file; config requests 16000 Hz
    wav_path = str(tmp_path / "hifi.wav")
    samples = np.zeros(44100, dtype=np.float32)
    sf.write(wav_path, samples, 44100)

    config = AudioConfig(audio_sample_rate=16000)
    extractor = AudioExtractor(config)
    extractor.load()

    waveform, sr, duration = extractor.process(wav_path)

    assert sr == 16000
    assert duration == pytest.approx(1.0, abs=0.2)


def test_process_missing_file_raises_error():
    """process() on a non-existent file should raise AudioExtractionError."""
    config = AudioConfig()
    extractor = AudioExtractor(config)
    extractor.load()

    with pytest.raises(AudioExtractionError, match="File not found"):
        extractor.process("/nonexistent/path/to/audio.wav")


def test_write_temp_wav(tmp_path):
    """write_temp_wav() should create a valid WAV file readable by soundfile."""
    config = AudioConfig()
    extractor = AudioExtractor(config)
    extractor.load()

    waveform = np.zeros(16000, dtype=np.float32)
    temp_path = extractor.write_temp_wav(waveform, 16000)

    try:
        assert temp_path.endswith(".wav")
        assert os.path.exists(temp_path)
        data, sr = sf.read(temp_path)
        assert sr == 16000
        assert len(data) == 16000
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_get_audio_properties(tmp_path):
    """get_audio_properties() should return correct metadata for a WAV file."""
    wav_path = str(tmp_path / "props.wav")
    _make_wav(wav_path, duration_s=2.0, sr=16000)

    config = AudioConfig()
    extractor = AudioExtractor(config)
    props = extractor.get_audio_properties(wav_path)

    assert props["has_audio"] is True
    assert props["sample_rate"] == 16000
    assert props["channels"] == 1
    assert props["duration"] == pytest.approx(2.0, abs=0.1)


def test_get_audio_properties_missing_file():
    """get_audio_properties() on a missing file should return has_audio=False."""
    config = AudioConfig()
    extractor = AudioExtractor(config)
    props = extractor.get_audio_properties("/nonexistent/file.wav")
    assert props["has_audio"] is False


def test_context_manager():
    """AudioExtractor should work as a context manager."""
    config = AudioConfig()
    with AudioExtractor(config) as extractor:
        assert extractor.is_loaded
    assert not extractor.is_loaded
