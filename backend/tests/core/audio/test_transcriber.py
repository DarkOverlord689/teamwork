"""Tests for Transcriber (T3.3).

All Whisper model calls are fully mocked — no model download required.
"""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from app.core.audio.config import AudioConfig
from app.core.audio.data_types import SpeakerSegment, TranscriptSegment
from app.core.audio.transcriber import Transcriber, TranscriptionError


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_whisper_result() -> dict:
    return {
        "text": "Hola como están",
        "no_speech_prob": 0.05,
        "segments": [
            {
                "words": [
                    {"word": "Hola", "start": 0.0, "end": 0.3, "probability": 0.99},
                    {"word": "como", "start": 0.4, "end": 0.7, "probability": 0.98},
                    {"word": "están", "start": 0.8, "end": 1.2, "probability": 0.97},
                ]
            }
        ],
        "language": "es",
    }


def _make_transcriber(config: AudioConfig | None = None) -> Transcriber:
    """Return a Transcriber with a mocked (unloaded) model slot."""
    if config is None:
        config = AudioConfig()
    t = Transcriber(config)
    t._loaded = False
    return t


# ---------------------------------------------------------------------------
# T3.3-1: basic single-segment transcription
# ---------------------------------------------------------------------------


def test_transcriber_process_single_segment(mock_whisper_result):
    """process() should return a TranscriptSegment for each input segment."""
    config = AudioConfig()
    transcriber = Transcriber(config)
    mock_model = MagicMock()
    mock_model.transcribe.return_value = mock_whisper_result
    transcriber._model = mock_model
    transcriber._loaded = True

    waveform = np.zeros(32000, dtype=np.float32)  # 2 s at 16 kHz
    segments = [SpeakerSegment(start=0.0, end=2.0, speaker_id="speaker_0")]

    results = transcriber.process(waveform, segments)

    assert len(results) == 1
    assert isinstance(results[0], TranscriptSegment)
    assert results[0].speaker_id == "speaker_0"
    assert results[0].text == "Hola como están"


# ---------------------------------------------------------------------------
# T3.3-2: short segments are skipped
# ---------------------------------------------------------------------------


def test_transcriber_skips_short_segments():
    """Segments shorter than 0.5 seconds should be skipped."""
    config = AudioConfig()
    transcriber = Transcriber(config)
    transcriber._loaded = True
    transcriber._model = MagicMock()

    waveform = np.zeros(32000, dtype=np.float32)
    short_segment = SpeakerSegment(start=0.0, end=0.3, speaker_id="speaker_0")
    results = transcriber.process(waveform, [short_segment])

    assert len(results) == 0
    transcriber._model.transcribe.assert_not_called()


# ---------------------------------------------------------------------------
# T3.3-3: high no_speech_prob segments are excluded
# ---------------------------------------------------------------------------


def test_transcriber_skips_no_speech_segments(mock_whisper_result):
    """Segments with high no_speech_prob should be excluded from results."""
    config = AudioConfig(whisper_no_speech_threshold=0.5)
    transcriber = Transcriber(config)
    transcriber._loaded = True
    mock_model = MagicMock()
    mock_model.transcribe.return_value = {**mock_whisper_result, "no_speech_prob": 0.9}
    transcriber._model = mock_model

    waveform = np.zeros(32000, dtype=np.float32)
    segments = [SpeakerSegment(start=0.0, end=2.0, speaker_id="speaker_0")]
    results = transcriber.process(waveform, segments)

    assert len(results) == 0


# ---------------------------------------------------------------------------
# T3.3-4: word timestamps are offset by segment.start
# ---------------------------------------------------------------------------


def test_word_timestamps_have_offset_applied(mock_whisper_result):
    """Word start/end times should be offset by segment.start."""
    config = AudioConfig()
    transcriber = Transcriber(config)
    transcriber._loaded = True
    mock_model = MagicMock()
    mock_model.transcribe.return_value = mock_whisper_result
    transcriber._model = mock_model

    waveform = np.zeros(80000, dtype=np.float32)  # 5 s at 16 kHz
    # Segment starting at 3.0 s
    segments = [SpeakerSegment(start=3.0, end=5.0, speaker_id="speaker_0")]
    results = transcriber.process(waveform, segments)

    assert len(results) == 1
    # "Hola" originally at 0.0 → offset to 3.0
    assert results[0].words[0].start == pytest.approx(3.0, abs=0.01)
    # "como" originally at 0.4 → offset to 3.4
    assert results[0].words[1].start == pytest.approx(3.4, abs=0.01)
    # "están" originally at 0.8 → offset to 3.8
    assert results[0].words[2].start == pytest.approx(3.8, abs=0.01)


# ---------------------------------------------------------------------------
# T3.3-5: multiple segments, mixed results
# ---------------------------------------------------------------------------


def test_transcriber_multiple_segments(mock_whisper_result):
    """process() should handle multiple segments independently."""
    config = AudioConfig()
    transcriber = Transcriber(config)
    transcriber._loaded = True

    # First call speech, second call no-speech.
    speech_result = mock_whisper_result
    no_speech_result = {**mock_whisper_result, "no_speech_prob": 0.95, "text": ""}
    mock_model = MagicMock()
    mock_model.transcribe.side_effect = [speech_result, no_speech_result]
    transcriber._model = mock_model

    waveform = np.zeros(80000, dtype=np.float32)
    segments = [
        SpeakerSegment(start=0.0, end=2.0, speaker_id="speaker_0"),
        SpeakerSegment(start=2.5, end=4.5, speaker_id="speaker_1"),
    ]
    results = transcriber.process(waveform, segments)

    assert len(results) == 1
    assert results[0].speaker_id == "speaker_0"


# ---------------------------------------------------------------------------
# T3.3-6: waveform is cast to float32
# ---------------------------------------------------------------------------


def test_transcriber_casts_int16_waveform(mock_whisper_result):
    """Non-float32 waveforms should be silently cast before transcription."""
    config = AudioConfig()
    transcriber = Transcriber(config)
    transcriber._loaded = True
    mock_model = MagicMock()
    mock_model.transcribe.return_value = mock_whisper_result
    transcriber._model = mock_model

    # int16 waveform (common from raw PCM files)
    waveform = np.zeros(32000, dtype=np.int16)
    segments = [SpeakerSegment(start=0.0, end=2.0, speaker_id="speaker_0")]
    results = transcriber.process(waveform, segments)

    assert len(results) == 1
    # Verify transcribe was called with a float32 array.
    called_audio = mock_model.transcribe.call_args[0][0]
    assert called_audio.dtype == np.float32


# ---------------------------------------------------------------------------
# T3.3-7: empty segment list
# ---------------------------------------------------------------------------


def test_transcriber_empty_segments():
    """Empty segment list should return an empty list without calling the model."""
    config = AudioConfig()
    transcriber = Transcriber(config)
    transcriber._loaded = True
    mock_model = MagicMock()
    transcriber._model = mock_model

    waveform = np.zeros(32000, dtype=np.float32)
    results = transcriber.process(waveform, [])

    assert results == []
    mock_model.transcribe.assert_not_called()


# ---------------------------------------------------------------------------
# T3.3-8: TranscriptionError is importable
# ---------------------------------------------------------------------------


def test_transcription_error_is_exception():
    """TranscriptionError should be a subclass of Exception."""
    assert issubclass(TranscriptionError, Exception)
