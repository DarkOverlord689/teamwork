"""Tests for Transcriber (T3.3).

All Deepgram API calls are fully mocked — no network request required.
"""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock

from app.core.audio.config import AudioConfig
from app.core.audio.data_types import SpeakerSegment, TranscriptSegment
from app.core.audio.transcriber import Transcriber, TranscriptionError


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_transcriber(config: AudioConfig | None = None) -> Transcriber:
    """Return a Transcriber with a mocked client."""
    if config is None:
        config = AudioConfig(deepgram_api_key="fake")
    t = Transcriber(config)
    t._loaded = True
    return t


def _make_mock_word(word: str, start: float, end: float, speaker: int = 0) -> MagicMock:
    m = MagicMock()
    m.word = word
    m.start = start
    m.end = end
    m.speaker = speaker
    m.confidence = 0.99
    return m


def _make_mock_response(words: list) -> MagicMock:
    mock_alt = MagicMock()
    mock_alt.words = words

    mock_channel = MagicMock()
    mock_channel.alternatives = [mock_alt]

    mock_results = MagicMock()
    mock_results.channels = [mock_channel]

    mock_response = MagicMock()
    mock_response.results = mock_results
    return mock_response


# ---------------------------------------------------------------------------
# T3.3-1: basic single-segment transcription
# ---------------------------------------------------------------------------


def test_transcriber_process_single_segment():
    """process() should return a TranscriptSegment for each speaker block."""
    config = AudioConfig(deepgram_api_key="fake")
    transcriber = _make_transcriber(config)

    words = [
        _make_mock_word("Hola", 0.0, 0.3, 0),
        _make_mock_word("como", 0.4, 0.7, 0),
        _make_mock_word("están", 0.8, 1.2, 0),
    ]
    mock_response = _make_mock_response(words)

    mock_client = MagicMock()
    mock_client.listen.v1.media.transcribe_file.return_value = mock_response
    transcriber._client = mock_client

    waveform = np.zeros(32000, dtype=np.float32)
    segments = [SpeakerSegment(start=0.0, end=2.0, speaker_id="speaker_0")]

    results = transcriber.process(waveform, segments)

    assert len(results) == 1
    assert isinstance(results[0], TranscriptSegment)
    assert results[0].speaker_id == "speaker_0"
    assert results[0].text == "Hola como están"
    assert len(results[0].words) == 3


# ---------------------------------------------------------------------------
# T3.3-2: empty segments are skipped
# ---------------------------------------------------------------------------


def test_transcriber_empty_segments():
    """Empty segment list should return an empty list without calling the API."""
    config = AudioConfig(deepgram_api_key="fake")
    transcriber = _make_transcriber(config)
    mock_client = MagicMock()
    transcriber._client = mock_client

    waveform = np.zeros(32000, dtype=np.float32)
    results = transcriber.process(waveform, [])

    assert results == []
    mock_client.listen.v1.media.transcribe_file.assert_not_called()


# ---------------------------------------------------------------------------
# T3.3-3: multiple segments, mixed results
# ---------------------------------------------------------------------------


def test_transcriber_multiple_segments():
    """process() should handle multiple speakers independently."""
    config = AudioConfig(deepgram_api_key="fake")
    transcriber = _make_transcriber(config)

    words = [
        _make_mock_word("Hola", 0.0, 0.5, 0),
        _make_mock_word("mundo", 0.6, 1.0, 1),
    ]
    mock_response = _make_mock_response(words)

    mock_client = MagicMock()
    mock_client.listen.v1.media.transcribe_file.return_value = mock_response
    transcriber._client = mock_client

    waveform = np.zeros(32000, dtype=np.float32)
    segments = [
        SpeakerSegment(start=0.0, end=2.0, speaker_id="speaker_0"),
        SpeakerSegment(start=0.0, end=2.0, speaker_id="speaker_1"),
    ]
    results = transcriber.process(waveform, segments)

    assert len(results) == 2
    assert results[0].text == "Hola"
    assert results[1].text == "mundo"


# ---------------------------------------------------------------------------
# T3.3-4: word timestamps are preserved
# ---------------------------------------------------------------------------


def test_word_timestamps_preserved():
    """Word start/end times should match Deepgram output."""
    config = AudioConfig(deepgram_api_key="fake")
    transcriber = _make_transcriber(config)

    words = [
        _make_mock_word("Hola", 0.0, 0.3, 0),
        _make_mock_word("como", 0.4, 0.7, 0),
        _make_mock_word("están", 0.8, 1.2, 0),
    ]
    mock_response = _make_mock_response(words)

    mock_client = MagicMock()
    mock_client.listen.v1.media.transcribe_file.return_value = mock_response
    transcriber._client = mock_client

    waveform = np.zeros(32000, dtype=np.float32)
    segments = [SpeakerSegment(start=0.0, end=2.0, speaker_id="speaker_0")]
    results = transcriber.process(waveform, segments)

    assert len(results) == 1
    assert results[0].words[0].start == pytest.approx(0.0, abs=0.01)
    assert results[0].words[1].start == pytest.approx(0.4, abs=0.01)
    assert results[0].words[2].start == pytest.approx(0.8, abs=0.01)


# ---------------------------------------------------------------------------
# T3.3-5: no alternatives returned
# ---------------------------------------------------------------------------


def test_transcriber_no_alternatives():
    """When Deepgram returns no alternatives, return an empty list."""
    config = AudioConfig(deepgram_api_key="fake")
    transcriber = _make_transcriber(config)

    mock_results = MagicMock()
    mock_results.channels = []

    mock_response = MagicMock()
    mock_response.results = mock_results

    mock_client = MagicMock()
    mock_client.listen.v1.media.transcribe_file.return_value = mock_response
    transcriber._client = mock_client

    waveform = np.zeros(32000, dtype=np.float32)
    segments = [SpeakerSegment(start=0.0, end=2.0, speaker_id="speaker_0")]
    results = transcriber.process(waveform, segments)

    assert results == []


# ---------------------------------------------------------------------------
# T3.3-6: waveform is cast to float32
# ---------------------------------------------------------------------------


def test_transcriber_casts_int16_waveform():
    """Non-float32 waveforms should be handled correctly."""
    config = AudioConfig(deepgram_api_key="fake")
    transcriber = _make_transcriber(config)

    words = [_make_mock_word("test", 0.0, 0.5, 0)]
    mock_response = _make_mock_response(words)

    mock_client = MagicMock()
    mock_client.listen.v1.media.transcribe_file.return_value = mock_response
    transcriber._client = mock_client

    waveform = np.zeros(32000, dtype=np.int16)
    segments = [SpeakerSegment(start=0.0, end=2.0, speaker_id="speaker_0")]
    results = transcriber.process(waveform, segments)

    assert len(results) == 1


# ---------------------------------------------------------------------------
# T3.3-7: load without API key raises error
# ---------------------------------------------------------------------------


def test_transcriber_load_without_key():
    """load() should raise TranscriptionError when deepgram_api_key is empty."""
    config = AudioConfig(deepgram_api_key="")
    transcriber = Transcriber(config)

    with pytest.raises(TranscriptionError, match="deepgram_api_key is required"):
        transcriber.load()


# ---------------------------------------------------------------------------
# T3.3-8: TranscriptionError is importable
# ---------------------------------------------------------------------------


def test_transcription_error_is_exception():
    """TranscriptionError should be a subclass of Exception."""
    assert issubclass(TranscriptionError, Exception)
