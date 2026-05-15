"""Tests for Diarizer (T2.4).

Deepgram is mocked entirely to avoid API calls.  The tests focus on:
- auth-token validation
- label normalization logic (via the public _normalize_labels helper)
- process() output format and sorting
- context-manager protocol
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.core.audio.config import AudioConfig
from app.core.audio.data_types import SpeakerSegment
from app.core.audio.diarizer import DiarizationError, Diarizer


# ---------------------------------------------------------------------------
# Auth-token validation
# ---------------------------------------------------------------------------


def test_diarizer_requires_auth_token():
    """load() should raise DiarizationError when deepgram_api_key is empty."""
    config = AudioConfig(deepgram_api_key="")
    diarizer = Diarizer(config)
    with pytest.raises(DiarizationError, match="deepgram_api_key is required"):
        diarizer.load()


def test_diarizer_load_sets_is_loaded():
    """load() should set is_loaded=True when the client is available."""
    config = AudioConfig(deepgram_api_key="fake-key")
    diarizer = Diarizer(config)

    with patch("app.core.audio.diarizer.DeepgramClient") as MockClient:
        diarizer.load()

    assert diarizer.is_loaded


def test_diarizer_unload():
    """unload() should set is_loaded=False and clear the client."""
    config = AudioConfig(deepgram_api_key="fake-key")
    diarizer = Diarizer(config)
    diarizer._client = MagicMock()
    diarizer._loaded = True

    diarizer.unload()

    assert not diarizer.is_loaded
    assert diarizer._client is None


# ---------------------------------------------------------------------------
# Label normalization
# ---------------------------------------------------------------------------


def test_speaker_label_normalization_basic():
    """Labels should be normalized by order of first appearance."""
    config = AudioConfig(deepgram_api_key="fake-key")
    diarizer = Diarizer(config)

    raw = ["2", "0", "2", "1"]
    normalized = diarizer._normalize_labels(raw)

    assert normalized[0] == "speaker_0"
    assert normalized[1] == "speaker_1"
    assert normalized[2] == "speaker_0"
    assert normalized[3] == "speaker_2"


def test_speaker_label_normalization_single():
    """Single speaker should always be speaker_0."""
    config = AudioConfig(deepgram_api_key="fake-key")
    diarizer = Diarizer(config)

    raw = ["99", "99", "99"]
    normalized = diarizer._normalize_labels(raw)

    assert all(n == "speaker_0" for n in normalized)


def test_speaker_label_normalization_empty():
    """Empty list should return empty list."""
    config = AudioConfig(deepgram_api_key="fake-key")
    diarizer = Diarizer(config)
    assert diarizer._normalize_labels([]) == []


def test_speaker_label_normalization_writes_map():
    """_normalize_labels should populate the label_map argument."""
    config = AudioConfig(deepgram_api_key="fake-key")
    diarizer = Diarizer(config)

    mapping: dict = {}
    diarizer._normalize_labels(["A", "B", "A"], label_map=mapping)

    assert mapping["A"] == "speaker_0"
    assert mapping["B"] == "speaker_1"


# ---------------------------------------------------------------------------
# process() with mocked Deepgram client
# ---------------------------------------------------------------------------


def _make_mock_utterance(start: float, end: float, speaker: int) -> MagicMock:
    u = MagicMock()
    u.start = start
    u.end = end
    u.speaker = speaker
    return u


def test_process_returns_sorted_speaker_segments(tmp_path):
    """process() should return sorted SpeakerSegments with normalized labels."""
    config = AudioConfig(deepgram_api_key="fake-key")
    diarizer = Diarizer(config)

    u1 = _make_mock_utterance(3.0, 5.0, 1)
    u2 = _make_mock_utterance(0.0, 2.5, 0)

    mock_results = MagicMock()
    mock_results.utterances = [u1, u2]

    mock_response = MagicMock()
    mock_response.results = mock_results

    mock_client = MagicMock()
    mock_client.listen.v1.media.transcribe_file.return_value = mock_response

    diarizer._client = mock_client
    diarizer._loaded = True

    wav = tmp_path / "fake.wav"
    wav.write_bytes(b"RIFF" + b"\x00" * 100)

    segments = diarizer.process(str(wav))

    assert len(segments) == 2
    assert all(isinstance(s, SpeakerSegment) for s in segments)
    assert segments[0].start <= segments[1].start
    assert segments[0].start == pytest.approx(0.0)
    assert segments[1].start == pytest.approx(3.0)


def test_process_normalizes_labels(tmp_path):
    """process() should assign speaker IDs by order of first appearance."""
    config = AudioConfig(deepgram_api_key="fake-key", diarize_min_duration=0.0)
    diarizer = Diarizer(config)

    u1 = _make_mock_utterance(0.0, 1.0, 2)
    u2 = _make_mock_utterance(1.5, 2.5, 0)
    u3 = _make_mock_utterance(3.0, 4.0, 2)

    mock_results = MagicMock()
    mock_results.utterances = [u1, u2, u3]

    mock_response = MagicMock()
    mock_response.results = mock_results

    mock_client = MagicMock()
    mock_client.listen.v1.media.transcribe_file.return_value = mock_response

    diarizer._client = mock_client
    diarizer._loaded = True

    wav = tmp_path / "fake.wav"
    wav.write_bytes(b"RIFF" + b"\x00" * 100)

    segments = diarizer.process(str(wav))
    by_start = sorted(segments, key=lambda s: s.start)
    assert by_start[0].speaker_id == "speaker_0"
    assert by_start[1].speaker_id == "speaker_1"
    assert by_start[2].speaker_id == "speaker_0"


def test_process_filters_short_segments(tmp_path):
    """Segments shorter than diarize_min_duration should be dropped."""
    config = AudioConfig(deepgram_api_key="fake-key", diarize_min_duration=0.5)
    diarizer = Diarizer(config)

    short_u = _make_mock_utterance(0.0, 0.2, 0)
    long_u = _make_mock_utterance(1.0, 2.0, 1)

    mock_results = MagicMock()
    mock_results.utterances = [short_u, long_u]

    mock_response = MagicMock()
    mock_response.results = mock_results

    mock_client = MagicMock()
    mock_client.listen.v1.media.transcribe_file.return_value = mock_response

    diarizer._client = mock_client
    diarizer._loaded = True

    wav = tmp_path / "fake.wav"
    wav.write_bytes(b"RIFF" + b"\x00" * 100)

    segments = diarizer.process(str(wav))
    assert len(segments) == 1
    assert segments[0].start == pytest.approx(1.0)


def test_process_raises_when_not_loaded():
    """process() should raise DiarizationError if the client is not loaded."""
    config = AudioConfig(deepgram_api_key="fake-key")
    diarizer = Diarizer(config)

    with pytest.raises(DiarizationError, match="not loaded"):
        diarizer.process("fake.wav")


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


def test_context_manager():
    """Diarizer should work as a context manager (load on enter, unload on exit)."""
    config = AudioConfig(deepgram_api_key="fake-key")
    diarizer = Diarizer(config)

    with patch("app.core.audio.diarizer.DeepgramClient") as MockClient:
        with diarizer as d:
            assert d.is_loaded

    assert not diarizer.is_loaded
