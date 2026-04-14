"""Tests for Diarizer (T2.4).

Pyannote is mocked entirely to avoid model downloads.  The tests focus on:
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
    """load() should raise DiarizationError when pyannote_auth_token is empty."""
    config = AudioConfig(pyannote_auth_token="")
    diarizer = Diarizer(config)
    with pytest.raises(DiarizationError, match="pyannote_auth_token is required"):
        diarizer.load()


def test_diarizer_load_sets_is_loaded():
    """load() should set is_loaded=True when the pipeline is available."""
    config = AudioConfig(pyannote_auth_token="fake-token")
    diarizer = Diarizer(config)

    mock_pipeline = MagicMock()
    with patch("app.core.audio.diarizer.Pipeline") as MockPipeline:
        MockPipeline.from_pretrained.return_value = mock_pipeline
        diarizer.load()

    assert diarizer.is_loaded


def test_diarizer_unload():
    """unload() should set is_loaded=False and clear the pipeline."""
    config = AudioConfig(pyannote_auth_token="fake-token")
    diarizer = Diarizer(config)
    diarizer._pipeline = MagicMock()
    diarizer._loaded = True

    diarizer.unload()

    assert not diarizer.is_loaded
    assert diarizer._pipeline is None


# ---------------------------------------------------------------------------
# Label normalization
# ---------------------------------------------------------------------------


def test_speaker_label_normalization_basic():
    """Labels should be normalized by order of first appearance."""
    config = AudioConfig(pyannote_auth_token="fake-token")
    diarizer = Diarizer(config)

    raw = ["SPEAKER_02", "SPEAKER_00", "SPEAKER_02", "SPEAKER_01"]
    normalized = diarizer._normalize_labels(raw)

    # First appearance: SPEAKER_02 -> speaker_0, SPEAKER_00 -> speaker_1, SPEAKER_01 -> speaker_2
    assert normalized[0] == "speaker_0"
    assert normalized[1] == "speaker_1"
    assert normalized[2] == "speaker_0"  # same label as first occurrence
    assert normalized[3] == "speaker_2"


def test_speaker_label_normalization_single():
    """Single speaker should always be speaker_0."""
    config = AudioConfig(pyannote_auth_token="fake-token")
    diarizer = Diarizer(config)

    raw = ["SPEAKER_99", "SPEAKER_99", "SPEAKER_99"]
    normalized = diarizer._normalize_labels(raw)

    assert all(n == "speaker_0" for n in normalized)


def test_speaker_label_normalization_empty():
    """Empty list should return empty list."""
    config = AudioConfig(pyannote_auth_token="fake-token")
    diarizer = Diarizer(config)
    assert diarizer._normalize_labels([]) == []


def test_speaker_label_normalization_writes_map():
    """_normalize_labels should populate the label_map argument."""
    config = AudioConfig(pyannote_auth_token="fake-token")
    diarizer = Diarizer(config)

    mapping: dict = {}
    diarizer._normalize_labels(["A", "B", "A"], label_map=mapping)

    assert mapping["A"] == "speaker_0"
    assert mapping["B"] == "speaker_1"


# ---------------------------------------------------------------------------
# process() with mocked pipeline
# ---------------------------------------------------------------------------


def _make_mock_turn(start: float, end: float) -> MagicMock:
    turn = MagicMock()
    turn.start = start
    turn.end = end
    return turn


def test_process_returns_sorted_speaker_segments():
    """process() should return sorted SpeakerSegments with normalized labels."""
    config = AudioConfig(pyannote_auth_token="fake-token")
    diarizer = Diarizer(config)

    turn1 = _make_mock_turn(3.0, 5.0)
    turn2 = _make_mock_turn(0.0, 2.5)

    mock_diarization = MagicMock()
    mock_diarization.itertracks.return_value = [
        (turn1, None, "SPEAKER_01"),
        (turn2, None, "SPEAKER_00"),
    ]

    mock_pipeline = MagicMock()
    mock_pipeline.return_value = mock_diarization

    diarizer._pipeline = mock_pipeline
    diarizer._loaded = True

    segments = diarizer.process("fake_path.wav")

    assert len(segments) == 2
    assert all(isinstance(s, SpeakerSegment) for s in segments)
    # Should be sorted ascending by start
    assert segments[0].start <= segments[1].start
    assert segments[0].start == pytest.approx(0.0)
    assert segments[1].start == pytest.approx(3.0)


def test_process_normalizes_labels():
    """process() should assign speaker IDs by order of first appearance."""
    config = AudioConfig(pyannote_auth_token="fake-token", diarize_min_duration=0.0)
    diarizer = Diarizer(config)

    turn1 = _make_mock_turn(0.0, 1.0)
    turn2 = _make_mock_turn(1.5, 2.5)
    turn3 = _make_mock_turn(3.0, 4.0)

    mock_diarization = MagicMock()
    mock_diarization.itertracks.return_value = [
        (turn1, None, "SPEAKER_02"),
        (turn2, None, "SPEAKER_00"),
        (turn3, None, "SPEAKER_02"),
    ]

    mock_pipeline = MagicMock()
    mock_pipeline.return_value = mock_diarization

    diarizer._pipeline = mock_pipeline
    diarizer._loaded = True

    segments = diarizer.process("fake.wav")
    # Sorted by start: turn1=speaker_0, turn2=speaker_1, turn3=speaker_0
    by_start = sorted(segments, key=lambda s: s.start)
    assert by_start[0].speaker_id == "speaker_0"
    assert by_start[1].speaker_id == "speaker_1"
    assert by_start[2].speaker_id == "speaker_0"


def test_process_filters_short_segments():
    """Segments shorter than diarize_min_duration should be dropped."""
    config = AudioConfig(pyannote_auth_token="fake-token", diarize_min_duration=0.5)
    diarizer = Diarizer(config)

    short_turn = _make_mock_turn(0.0, 0.2)   # 0.2s < 0.5s → filtered
    long_turn = _make_mock_turn(1.0, 2.0)    # 1.0s >= 0.5s → kept

    mock_diarization = MagicMock()
    mock_diarization.itertracks.return_value = [
        (short_turn, None, "SPEAKER_00"),
        (long_turn, None, "SPEAKER_01"),
    ]

    mock_pipeline = MagicMock()
    mock_pipeline.return_value = mock_diarization

    diarizer._pipeline = mock_pipeline
    diarizer._loaded = True

    segments = diarizer.process("fake.wav")
    assert len(segments) == 1
    assert segments[0].start == pytest.approx(1.0)


def test_process_raises_when_not_loaded():
    """process() should raise DiarizationError if the pipeline is not loaded."""
    config = AudioConfig(pyannote_auth_token="fake-token")
    diarizer = Diarizer(config)  # NOT loaded

    with pytest.raises(DiarizationError, match="not loaded"):
        diarizer.process("fake.wav")


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


def test_context_manager():
    """Diarizer should work as a context manager (load on enter, unload on exit)."""
    config = AudioConfig(pyannote_auth_token="fake-token")
    diarizer = Diarizer(config)

    mock_pipeline = MagicMock()
    with patch("app.core.audio.diarizer.Pipeline") as MockPipeline:
        MockPipeline.from_pretrained.return_value = mock_pipeline
        with diarizer as d:
            assert d.is_loaded

    assert not diarizer.is_loaded
