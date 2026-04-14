"""Tests for TurnAnalyzer (T3.4).

Pure Python — no mocking required.
"""

from __future__ import annotations

import pytest

from app.core.audio.config import AudioConfig
from app.core.audio.data_types import SpeakerSegment, SpeakerTurn
from app.core.audio.turn_analyzer import TurnAnalyzer, TurnAnalysisResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config() -> AudioConfig:
    return AudioConfig(turn_min_gap=0.5)


@pytest.fixture
def analyzer(config: AudioConfig) -> TurnAnalyzer:
    return TurnAnalyzer(config)


# ---------------------------------------------------------------------------
# T3.4-1: single speaker
# ---------------------------------------------------------------------------


def test_single_speaker(analyzer: TurnAnalyzer):
    """Single speaker, no overlaps, CV should be 0."""
    segments = [
        SpeakerSegment(0.0, 2.0, "speaker_0"),
        SpeakerSegment(3.0, 5.0, "speaker_0"),
    ]
    result = analyzer.process(segments, total_duration=10.0)

    assert len(result.overlaps) == 0
    assert result.participation_cv == 0.0
    assert result.speaking_time["speaker_0"] == pytest.approx(4.0, abs=0.01)


# ---------------------------------------------------------------------------
# T3.4-2: alternating speakers
# ---------------------------------------------------------------------------


def test_alternating_speakers(analyzer: TurnAnalyzer):
    """Two speakers alternating — no overlap."""
    segments = [
        SpeakerSegment(0.0, 2.0, "speaker_0"),
        SpeakerSegment(3.0, 5.0, "speaker_1"),
        SpeakerSegment(6.0, 8.0, "speaker_0"),
    ]
    result = analyzer.process(segments, total_duration=10.0)

    assert len(result.overlaps) == 0
    assert result.turn_count["speaker_0"] == 2
    assert result.turn_count["speaker_1"] == 1


# ---------------------------------------------------------------------------
# T3.4-3: overlapping speakers detected
# ---------------------------------------------------------------------------


def test_overlapping_speakers_detected(analyzer: TurnAnalyzer):
    """Overlapping segments from different speakers should be detected."""
    segments = [
        SpeakerSegment(0.0, 3.0, "speaker_0"),
        SpeakerSegment(2.0, 4.0, "speaker_1"),  # overlaps speaker_0 by 1 s
    ]
    result = analyzer.process(segments, total_duration=10.0)

    assert len(result.overlaps) == 1
    overlap_start, overlap_end, *speakers = result.overlaps[0]
    assert overlap_start == pytest.approx(2.0)
    assert overlap_end == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# T3.4-4: merge segments with small gap
# ---------------------------------------------------------------------------


def test_merge_segments_with_small_gap(analyzer: TurnAnalyzer):
    """Segments from same speaker with gap < turn_min_gap should be merged."""
    segments = [
        SpeakerSegment(0.0, 2.0, "speaker_0"),
        SpeakerSegment(2.3, 4.0, "speaker_0"),  # gap = 0.3 s < 0.5 s threshold
    ]
    result = analyzer.process(segments, total_duration=10.0)

    speaker_0_turns = [t for t in result.turns if t.speaker_id == "speaker_0"]
    assert len(speaker_0_turns) == 1
    assert speaker_0_turns[0].segment_count == 2


# ---------------------------------------------------------------------------
# T3.4-5: do not merge segments with large gap
# ---------------------------------------------------------------------------


def test_merge_segments_with_large_gap(analyzer: TurnAnalyzer):
    """Segments from same speaker with gap > turn_min_gap should NOT be merged."""
    segments = [
        SpeakerSegment(0.0, 2.0, "speaker_0"),
        SpeakerSegment(3.0, 5.0, "speaker_0"),  # gap = 1.0 s > 0.5 s threshold
    ]
    result = analyzer.process(segments, total_duration=10.0)

    speaker_0_turns = [t for t in result.turns if t.speaker_id == "speaker_0"]
    assert len(speaker_0_turns) == 2


# ---------------------------------------------------------------------------
# T3.4-6: participation CV for equal speaking times
# ---------------------------------------------------------------------------


def test_participation_cv_equal_speakers(analyzer: TurnAnalyzer):
    """Equal speaking time → CV should be 0."""
    segments = [
        SpeakerSegment(0.0, 5.0, "speaker_0"),
        SpeakerSegment(5.0, 10.0, "speaker_1"),
    ]
    result = analyzer.process(segments, total_duration=10.0)

    assert result.participation_cv == pytest.approx(0.0, abs=0.01)


# ---------------------------------------------------------------------------
# T3.4-7: empty segments
# ---------------------------------------------------------------------------


def test_empty_segments(analyzer: TurnAnalyzer):
    """Empty segment list should return empty TurnAnalysisResult."""
    result = analyzer.process([], total_duration=10.0)

    assert result.turns == []
    assert result.overlaps == []
    assert result.speaking_time == {}
    assert result.participation_cv == 0.0


# ---------------------------------------------------------------------------
# T3.4-8: turn alternation rate
# ---------------------------------------------------------------------------


def test_turn_alternation_rate(analyzer: TurnAnalyzer):
    """turn_alternation_rate should be total_turns / (duration_minutes)."""
    segments = [
        SpeakerSegment(0.0, 10.0, "speaker_0"),
        SpeakerSegment(15.0, 25.0, "speaker_1"),
        SpeakerSegment(30.0, 40.0, "speaker_0"),
    ]
    # 3 turns, 60 s duration → 3 turns/min
    result = analyzer.process(segments, total_duration=60.0)

    assert result.turn_alternation_rate == pytest.approx(3.0, abs=0.01)


# ---------------------------------------------------------------------------
# T3.4-9: avg turn duration
# ---------------------------------------------------------------------------


def test_avg_turn_duration(analyzer: TurnAnalyzer):
    """avg_turn_duration should reflect average over all turns per speaker."""
    segments = [
        SpeakerSegment(0.0, 4.0, "speaker_0"),   # 4 s
        SpeakerSegment(5.0, 7.0, "speaker_0"),   # 2 s
        SpeakerSegment(8.0, 9.0, "speaker_1"),   # 1 s
    ]
    result = analyzer.process(segments, total_duration=10.0)

    assert result.avg_turn_duration["speaker_0"] == pytest.approx(3.0, abs=0.01)
    assert result.avg_turn_duration["speaker_1"] == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# T3.4-10: unequal participation has non-zero CV
# ---------------------------------------------------------------------------


def test_participation_cv_unequal_speakers():
    """Unequal speaking times should produce non-zero CV."""
    config = AudioConfig(turn_min_gap=0.5)
    analyzer = TurnAnalyzer(config)
    segments = [
        SpeakerSegment(0.0, 9.0, "speaker_0"),   # 9 s
        SpeakerSegment(9.5, 10.0, "speaker_1"),  # 0.5 s
    ]
    result = analyzer.process(segments, total_duration=10.0)

    assert result.participation_cv > 0.0


# ---------------------------------------------------------------------------
# T3.4-11: TurnAnalysisResult is a dataclass (no ML inheritance)
# ---------------------------------------------------------------------------


def test_turn_analysis_result_is_dataclass():
    """TurnAnalysisResult should be a plain dataclass, not an AudioBaseProcessor."""
    from app.core.audio.base_processor import AudioBaseProcessor

    assert not issubclass(TurnAnalyzer, AudioBaseProcessor)
    result = TurnAnalysisResult()
    assert hasattr(result, "turns")
    assert hasattr(result, "overlaps")
    assert hasattr(result, "speaking_time")
    assert hasattr(result, "participation_cv")
    assert hasattr(result, "turn_alternation_rate")
