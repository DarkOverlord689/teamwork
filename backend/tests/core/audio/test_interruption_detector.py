"""Tests for InterruptionDetector (T4.1)."""

import pytest

from app.core.audio.config import AudioConfig
from app.core.audio.data_types import (
    Interruption,
    InterruptionType,
    SpeakerTurn,
    TranscriptSegment,
)
from app.core.audio.interruption_detector import InterruptionDetector


@pytest.fixture
def detector() -> InterruptionDetector:
    return InterruptionDetector(
        AudioConfig(
            interruption_overlap_threshold=0.3,
            interruption_gap_threshold=0.2,
        )
    )


def test_overlap_below_threshold_ignored(detector: InterruptionDetector) -> None:
    """Overlaps shorter than threshold should not produce interruptions."""
    turns = [
        SpeakerTurn(0.0, 3.0, "speaker_0", 3.0),
        SpeakerTurn(2.9, 4.0, "speaker_1", 1.1),
    ]
    # Overlap is 0.1 s < 0.3 s threshold
    overlaps = [(2.9, 3.0, "speaker_0", "speaker_1")]
    result = detector.process(turns, [], overlaps)
    assert len(result) == 0


def test_disruptive_interruption_detected(detector: InterruptionDetector) -> None:
    """Clear cut-off interruption should be classified as DISRUPTIVE."""
    turns = [
        SpeakerTurn(0.0, 3.0, "speaker_0", 3.0),  # ends at 3.0
        SpeakerTurn(2.5, 5.0, "speaker_1", 2.5),  # starts at 2.5, overlap 0.5 s
    ]
    # speaker_0 started first; speaker_1 interrupts at 2.5
    overlaps = [(2.5, 3.0, "speaker_0", "speaker_1")]  # 0.5 s overlap
    result = detector.process(turns, [], overlaps)
    assert len(result) == 1
    assert result[0].interruption_type == InterruptionType.DISRUPTIVE
    assert result[0].interrupter_id == "speaker_1"
    assert result[0].interrupted_id == "speaker_0"


def test_back_channel_classified(detector: InterruptionDetector) -> None:
    """Short Spanish back-channel text during overlap should be BACK_CHANNEL."""
    turns = [
        SpeakerTurn(0.0, 4.0, "speaker_0", 4.0),
        SpeakerTurn(2.5, 3.1, "speaker_1", 0.6),
    ]
    overlaps = [(2.5, 3.0, "speaker_0", "speaker_1")]
    transcripts = [
        TranscriptSegment(2.5, 3.1, "speaker_1", "sí"),
    ]
    result = detector.process(turns, transcripts, overlaps)
    assert len(result) == 1
    assert result[0].interruption_type == InterruptionType.BACK_CHANNEL


def test_is_back_channel_detection(detector: InterruptionDetector) -> None:
    """Test back-channel classification for various phrases."""
    assert detector._is_back_channel("sí") is True
    assert detector._is_back_channel("claro") is True
    assert detector._is_back_channel("mm") is True
    assert detector._is_back_channel("pero espera") is False
    assert detector._is_back_channel("no estoy de acuerdo con eso") is False


def test_no_overlaps_no_interruptions(detector: InterruptionDetector) -> None:
    """No overlaps means no interruptions."""
    turns = [SpeakerTurn(0.0, 2.0, "speaker_0", 2.0)]
    result = detector.process(turns, [], [])
    assert result == []


def test_cooperative_overlap(detector: InterruptionDetector) -> None:
    """Overlap where interrupted speaker was almost done → COOPERATIVE.

    For COOPERATIVE we need:
    - overlap duration >= interruption_overlap_threshold (0.3 s)
    - remaining time of interrupted speaker < interruption_gap_threshold (0.2 s)

    speaker_0 ends at 3.1, overlap starts at 2.95:
    - overlap duration = 3.1 - 2.95 = 0.15 s  ← too short, filtered

    Use explicit overlap tuple spanning [2.7, 3.1] (0.4 s >= 0.3 threshold).
    remaining = 3.1 - 2.7 = 0.4 s ≥ 0.2 → still DISRUPTIVE.

    Only way to get COOPERATIVE: remaining < 0.2 AND overlap duration >= 0.3.
    speaker_0 ends at 3.0, speaker_1 starts at 2.95:
    - overlap start forced at 2.95, overlap end 3.5 → duration 0.55 s ≥ 0.3 ✓
    - remaining of speaker_0 = 3.0 - 2.95 = 0.05 < 0.2 ✓ → COOPERATIVE
    """
    turns = [
        SpeakerTurn(0.0, 3.0, "speaker_0", 3.0),   # ends at 3.0
        SpeakerTurn(2.95, 5.0, "speaker_1", 2.05),  # starts at 2.95
    ]
    # overlap start=2.95, end=3.5 → duration 0.55 s >= 0.3 threshold
    # remaining of speaker_0 = 3.0 - 2.95 = 0.05 < 0.2 → COOPERATIVE
    overlaps = [(2.95, 3.5, "speaker_0", "speaker_1")]
    result = detector.process(turns, [], overlaps)
    assert len(result) == 1
    assert result[0].interruption_type == InterruptionType.COOPERATIVE


def test_multi_word_back_channel(detector: InterruptionDetector) -> None:
    """Multi-word back-channel phrase should be detected correctly."""
    assert detector._is_back_channel("claro que sí") is True
    assert detector._is_back_channel("por supuesto") is True
    assert detector._is_back_channel("efectivamente") is True


def test_back_channel_case_insensitive(detector: InterruptionDetector) -> None:
    """Back-channel detection should be case-insensitive."""
    assert detector._is_back_channel("Sí") is True
    assert detector._is_back_channel("CLARO") is True
    assert detector._is_back_channel("  MM  ") is True
