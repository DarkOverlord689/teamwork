"""Unit tests for app.core.audio.data_types."""

from __future__ import annotations

import pytest

from app.core.audio.data_types import (
    AudioResult,
    AudioSessionMetrics,
    Interruption,
    InterruptionType,
    SpeakerMetrics,
    SpeakerRole,
    SpeakerSegment,
    SpeakerTurn,
    TranscriptSegment,
    WordTimestamp,
)


# ---------------------------------------------------------------------------
# InterruptionType enum
# ---------------------------------------------------------------------------


class TestInterruptionTypeEnum:
    def test_values(self) -> None:
        assert InterruptionType.DISRUPTIVE == "disruptive"
        assert InterruptionType.BACK_CHANNEL == "back_channel"
        assert InterruptionType.COOPERATIVE == "cooperative"

    def test_is_str_subclass(self) -> None:
        assert isinstance(InterruptionType.DISRUPTIVE, str)


class TestSpeakerRoleEnum:
    def test_values(self) -> None:
        assert SpeakerRole.UNKNOWN == "unknown"
        assert SpeakerRole.LEADER == "leader"
        assert SpeakerRole.CONTRIBUTOR == "contributor"
        assert SpeakerRole.PASSIVE == "passive"

    def test_is_str_subclass(self) -> None:
        assert isinstance(SpeakerRole.LEADER, str)


# ---------------------------------------------------------------------------
# SpeakerSegment
# ---------------------------------------------------------------------------


class TestSpeakerSegment:
    def test_to_dict_without_confidence(self) -> None:
        seg = SpeakerSegment(start=0.0, end=5.0, speaker_id="SPEAKER_00")
        d = seg.to_dict()
        assert d == {"start": 0.0, "end": 5.0, "speaker_id": "SPEAKER_00"}
        assert "confidence" not in d

    def test_to_dict_with_confidence(self) -> None:
        seg = SpeakerSegment(start=1.0, end=3.5, speaker_id="SPEAKER_01", confidence=0.92)
        d = seg.to_dict()
        assert d["confidence"] == pytest.approx(0.92)
        assert d["speaker_id"] == "SPEAKER_01"

    def test_default_confidence_is_none(self) -> None:
        seg = SpeakerSegment(start=0.0, end=1.0, speaker_id="SPEAKER_00")
        assert seg.confidence is None


# ---------------------------------------------------------------------------
# SpeakerMetrics defaults
# ---------------------------------------------------------------------------


class TestSpeakerMetricsDefaults:
    def test_back_channel_count_defaults_to_zero(self) -> None:
        metrics = SpeakerMetrics(
            speaker_id="SPEAKER_00",
            speaking_time_seconds=120.0,
            turn_count=10,
            interruption_count=2,
            interrupted_count=1,
            avg_turn_duration=12.0,
            participation_ratio=0.4,
        )
        assert metrics.back_channel_count == 0

    def test_to_dict_contains_all_fields(self) -> None:
        metrics = SpeakerMetrics(
            speaker_id="SPEAKER_00",
            speaking_time_seconds=120.0,
            turn_count=10,
            interruption_count=2,
            interrupted_count=1,
            avg_turn_duration=12.0,
            participation_ratio=0.4,
        )
        d = metrics.to_dict()
        expected_keys = {
            "speaker_id",
            "speaking_time_seconds",
            "turn_count",
            "interruption_count",
            "interrupted_count",
            "avg_turn_duration",
            "participation_ratio",
            "back_channel_count",
        }
        assert set(d.keys()) == expected_keys


# ---------------------------------------------------------------------------
# AudioSessionMetrics
# ---------------------------------------------------------------------------


class TestAudioSessionMetrics:
    def _make_speaker_metrics(self, speaker_id: str) -> SpeakerMetrics:
        return SpeakerMetrics(
            speaker_id=speaker_id,
            speaking_time_seconds=60.0,
            turn_count=5,
            interruption_count=1,
            interrupted_count=0,
            avg_turn_duration=12.0,
            participation_ratio=0.5,
        )

    def test_to_dict_with_empty_per_speaker(self) -> None:
        metrics = AudioSessionMetrics(
            total_speakers=0,
            duration=300.0,
            total_speaking_time=0.0,
            silence_ratio=1.0,
            participation_cv=0.0,
            turn_alternation_rate=0.0,
        )
        d = metrics.to_dict()
        assert d["per_speaker_metrics"] == []

    def test_to_dict_with_per_speaker_metrics(self) -> None:
        spk0 = self._make_speaker_metrics("SPEAKER_00")
        spk1 = self._make_speaker_metrics("SPEAKER_01")
        session = AudioSessionMetrics(
            total_speakers=2,
            duration=300.0,
            total_speaking_time=240.0,
            silence_ratio=0.2,
            participation_cv=0.05,
            turn_alternation_rate=0.8,
            per_speaker_metrics=[spk0, spk1],
        )
        d = session.to_dict()
        assert len(d["per_speaker_metrics"]) == 2
        assert d["per_speaker_metrics"][0]["speaker_id"] == "SPEAKER_00"
        assert d["total_speakers"] == 2
        assert d["silence_ratio"] == pytest.approx(0.2)

    def test_default_per_speaker_metrics_is_empty_list(self) -> None:
        session = AudioSessionMetrics(
            total_speakers=0,
            duration=0.0,
            total_speaking_time=0.0,
            silence_ratio=0.0,
            participation_cv=0.0,
            turn_alternation_rate=0.0,
        )
        assert session.per_speaker_metrics == []


# ---------------------------------------------------------------------------
# AudioResult
# ---------------------------------------------------------------------------


class TestAudioResult:
    def _make_session_metrics(self) -> AudioSessionMetrics:
        spk = SpeakerMetrics(
            speaker_id="SPEAKER_00",
            speaking_time_seconds=90.0,
            turn_count=6,
            interruption_count=1,
            interrupted_count=2,
            avg_turn_duration=15.0,
            participation_ratio=0.6,
        )
        return AudioSessionMetrics(
            total_speakers=1,
            duration=150.0,
            total_speaking_time=90.0,
            silence_ratio=0.4,
            participation_cv=0.0,
            turn_alternation_rate=0.5,
            per_speaker_metrics=[spk],
        )

    def test_to_dict_empty_result(self) -> None:
        result = AudioResult(
            video_path="/data/session.mp4",
            duration_seconds=300.0,
            sample_rate=16000,
        )
        d = result.to_dict()
        assert d["video_path"] == "/data/session.mp4"
        assert d["duration_seconds"] == pytest.approx(300.0)
        assert d["sample_rate"] == 16000
        assert d["segments"] == []
        assert d["turns"] == []
        assert d["transcripts"] == []
        assert d["interruptions"] == []
        assert "session_metrics" not in d

    def test_to_dict_with_nested_objects(self) -> None:
        seg = SpeakerSegment(start=0.0, end=5.0, speaker_id="SPEAKER_00", confidence=0.9)
        turn = SpeakerTurn(
            start=0.0, end=5.0, speaker_id="SPEAKER_00", duration=5.0, segment_count=1
        )
        word = WordTimestamp(word="hola", start=0.5, end=1.0, confidence=0.95)
        transcript = TranscriptSegment(
            start=0.0,
            end=5.0,
            speaker_id="SPEAKER_00",
            text="hola mundo",
            words=[word],
        )
        interruption = Interruption(
            time=3.0,
            interrupter_id="SPEAKER_01",
            interrupted_id="SPEAKER_00",
            overlap_duration=0.5,
            interruption_type=InterruptionType.COOPERATIVE,
        )
        session_metrics = self._make_session_metrics()

        result = AudioResult(
            video_path="/data/session.mp4",
            duration_seconds=300.0,
            sample_rate=16000,
            segments=[seg],
            turns=[turn],
            transcripts=[transcript],
            interruptions=[interruption],
            processing_time_seconds=12.5,
            session_metrics=session_metrics,
        )
        d = result.to_dict()

        # Top-level fields
        assert d["processing_time_seconds"] == pytest.approx(12.5)

        # Nested segments
        assert len(d["segments"]) == 1
        assert d["segments"][0]["speaker_id"] == "SPEAKER_00"
        assert d["segments"][0]["confidence"] == pytest.approx(0.9)

        # Nested turns
        assert len(d["turns"]) == 1
        assert d["turns"][0]["duration"] == pytest.approx(5.0)

        # Nested transcripts + words
        assert len(d["transcripts"]) == 1
        assert d["transcripts"][0]["text"] == "hola mundo"
        assert len(d["transcripts"][0]["words"]) == 1
        assert d["transcripts"][0]["words"][0]["word"] == "hola"

        # Nested interruptions
        assert len(d["interruptions"]) == 1
        assert d["interruptions"][0]["interruption_type"] == "cooperative"

        # Nested session_metrics
        assert "session_metrics" in d
        assert d["session_metrics"]["total_speakers"] == 1
        assert len(d["session_metrics"]["per_speaker_metrics"]) == 1

    def test_default_lists_are_independent_instances(self) -> None:
        """Mutable default field_factory should create independent instances."""
        r1 = AudioResult(video_path="a.mp4", duration_seconds=10.0, sample_rate=16000)
        r2 = AudioResult(video_path="b.mp4", duration_seconds=20.0, sample_rate=16000)
        r1.segments.append(SpeakerSegment(start=0.0, end=1.0, speaker_id="S0"))
        assert r2.segments == []
