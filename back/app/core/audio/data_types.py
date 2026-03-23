"""Data classes for the audio processing pipeline.

These dataclasses define the structured output at every level of the pipeline:
per-word timestamps (WordTimestamp), speaker diarization segments
(SpeakerSegment / SpeakerTurn), aligned transcripts (TranscriptSegment),
detected interruptions (Interruption), per-speaker metrics (SpeakerMetrics),
session-level aggregates (AudioSessionMetrics), and the full video result
(AudioResult).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SpeakerRole(str, Enum):
    """Inferred participation role for a speaker across the session."""

    UNKNOWN = "unknown"
    LEADER = "leader"
    CONTRIBUTOR = "contributor"
    PASSIVE = "passive"


class InterruptionType(str, Enum):
    """Semantic classification of a detected interruption event."""

    DISRUPTIVE = "disruptive"
    BACK_CHANNEL = "back_channel"
    COOPERATIVE = "cooperative"


# ---------------------------------------------------------------------------
# Low-level primitives
# ---------------------------------------------------------------------------


@dataclass
class WordTimestamp:
    """Timing and confidence for a single recognised word."""

    word: str
    start: float
    end: float
    confidence: float

    def to_dict(self) -> dict:
        return {
            "word": self.word,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
        }


# ---------------------------------------------------------------------------
# Diarisation output
# ---------------------------------------------------------------------------


@dataclass
class SpeakerSegment:
    """A contiguous speech segment assigned to a single speaker by the diariser."""

    start: float
    end: float
    speaker_id: str
    confidence: float | None = None

    def to_dict(self) -> dict:
        result: dict = {
            "start": self.start,
            "end": self.end,
            "speaker_id": self.speaker_id,
        }
        if self.confidence is not None:
            result["confidence"] = self.confidence
        return result


@dataclass
class SpeakerTurn:
    """A speaker turn — one or more consecutive segments by the same speaker."""

    start: float
    end: float
    speaker_id: str
    duration: float
    segment_count: int = 1

    def to_dict(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "speaker_id": self.speaker_id,
            "duration": self.duration,
            "segment_count": self.segment_count,
        }


# ---------------------------------------------------------------------------
# Transcription output
# ---------------------------------------------------------------------------


@dataclass
class TranscriptSegment:
    """Whisper transcript aligned to a diarised speaker segment."""

    start: float
    end: float
    speaker_id: str
    text: str
    words: List[WordTimestamp] = field(default_factory=list)
    language: str = "es"
    no_speech_prob: float = 0.0

    def to_dict(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "speaker_id": self.speaker_id,
            "text": self.text,
            "words": [w.to_dict() for w in self.words],
            "language": self.language,
            "no_speech_prob": self.no_speech_prob,
        }


# ---------------------------------------------------------------------------
# Interruption detection
# ---------------------------------------------------------------------------


@dataclass
class Interruption:
    """A detected interruption event between two speakers."""

    time: float
    interrupter_id: str
    interrupted_id: str
    overlap_duration: float
    interruption_type: str = InterruptionType.DISRUPTIVE

    def to_dict(self) -> dict:
        return {
            "time": self.time,
            "interrupter_id": self.interrupter_id,
            "interrupted_id": self.interrupted_id,
            "overlap_duration": self.overlap_duration,
            "interruption_type": self.interruption_type,
        }


# ---------------------------------------------------------------------------
# Per-speaker metrics
# ---------------------------------------------------------------------------


@dataclass
class SpeakerMetrics:
    """Aggregated participation metrics for one speaker across the session."""

    speaker_id: str
    speaking_time_seconds: float
    turn_count: int
    interruption_count: int
    interrupted_count: int
    avg_turn_duration: float
    participation_ratio: float
    back_channel_count: int = 0

    def to_dict(self) -> dict:
        return {
            "speaker_id": self.speaker_id,
            "speaking_time_seconds": self.speaking_time_seconds,
            "turn_count": self.turn_count,
            "interruption_count": self.interruption_count,
            "interrupted_count": self.interrupted_count,
            "avg_turn_duration": self.avg_turn_duration,
            "participation_ratio": self.participation_ratio,
            "back_channel_count": self.back_channel_count,
        }


# ---------------------------------------------------------------------------
# Session-level aggregate metrics
# ---------------------------------------------------------------------------


@dataclass
class AudioSessionMetrics:
    """Aggregated metrics for the entire audio analysis session."""

    total_speakers: int
    duration: float
    total_speaking_time: float
    silence_ratio: float
    participation_cv: float  # coefficient of variation — equity measure
    turn_alternation_rate: float
    per_speaker_metrics: List[SpeakerMetrics] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_speakers": self.total_speakers,
            "duration": self.duration,
            "total_speaking_time": self.total_speaking_time,
            "silence_ratio": self.silence_ratio,
            "participation_cv": self.participation_cv,
            "turn_alternation_rate": self.turn_alternation_rate,
            "per_speaker_metrics": [m.to_dict() for m in self.per_speaker_metrics],
        }


# ---------------------------------------------------------------------------
# Full-video result
# ---------------------------------------------------------------------------


@dataclass
class AudioResult:
    """Complete output of the audio pipeline for one video file."""

    video_path: str
    duration_seconds: float
    sample_rate: int
    segments: List[SpeakerSegment] = field(default_factory=list)
    turns: List[SpeakerTurn] = field(default_factory=list)
    transcripts: List[TranscriptSegment] = field(default_factory=list)
    interruptions: List[Interruption] = field(default_factory=list)
    processing_time_seconds: float = 0.0
    session_metrics: AudioSessionMetrics | None = None

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict for Celery/API transport."""
        result: dict = {
            "video_path": self.video_path,
            "duration_seconds": self.duration_seconds,
            "sample_rate": self.sample_rate,
            "processing_time_seconds": self.processing_time_seconds,
            "segments": [s.to_dict() for s in self.segments],
            "turns": [t.to_dict() for t in self.turns],
            "transcripts": [t.to_dict() for t in self.transcripts],
            "interruptions": [i.to_dict() for i in self.interruptions],
        }
        if self.session_metrics is not None:
            result["session_metrics"] = self.session_metrics.to_dict()
        return result
