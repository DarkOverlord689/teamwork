"""Data classes for the multimodal fusion pipeline (Module 2.3).

These dataclasses define the structured output at every level of the fusion
pipeline: temporal alignment windows (TemporalWindow), aligned feature sets
(AlignedFeatures), per-student fused metrics (StudentMetrics), group-level
aggregates (GroupMetrics), rubric scoring (RubricScores, GroupRubricScores),
LLM explanation results (ExplanationResult), and the full fusion output
(FusionResult).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


# ---------------------------------------------------------------------------
# Temporal alignment primitives
# ---------------------------------------------------------------------------


@dataclass
class TemporalWindow:
    """A 500 ms time slice carrying both visual and audio signals.

    The window is the unit of alignment: for each audio turn segment we find
    overlapping visual windows and assign the speaker identity to the tracked
    person whose face embedding matches.
    """

    start: float  # seconds from video start
    end: float
    speaker_id: str | None  # assigned by diarization (may be None = silence)
    person_id: str | None  # assigned by face tracking (may be None = not visible)
    gaze_at_camera: bool = False
    gaze_confidence: float = 0.0
    body_orientation: float = 0.0  # degrees; 0 = facing camera
    gesture_type: str = "neutral"
    emotion: str = "neutral"
    overlap_ratio: float = 0.0  # fraction of window overlapping with speech turn

    def to_dict(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "speaker_id": self.speaker_id,
            "person_id": self.person_id,
            "gaze_at_camera": self.gaze_at_camera,
            "gaze_confidence": self.gaze_confidence,
            "body_orientation": self.body_orientation,
            "gesture_type": self.gesture_type,
            "emotion": self.emotion,
            "overlap_ratio": self.overlap_ratio,
        }


@dataclass
class AlignedFeatures:
    """Full result of temporal alignment for a single video.

    Contains a per-student timeline built by mapping diarizer speaker IDs
    to tracked face person IDs via face-embedding similarity.
    """

    video_path: str
    duration_seconds: float
    windows: List[TemporalWindow] = field(default_factory=list)
    # speaker_to_person: maps diarizer speaker_id -> vision person_id
    speaker_to_person: Dict[str, str] = field(default_factory=dict)
    # person_to_speaker: reverse mapping
    person_to_speaker: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "video_path": self.video_path,
            "duration_seconds": self.duration_seconds,
            "windows": [w.to_dict() for w in self.windows],
            "speaker_to_person": dict(self.speaker_to_person),
            "person_to_speaker": dict(self.person_to_speaker),
        }


# ---------------------------------------------------------------------------
# Per-student fused metrics
# ---------------------------------------------------------------------------


@dataclass
class StudentMetrics:
    """Fused audio + visual metrics for one identified student."""

    student_id: str  # may be speaker_id or person_id depending on alignment quality
    speaking_time_seconds: float
    turn_count: int
    interruption_count: int
    interrupted_count: int
    participation_ratio: float  # fraction of total speaking time
    gaze_contact_percentage: float  # % of frames where looking at camera / peers
    avg_body_orientation: float  # degrees; near 0 = engaged
    dominant_emotion: str
    initiation_count: int  # turns that start a new topic or follow silence
    back_channel_count: int  # short acknowledgement turns

    def to_dict(self) -> dict:
        return {
            "student_id": self.student_id,
            "speaking_time_seconds": self.speaking_time_seconds,
            "turn_count": self.turn_count,
            "interruption_count": self.interruption_count,
            "interrupted_count": self.interrupted_count,
            "participation_ratio": self.participation_ratio,
            "gaze_contact_percentage": self.gaze_contact_percentage,
            "avg_body_orientation": self.avg_body_orientation,
            "dominant_emotion": self.dominant_emotion,
            "initiation_count": self.initiation_count,
            "back_channel_count": self.back_channel_count,
        }


# ---------------------------------------------------------------------------
# Group-level metrics
# ---------------------------------------------------------------------------


@dataclass
class GroupMetrics:
    """Aggregated group-level collaboration metrics."""

    total_students: int
    duration_seconds: float
    participation_cv: float  # coefficient of variation of speaking times; target < 0.3
    disruptive_interruption_rate: float  # disruptive interruptions / total turns
    turn_synchronization_score: float  # 0.0–1.0; ratio of smooth transitions
    avg_gaze_contact_percentage: float  # mean across students
    silence_ratio: float  # fraction of session with no speech
    per_student_metrics: List[StudentMetrics] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_students": self.total_students,
            "duration_seconds": self.duration_seconds,
            "participation_cv": self.participation_cv,
            "disruptive_interruption_rate": self.disruptive_interruption_rate,
            "turn_synchronization_score": self.turn_synchronization_score,
            "avg_gaze_contact_percentage": self.avg_gaze_contact_percentage,
            "silence_ratio": self.silence_ratio,
            "per_student_metrics": [m.to_dict() for m in self.per_student_metrics],
        }


# ---------------------------------------------------------------------------
# UPAO rubric scores
# ---------------------------------------------------------------------------


@dataclass
class RubricScores:
    """UPAO rubric dimension scores (0–20 scale) for one student."""

    student_id: str
    collaboration: float  # turn distribution + visual contact
    communication: float  # interruption rate + turn synchronization
    responsibility: float  # participation consistency + CV contribution
    leadership: float  # speaking share + topic initiation
    technical_contribution: float  # turn count + transcript keyword density

    def to_dict(self) -> dict:
        return {
            "student_id": self.student_id,
            "collaboration": self.collaboration,
            "communication": self.communication,
            "responsibility": self.responsibility,
            "leadership": self.leadership,
            "technical_contribution": self.technical_contribution,
        }


@dataclass
class GroupRubricScores:
    """UPAO rubric scores averaged at group level (0–20 scale)."""

    collaboration: float
    communication: float
    responsibility: float
    leadership: float
    technical_contribution: float
    per_student_scores: List[RubricScores] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "collaboration": self.collaboration,
            "communication": self.communication,
            "responsibility": self.responsibility,
            "leadership": self.leadership,
            "technical_contribution": self.technical_contribution,
            "per_student_scores": [s.to_dict() for s in self.per_student_scores],
        }


# ---------------------------------------------------------------------------
# LLM explanation result
# ---------------------------------------------------------------------------


@dataclass
class ExplanationResult:
    """Narrative explanation produced by the LLM (or rule-based fallback)."""

    narrative_text: str  # full paragraph explanation
    strengths: List[str] = field(default_factory=list)  # bullet-point list
    improvements: List[str] = field(default_factory=list)  # areas to improve
    recommendations: List[str] = field(default_factory=list)  # specific actions
    generated_by: str = "llm"  # "llm" | "rule_based"
    model_used: str | None = None
    # LLM-generated session-level and per-student fields
    topic_description: str | None = None  # 2-3 sentence description of the session topic
    intervention_summaries: Dict[str, str] = field(default_factory=dict)  # student_id -> summary

    def to_dict(self) -> dict:
        return {
            "narrative_text": self.narrative_text,
            "strengths": list(self.strengths),
            "improvements": list(self.improvements),
            "recommendations": list(self.recommendations),
            "generated_by": self.generated_by,
            "model_used": self.model_used,
            "topic_description": self.topic_description,
            "intervention_summaries": dict(self.intervention_summaries),
        }


# ---------------------------------------------------------------------------
# Full fusion result
# ---------------------------------------------------------------------------


@dataclass
class FusionResult:
    """Complete output of the multimodal fusion pipeline for one video."""

    video_path: str
    aligned_features: AlignedFeatures
    group_metrics: GroupMetrics
    rubric_scores: GroupRubricScores
    explanation: ExplanationResult
    processing_time_seconds: float = 0.0

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict for Celery/API transport."""
        return {
            "video_path": self.video_path,
            "aligned_features": self.aligned_features.to_dict(),
            "group_metrics": self.group_metrics.to_dict(),
            "rubric_scores": self.rubric_scores.to_dict(),
            "explanation": self.explanation.to_dict(),
            "processing_time_seconds": self.processing_time_seconds,
        }
