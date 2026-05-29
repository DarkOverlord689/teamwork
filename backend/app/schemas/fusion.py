"""Pydantic schemas for fusion processing API responses (Module 2.3).

Follows the same conventions as ``app.schemas.audio`` and ``app.schemas.vision``:
flat response models with ``model_config`` for ORM compatibility where needed.
"""

from __future__ import annotations

from typing import Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Temporal alignment
# ---------------------------------------------------------------------------


class TemporalWindowSchema(BaseModel):
    start: float
    end: float
    speaker_id: str | None = None
    person_id: str | None = None
    gaze_at_camera: bool = False
    gaze_confidence: float = 0.0
    body_orientation: float = 0.0
    gesture_type: str = "neutral"
    emotion: str = "neutral"
    overlap_ratio: float = 0.0


class AlignedFeaturesSchema(BaseModel):
    video_path: str
    duration_seconds: float
    windows: List[TemporalWindowSchema] = []
    speaker_to_person: Dict[str, str] = {}
    person_to_speaker: Dict[str, str] = {}


# ---------------------------------------------------------------------------
# Per-student metrics
# ---------------------------------------------------------------------------


class StudentMetricsSchema(BaseModel):
    student_id: str
    speaking_time_seconds: float
    turn_count: int
    interruption_count: int
    interrupted_count: int
    participation_ratio: float
    gaze_contact_percentage: float
    avg_body_orientation: float
    dominant_emotion: str
    initiation_count: int
    back_channel_count: int
    intervention_summary: str | None = None


# ---------------------------------------------------------------------------
# Group metrics
# ---------------------------------------------------------------------------


class GroupMetricsSchema(BaseModel):
    total_students: int
    duration_seconds: float
    total_speaking_time: float
    participation_cv: float
    disruptive_interruption_rate: float
    turn_synchronization_score: float
    avg_gaze_contact_percentage: float
    silence_ratio: float
    per_student_metrics: List[StudentMetricsSchema] = []


# ---------------------------------------------------------------------------
# VALUE rubric scores (AAC&U Teamwork Rubric)
# ---------------------------------------------------------------------------


class RubricScoresSchema(BaseModel):
    student_id: str
    contributes_to_team_meetings: float
    facilitates_contributions: float
    fosters_constructive_climate: float
    responds_to_conflict: float
    individual_contributions_outside: float = 0.0


class GroupRubricScoresSchema(BaseModel):
    contributes_to_team_meetings: float
    facilitates_contributions: float
    fosters_constructive_climate: float
    responds_to_conflict: float
    individual_contributions_outside: float = 0.0
    per_student_scores: List[RubricScoresSchema] = []


# ---------------------------------------------------------------------------
# Explanation
# ---------------------------------------------------------------------------


class ExplanationResultSchema(BaseModel):
    narrative_text: str
    strengths: List[str] = []
    improvements: List[str] = []
    recommendations: List[str] = []
    generated_by: str = "llm"
    model_used: str | None = None
    topic_description: str | None = None
    intervention_summaries: Dict[str, str] = {}


# ---------------------------------------------------------------------------
# Full fusion result
# ---------------------------------------------------------------------------


class FusionResultSchema(BaseModel):
    video_path: str
    aligned_features: AlignedFeaturesSchema
    group_metrics: GroupMetricsSchema
    rubric_scores: GroupRubricScoresSchema
    explanation: ExplanationResultSchema
    processing_time_seconds: float = 0.0


# ---------------------------------------------------------------------------
# API request / response wrappers
# ---------------------------------------------------------------------------


class StartFusionRequest(BaseModel):
    """Request body for starting multimodal fusion analysis."""

    session_id: UUID = Field(..., description="ID of the analysis session to process")
    config_overrides: Optional[dict] = Field(
        None,
        description="Optional FusionConfig overrides (e.g. llm_model, enable_explanation)",
    )


class FusionProcessingResponse(BaseModel):
    """Returned when a fusion processing task is enqueued."""

    session_id: UUID
    task_id: str
    status: str = "queued"


class FusionStatusResponse(BaseModel):
    """Returned when checking the status of a fusion processing task."""

    session_id: UUID
    task_id: str | None = None
    status: str
    progress: float | None = None  # 0.0 – 1.0
    error: str | None = None
