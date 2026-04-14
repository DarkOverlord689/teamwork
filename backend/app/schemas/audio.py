"""Pydantic schemas for audio processing API responses.

Follows the same conventions as ``app.schemas.vision``: flat response models
with ``model_config`` for ORM compatibility.
"""

from __future__ import annotations

from typing import Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Low-level primitives
# ---------------------------------------------------------------------------


class WordTimestampSchema(BaseModel):
    word: str
    start: float
    end: float
    confidence: float


class SpeakerSegmentSchema(BaseModel):
    start: float
    end: float
    speaker_id: str
    confidence: float | None = None


class SpeakerTurnSchema(BaseModel):
    start: float
    end: float
    speaker_id: str
    duration: float
    segment_count: int


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------


class TranscriptSegmentSchema(BaseModel):
    start: float
    end: float
    speaker_id: str
    text: str
    words: List[WordTimestampSchema] = []
    language: str = "es"
    no_speech_prob: float = 0.0


# ---------------------------------------------------------------------------
# Interruptions
# ---------------------------------------------------------------------------


class InterruptionSchema(BaseModel):
    time: float
    interrupter_id: str
    interrupted_id: str
    overlap_duration: float
    interruption_type: str


# ---------------------------------------------------------------------------
# Per-speaker and session metrics
# ---------------------------------------------------------------------------


class SpeakerMetricsSchema(BaseModel):
    speaker_id: str
    speaking_time_seconds: float
    turn_count: int
    interruption_count: int
    interrupted_count: int
    avg_turn_duration: float
    participation_ratio: float
    back_channel_count: int = 0


class AudioSessionMetricsSchema(BaseModel):
    total_speakers: int
    duration: float
    total_speaking_time: float
    silence_ratio: float
    participation_cv: float
    turn_alternation_rate: float
    per_speaker_metrics: List[SpeakerMetricsSchema] = []


# ---------------------------------------------------------------------------
# Full audio result
# ---------------------------------------------------------------------------


class AudioResultSchema(BaseModel):
    video_path: str
    duration_seconds: float
    sample_rate: int
    segments: List[SpeakerSegmentSchema] = []
    turns: List[SpeakerTurnSchema] = []
    transcripts: List[TranscriptSegmentSchema] = []
    interruptions: List[InterruptionSchema] = []
    processing_time_seconds: float = 0.0
    session_metrics: Optional[AudioSessionMetricsSchema] = None


# ---------------------------------------------------------------------------
# API request / response wrappers
# ---------------------------------------------------------------------------


class StartAudioRequest(BaseModel):
    """Request body for starting audio processing."""

    session_id: UUID = Field(..., description="ID of the analysis session to process")
    config_overrides: Optional[dict] = Field(
        None, description="Optional AudioConfig overrides (e.g. whisper_model_size)"
    )


class AudioProcessingResponse(BaseModel):
    """Returned when an audio processing task is enqueued."""

    session_id: UUID
    task_id: str
    status: str = "queued"


class AudioStatusResponse(BaseModel):
    """Returned when checking the status of an audio processing task."""

    session_id: UUID
    task_id: str | None = None
    status: str
    progress: float | None = None  # 0.0 – 1.0
    error: str | None = None
