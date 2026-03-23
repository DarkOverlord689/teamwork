"""Pydantic schemas for vision processing API responses.

Follows the same conventions as ``app.schemas.analysis``: flat response models
with ``Config.from_attributes = True`` for ORM compatibility.
"""

from __future__ import annotations

from typing import Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Nested detail schemas
# ---------------------------------------------------------------------------

class GazeDetail(BaseModel):
    direction: List[float] = Field(..., description="[yaw, pitch, roll] in degrees")
    is_looking_at_camera: bool
    confidence: float
    category: str


class GestureDetail(BaseModel):
    gesture_type: str
    confidence: float
    intensity: float = Field(..., ge=0.0, le=1.0)


class PoseDetail(BaseModel):
    body_orientation: float = Field(..., description="Degrees, 0 = facing camera")
    shoulder_angle: float
    confidence: float


class EmotionDetail(BaseModel):
    primary_emotion: str
    confidence: float
    all_emotions: Dict[str, float]


# ---------------------------------------------------------------------------
# Per-person, per-frame
# ---------------------------------------------------------------------------

class PersonFrameSchema(BaseModel):
    person_id: str
    bbox: List[int] = Field(..., min_length=4, max_length=4)
    gaze: Optional[GazeDetail] = None
    gesture: Optional[GestureDetail] = None
    pose: Optional[PoseDetail] = None
    emotion: Optional[EmotionDetail] = None


class FrameResultSchema(BaseModel):
    frame_number: int
    timestamp: float
    persons: List[PersonFrameSchema] = []


# ---------------------------------------------------------------------------
# Aggregation metrics
# ---------------------------------------------------------------------------

class PersonMetricsSchema(BaseModel):
    person_id: str
    total_frames_seen: int
    gaze_contact_percentage: float
    dominant_emotion: str
    emotion_distribution: Dict[str, float]
    average_body_orientation: float
    gesture_counts: Dict[str, int]
    attention_score: float


class SessionMetricsSchema(BaseModel):
    total_persons: int
    total_frames: int
    duration: float
    per_person_metrics: List[PersonMetricsSchema] = []


# ---------------------------------------------------------------------------
# Full video result
# ---------------------------------------------------------------------------

class VisionResultSchema(BaseModel):
    video_path: str
    total_frames: int
    fps_processed: float
    duration_seconds: float
    processing_time_seconds: float = 0.0
    frames: List[FrameResultSchema] = []
    person_embeddings: Optional[Dict[str, List[float]]] = None
    session_metrics: Optional[SessionMetricsSchema] = None


# ---------------------------------------------------------------------------
# API response wrappers
# ---------------------------------------------------------------------------

class VisionProcessingResponse(BaseModel):
    """Returned when a vision processing task is enqueued."""
    session_id: UUID
    task_id: str
    status: str = "pending"


class VisionStatusResponse(BaseModel):
    """Returned when checking the status of a vision processing task."""
    session_id: UUID
    task_id: str
    status: str
    progress: Optional[float] = None  # 0.0 - 1.0
    result: Optional[VisionResultSchema] = None
    error: Optional[str] = None
