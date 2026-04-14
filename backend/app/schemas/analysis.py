from typing import Any, Dict, Optional
from pydantic import BaseModel, model_validator
from uuid import UUID
from datetime import datetime


class AnalysisResponse(BaseModel):
    id: UUID
    group_id: UUID
    video_path: str
    duration_seconds: int | None
    processed_at: datetime | None
    status: str
    topic_description: Optional[str] = None
    # Fusion result fields (populated from result_data["fusion"] when available)
    group_metrics: Optional[Dict[str, Any]] = None
    rubric_scores: Optional[Dict[str, Any]] = None
    explanation: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True


class MetricsResponse(BaseModel):
    student_id: UUID
    visual_metrics: dict
    audio_metrics: dict
    rubric_scores: dict