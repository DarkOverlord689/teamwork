from pydantic import BaseModel
from uuid import UUID
from datetime import datetime


class AnalysisResponse(BaseModel):
    id: UUID
    group_id: UUID
    video_path: str
    duration_seconds: int | None
    processed_at: datetime | None
    status: str

    class Config:
        from_attributes = True


class MetricsResponse(BaseModel):
    student_id: UUID
    visual_metrics: dict
    audio_metrics: dict
    rubric_scores: dict