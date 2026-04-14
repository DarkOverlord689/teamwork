"""Celery tasks package."""

from app.tasks.celery_app import celery_app
from app.tasks.audio_tasks import process_audio_task
from app.tasks.fusion_tasks import process_fusion_task
from app.tasks.vision_tasks import process_video_task

__all__ = [
    "celery_app",
    "process_audio_task",
    "process_fusion_task",
    "process_video_task",
]
