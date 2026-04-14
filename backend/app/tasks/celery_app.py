"""Celery application configuration.

The broker and backend are both Redis, configured via
``app.utils.config.settings.redis_url``.
"""

from celery import Celery

from app.utils.config import settings

celery_app = Celery(
    "smatc",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)

# Explicitly include every task module so the worker registers all tasks on startup
celery_app.conf.include = [
    "app.tasks.vision_tasks",
    "app.tasks.audio_tasks",
    "app.tasks.fusion_tasks",
]
