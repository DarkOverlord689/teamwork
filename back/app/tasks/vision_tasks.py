"""Celery tasks for vision processing.

Each task is a thin wrapper that delegates to the vision pipeline and
persists results to the database.
"""

from __future__ import annotations

import logging
import traceback
from typing import Any, Dict, Optional

from app.tasks.celery_app import celery_app
from app.utils.config import settings

logger = logging.getLogger(__name__)


def _build_vision_config(overrides: Optional[Dict[str, Any]] = None):
    """Create a ``VisionConfig`` from application settings + optional overrides."""
    from app.core.vision.config import VisionConfig

    config = VisionConfig(
        fps=settings.vision_frame_fps,
        max_frames=settings.vision_max_frames,
        min_face_confidence=settings.vision_face_confidence,
        gaze_camera_threshold=settings.vision_gaze_threshold,
        gesture_window_size=settings.vision_gesture_window,
        clip_model=settings.vision_clip_model,
        device=settings.vision_device,
        enable_emotion=settings.vision_enable_emotion,
        enable_gaze=settings.vision_enable_gaze,
        enable_gesture=settings.vision_enable_gesture,
        enable_pose=settings.vision_enable_pose,
        enable_tracking=settings.vision_enable_tracking,
    )

    if overrides:
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                logger.warning("Unknown VisionConfig override ignored: %s", key)

    return config


def _update_session_status(session_id: str, status: str, result_data: Optional[dict] = None, error: Optional[str] = None) -> None:
    """Synchronously update an AnalysisSession record in the database.

    Uses a synchronous SQLAlchemy session because Celery tasks run in a
    synchronous context.
    """
    from sqlalchemy import create_engine, update
    from sqlalchemy.orm import Session

    from app.models import AnalysisSession

    sync_url = settings.database_url
    engine = create_engine(sync_url)

    try:
        with Session(engine) as db:
            stmt = (
                update(AnalysisSession)
                .where(AnalysisSession.id == session_id)
                .values(status=status)
            )
            db.execute(stmt)

            if status == "completed" and result_data is not None:
                from datetime import datetime
                stmt_result = (
                    update(AnalysisSession)
                    .where(AnalysisSession.id == session_id)
                    .values(
                        processed_at=datetime.utcnow(),
                        duration_seconds=int(result_data.get("duration_seconds", 0)),
                    )
                )
                db.execute(stmt_result)

            db.commit()
    except Exception:
        logger.exception("Failed to update session %s status to %s", session_id, status)
    finally:
        engine.dispose()


def _save_visual_metrics(session_id: str, result_data: dict) -> None:
    """Persist vision processing results to the database.

    Stores the session_metrics and per-person aggregation data alongside
    the session for later retrieval without hitting the Celery result backend.
    """
    from sqlalchemy import create_engine, update
    from sqlalchemy.orm import Session

    from app.models import AnalysisSession

    sync_url = settings.database_url
    engine = create_engine(sync_url)

    try:
        with Session(engine) as db:
            session_metrics = result_data.get("session_metrics")
            if session_metrics:
                stmt = (
                    update(AnalysisSession)
                    .where(AnalysisSession.id == session_id)
                    .values(result_data=result_data)
                )
                db.execute(stmt)
                db.commit()
            else:
                logger.info("No session_metrics to persist for session %s", session_id)
    except Exception:
        logger.exception("Failed to save visual metrics for session %s", session_id)
    finally:
        engine.dispose()


@celery_app.task(bind=True, name="vision.process_video", max_retries=1)
def process_video_task(
    self,
    video_path: str,
    session_id: str,
    config_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Process a video through the vision pipeline.

    Parameters
    ----------
    video_path : str
        Absolute path to the video file on disk.
    session_id : str
        UUID of the ``AnalysisSession`` row to update.
    config_overrides : dict, optional
        Key/value pairs to override default ``VisionConfig`` fields.

    Returns
    -------
    dict
        Serialised ``VisionResult.to_dict()``.
    """
    from app.core.vision.pipeline import VisionPipeline

    logger.info("Starting vision task for session %s – video: %s", session_id, video_path)

    # Mark as processing
    _update_session_status(session_id, "processing")

    try:
        config = _build_vision_config(config_overrides)

        # Report initial progress
        self.update_state(state="PROGRESS", meta={"progress": 0.0, "stage": "loading_models"})

        with VisionPipeline(config) as pipeline:
            # Report that models are loaded
            self.update_state(state="PROGRESS", meta={"progress": 0.05, "stage": "extracting_frames"})

            result = pipeline.process_video(
                video_path,
                progress_callback=lambda pct, stage: self.update_state(
                    state="PROGRESS",
                    meta={"progress": round(pct, 4), "stage": stage},
                ),
            )

        # Report completion
        self.update_state(state="PROGRESS", meta={"progress": 1.0, "stage": "saving_results"})

        result_dict = result.to_dict()

        # Persist results to visual_metrics table
        _save_visual_metrics(session_id, result_dict)

        # Persist success
        _update_session_status(session_id, "completed", result_data=result_dict)

        logger.info(
            "Vision task complete for session %s – %.1fs processing time",
            session_id, result.processing_time_seconds,
        )
        return result_dict

    except FileNotFoundError as exc:
        error_msg = f"Video file not found: {video_path}"
        logger.error(error_msg)
        _update_session_status(session_id, "failed", error=error_msg)
        raise

    except Exception as exc:
        error_msg = f"Vision processing failed: {exc}\n{traceback.format_exc()}"
        logger.exception("Vision task failed for session %s", session_id)
        _update_session_status(session_id, "failed", error=error_msg)
        raise
