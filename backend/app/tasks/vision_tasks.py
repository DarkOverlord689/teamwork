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
                    .values(session_metrics=result_data)
                )
                db.execute(stmt)
                db.commit()
            else:
                logger.info("No session_metrics to persist for session %s", session_id)
    except Exception:
        logger.exception("Failed to save visual metrics for session %s", session_id)
    finally:
        engine.dispose()


def _load_key_moments_from_db(session_id: str) -> list:
    """Load serialised key moments from result_data['key_moments'] in the database."""
    from sqlalchemy import create_engine, select
    from sqlalchemy.orm import Session

    from app.models import AnalysisSession

    sync_url = settings.database_url
    engine = create_engine(sync_url)

    try:
        with Session(engine) as db:
            stmt = select(AnalysisSession).where(AnalysisSession.id == session_id)
            session = db.execute(stmt).scalar_one_or_none()
            if session is None:
                return []
            result_data = getattr(session, "result_data", None) or {}
            return result_data.get("key_moments", [])
    except Exception:
        logger.exception("Failed to load key moments for session %s", session_id)
        return []
    finally:
        engine.dispose()


def _deserialize_key_moments(key_moments_data: list) -> list:
    """Deserialize a list of dicts into KeyMoment objects."""
    from app.core.audio.data_types import KeyMoment, MomentType

    result = []
    for km_dict in key_moments_data:
        try:
            moment_type_str = km_dict.get("moment_type", "speech_onset")
            # Handle both enum value strings and enum names
            try:
                moment_type = MomentType(moment_type_str)
            except ValueError:
                moment_type = MomentType.SPEECH_ONSET
            result.append(KeyMoment(
                timestamp_ms=float(km_dict.get("timestamp_ms", 0.0)),
                speaker_id=km_dict.get("speaker_id", ""),
                moment_type=moment_type,
                priority=int(km_dict.get("priority", 1)),
                context_text=km_dict.get("context_text", ""),
            ))
        except Exception:
            logger.warning("Skipping malformed key moment dict: %s", km_dict)
    return result


def _save_smart_vision_result(session_id: str, result_dict: dict) -> None:
    """Persist smart vision result to result_data['smart_vision'] in the database."""
    from sqlalchemy import create_engine, select as sa_select, update
    from sqlalchemy.orm import Session

    from app.models import AnalysisSession

    sync_url = settings.database_url
    engine = create_engine(sync_url)

    try:
        with Session(engine) as db:
            row = db.execute(
                sa_select(AnalysisSession).where(AnalysisSession.id == session_id)
            ).scalar_one_or_none()
            existing = dict(getattr(row, "result_data", None) or {}) if row else {}
            existing["smart_vision"] = result_dict
            stmt = (
                update(AnalysisSession)
                .where(AnalysisSession.id == session_id)
                .values(result_data=existing)
            )
            db.execute(stmt)
            db.commit()
            logger.info("Smart vision result saved for session %s", session_id)
    except Exception:
        logger.exception("Failed to save smart vision result for session %s", session_id)
        raise
    finally:
        engine.dispose()


def _load_audio_result_from_db(session_id: str):
    """Load and reconstruct audio result from the database."""
    from sqlalchemy import create_engine, select
    from sqlalchemy.orm import Session

    from app.models import AnalysisSession

    sync_url = settings.database_url
    engine = create_engine(sync_url)

    try:
        with Session(engine) as db:
            stmt = select(AnalysisSession).where(AnalysisSession.id == session_id)
            session = db.execute(stmt).scalar_one_or_none()
            if session is None:
                return None
            result_data = getattr(session, "result_data", None) or {}
            audio_data = result_data.get("audio", {})
            if not audio_data:
                return None
            return audio_data
    except Exception:
        logger.exception("Failed to load audio result for session %s", session_id)
        return None
    finally:
        engine.dispose()


@celery_app.task(bind=True, name="vision.process_video_smart", max_retries=1)
def process_video_smart_task(
    self,
    video_path: str,
    session_id: str,
    config_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Process a video through the smart vision pipeline using audio-guided key frames.

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
        Serialised ``VisionResult.to_dict()`` with smart mode fields.
    """
    from app.core.vision.pipeline import VisionPipeline

    logger.info("Starting smart vision task for session %s – video: %s", session_id, video_path)

    _update_session_status(session_id, "processing")

    try:
        # 1. Load key moments from DB
        key_moments_data = _load_key_moments_from_db(session_id)
        key_moments = _deserialize_key_moments(key_moments_data)
        logger.info(
            "Loaded %d key moments for session %s", len(key_moments), session_id
        )

        # 2. Load audio result (for speaker_segments)
        audio_data = _load_audio_result_from_db(session_id)
        speaker_segments: list = []
        if audio_data:
            from app.core.audio.data_types import SpeakerSegment
            for seg in audio_data.get("segments", []):
                try:
                    speaker_segments.append(SpeakerSegment(
                        start=seg["start"],
                        end=seg["end"],
                        speaker_id=seg["speaker_id"],
                        confidence=seg.get("confidence"),
                    ))
                except Exception:
                    logger.warning("Skipping malformed speaker segment: %s", seg)

        # 3. Build smart VisionConfig
        config = _build_vision_config(config_overrides)
        config.smart_frame_selection_enabled = True
        config.use_multimodal = settings.use_multimodal
        config.multimodal_timeout_seconds = settings.multimodal_timeout_seconds

        # 4. Run smart pipeline
        self.update_state(state="PROGRESS", meta={"progress": 0.0, "stage": "loading_models"})

        with VisionPipeline(config) as pipeline:
            self.update_state(
                state="PROGRESS", meta={"progress": 0.05, "stage": "extracting_targeted_frames"}
            )

            result = pipeline.process_video_smart(
                video_path,
                key_moments=key_moments,
                speaker_segments=speaker_segments,
                progress_callback=lambda pct, stage: self.update_state(
                    state="PROGRESS",
                    meta={"progress": round(pct, 4), "stage": stage},
                ),
                session_id=session_id,
            )

        self.update_state(state="PROGRESS", meta={"progress": 1.0, "stage": "saving_results"})

        result_dict = result.to_dict()

        # 5. Save smart vision result and persist session metrics so that
        #    GET /api/v1/vision/results/{session_id} can serve frame data
        #    (including frame_thumbnails) via the same code path as the
        #    regular pipeline.
        _save_smart_vision_result(session_id, result_dict)
        # Only overwrite session_metrics when we actually extracted frames.
        # If smart pipeline yields 0 frames (e.g. audio key moments missing),
        # preserve the metrics already saved by the regular vision task.
        if result.total_frames > 0 or len(result.frames) > 0:
            _save_visual_metrics(session_id, result_dict)
        else:
            logger.warning(
                "Smart vision task produced 0 frames for session %s; "
                "skipping session_metrics overwrite to preserve regular pipeline results.",
                session_id,
            )
        _update_session_status(session_id, "completed", result_data=result_dict)

        logger.info(
            "Smart vision task complete for session %s – %.1fs processing time",
            session_id,
            result.processing_time_seconds,
        )

        # 6. Dispatch fusion task
        from app.tasks.fusion_tasks import process_fusion_task
        process_fusion_task.delay(session_id)
        logger.info("Fusion task dispatched for session %s", session_id)

        return result_dict

    except FileNotFoundError:
        error_msg = f"Video file not found: {video_path}"
        logger.error(error_msg)
        _update_session_status(session_id, "failed", error=error_msg)
        raise

    except Exception as exc:
        error_msg = f"Smart vision processing failed: {exc}\n{traceback.format_exc()}"
        logger.exception("Smart vision task failed for session %s", session_id)
        _update_session_status(session_id, "failed", error=error_msg)
        raise


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
                session_id=session_id,
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

        # Chain: trigger audio processing now that vision is complete
        from app.tasks.audio_tasks import process_audio_task
        process_audio_task.delay(video_path, session_id)
        logger.info("Audio task dispatched for session %s", session_id)

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
