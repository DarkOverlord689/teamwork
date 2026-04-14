"""Celery tasks for audio processing.

Each task is a thin wrapper that delegates to the audio pipeline and
persists results to the database.
"""

from __future__ import annotations

import logging
import traceback
from typing import Any, Dict, Optional

from app.tasks.celery_app import celery_app
from app.utils.config import settings

logger = logging.getLogger(__name__)


def _build_audio_config(overrides: Optional[Dict[str, Any]] = None):
    """Create an ``AudioConfig`` from application settings + optional overrides."""
    from app.core.audio.config import AudioConfig

    config = AudioConfig(
        audio_sample_rate=settings.audio_sample_rate,
        audio_device=settings.audio_device,
        pyannote_auth_token=settings.pyannote_auth_token,
        openai_api_key=settings.openai_api_key,
        whisper_language=settings.whisper_language,
        enable_diarization=settings.enable_audio_diarization,
        enable_transcription=settings.enable_audio_transcription,
    )

    if overrides:
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                logger.warning("Unknown AudioConfig override ignored: %s", key)

    return config


def _update_session_status(
    session_id: str,
    status: str,
    result_data: Optional[dict] = None,
    error: Optional[str] = None,
) -> None:
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
                from sqlalchemy import select as sa_select

                # Merge audio result under "audio" key so other tasks' data is preserved.
                row = db.execute(
                    sa_select(AnalysisSession).where(AnalysisSession.id == session_id)
                ).scalar_one_or_none()
                existing = dict(getattr(row, "result_data", None) or {}) if row else {}
                existing["audio"] = result_data
                stmt_result = (
                    update(AnalysisSession)
                    .where(AnalysisSession.id == session_id)
                    .values(
                        processed_at=datetime.utcnow(),
                        duration_seconds=int(result_data.get("duration_seconds", 0)),
                        result_data=existing,
                    )
                )
                db.execute(stmt_result)

            db.commit()
    except Exception:
        logger.exception("Failed to update session %s status to %s", session_id, status)
    finally:
        engine.dispose()


def _save_audio_metrics(session_id: str, result_data: dict) -> None:
    """Persist audio processing results to the database.

    Audio pipeline produces speaker IDs (speaker_0, speaker_1) rather than
    student UUIDs — cross-modal alignment is handled by Module 2.3.  For now
    the full result dict is saved as JSON in result_data on the session row.
    AudioMetrics rows require a student_id FK so they cannot be written until
    alignment is complete.
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
                # Read the current result_data, merge the audio results under the
                # "audio" key, and write the full dict back.  This avoids clobbering
                # data written by other tasks (e.g. vision, fusion).
                from sqlalchemy import select as sa_select

                row = db.execute(
                    sa_select(AnalysisSession).where(AnalysisSession.id == session_id)
                ).scalar_one_or_none()
                existing = dict(getattr(row, "result_data", None) or {}) if row else {}
                existing["audio"] = result_data
                stmt = (
                    update(AnalysisSession)
                    .where(AnalysisSession.id == session_id)
                    .values(result_data=existing)
                )
                db.execute(stmt)
                db.commit()
                logger.info(
                    "Audio session_metrics saved to result_data['audio'] for session %s "
                    "(AudioMetrics rows require student alignment — deferred to Module 2.3)",
                    session_id,
                )
            else:
                logger.info("No session_metrics to persist for session %s", session_id)
    except Exception:
        logger.exception("Failed to save audio metrics for session %s", session_id)
        raise
    finally:
        engine.dispose()


def _save_key_moments(session_id: str, key_moments_data: list) -> None:
    """Persist serialised key moments to result_data['key_moments'] in the database."""
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
            existing["key_moments"] = key_moments_data
            stmt = (
                update(AnalysisSession)
                .where(AnalysisSession.id == session_id)
                .values(result_data=existing)
            )
            db.execute(stmt)
            db.commit()
            logger.info(
                "Saved %d key moments for session %s", len(key_moments_data), session_id
            )
    except Exception:
        logger.exception("Failed to save key moments for session %s", session_id)
        raise
    finally:
        engine.dispose()


def _run_key_frame_selection_and_dispatch(
    session_id: str,
    video_path: str,
    audio_result: Any,
    result_dict: Dict[str, Any],
) -> None:
    """Run KeyFrameSelector on audio result and dispatch smart vision task."""
    from app.core.audio.key_frame_selector import KeyFrameSelector

    config = _build_audio_config()
    selector = KeyFrameSelector(config)

    segments = getattr(audio_result, "segments", [])
    transcripts = getattr(audio_result, "transcripts", [])
    interruptions = getattr(audio_result, "interruptions", [])
    turns = getattr(audio_result, "turns", [])

    key_moments = selector.select(
        segments=segments,
        transcript=transcripts,
        interruptions=interruptions,
        turns=turns,
    )

    # Serialize key_moments for DB storage and Celery transport
    key_moments_data = [
        {
            "timestamp_ms": km.timestamp_ms,
            "speaker_id": km.speaker_id,
            "moment_type": km.moment_type.value
            if hasattr(km.moment_type, "value")
            else str(km.moment_type),
            "priority": km.priority,
            "context_text": km.context_text,
        }
        for km in key_moments
    ]

    _save_key_moments(session_id, key_moments_data)
    logger.info(
        "Key frame selection produced %d moments for session %s",
        len(key_moments_data),
        session_id,
    )

    from app.tasks.vision_tasks import process_video_smart_task
    process_video_smart_task.delay(video_path, session_id)
    logger.info("Smart vision task dispatched for session %s", session_id)


@celery_app.task(bind=True, name="audio.process_audio", max_retries=1)
def process_audio_task(
    self,
    video_path: str,
    session_id: str,
    config_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Process a video through the audio pipeline.

    Parameters
    ----------
    video_path : str
        Absolute path to the video file on disk.
    session_id : str
        UUID of the ``AnalysisSession`` row to update.
    config_overrides : dict, optional
        Key/value pairs to override default ``AudioConfig`` fields.

    Returns
    -------
    dict
        Serialised ``AudioResult.to_dict()``.
    """
    from app.core.audio.pipeline import AudioPipeline

    logger.info("Starting audio task for session %s – video: %s", session_id, video_path)

    # Mark as processing
    _update_session_status(session_id, "processing")

    try:
        config = _build_audio_config(config_overrides)

        # Report initial progress
        self.update_state(state="PROGRESS", meta={"progress": 0.0, "stage": "loading_models"})

        with AudioPipeline(config) as pipeline:
            # Report that models are loaded
            self.update_state(state="PROGRESS", meta={"progress": 0.05, "stage": "extracting_audio"})

            result = pipeline.process_audio(
                video_path,
                progress_callback=lambda pct, stage: self.update_state(
                    state="PROGRESS",
                    meta={"progress": round(pct, 4), "stage": stage},
                ),
            )

        # Report completion
        self.update_state(state="PROGRESS", meta={"progress": 1.0, "stage": "saving_results"})

        result_dict = result.to_dict()

        # Persist results to session result_data (AudioMetrics rows deferred to Module 2.3)
        _save_audio_metrics(session_id, result_dict)

        # Persist success
        _update_session_status(session_id, "completed", result_data=result_dict)

        logger.info(
            "Audio task complete for session %s – %.1fs processing time",
            session_id,
            result.processing_time_seconds,
        )

        # Chain: dispatch next task based on smart frame selection flag
        if settings.sfs_enabled:
            _run_key_frame_selection_and_dispatch(session_id, video_path, result, result_dict)
        else:
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
        error_msg = f"Audio processing failed: {exc}\n{traceback.format_exc()}"
        logger.exception("Audio task failed for session %s", session_id)
        _update_session_status(session_id, "failed", error=error_msg)
        raise
