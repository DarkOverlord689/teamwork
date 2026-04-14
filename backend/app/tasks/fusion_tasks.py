"""Celery tasks for multimodal fusion processing (Module 2.3).

Each task is a thin wrapper that loads vision + audio results from the
database, delegates to the fusion pipeline, and persists the output.
"""

from __future__ import annotations

import asyncio
import logging
import traceback
from typing import Any, Dict, Optional

from app.tasks.celery_app import celery_app
from app.utils.config import settings

logger = logging.getLogger(__name__)


def _build_fusion_config(overrides: Optional[Dict[str, Any]] = None):
    """Create a ``FusionConfig`` from application settings + optional overrides."""
    from app.core.fusion.config import FusionConfig

    config = FusionConfig(
        llm_model=settings.fusion_llm_model,
        openai_api_key=settings.openai_api_key,
        max_explanation_tokens=settings.fusion_max_explanation_tokens,
        enable_explanation=settings.fusion_enable_explanation,
    )

    if overrides:
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                logger.warning("Unknown FusionConfig override ignored: %s", key)

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

                # Read the current result_data and merge the new keys in so that
                # audio/vision data written by earlier tasks is not clobbered.
                row = db.execute(
                    sa_select(AnalysisSession).where(AnalysisSession.id == session_id)
                ).scalar_one_or_none()
                existing = dict(getattr(row, "result_data", None) or {}) if row else {}
                existing.update(result_data)
                stmt_result = (
                    update(AnalysisSession)
                    .where(AnalysisSession.id == session_id)
                    .values(
                        processed_at=datetime.utcnow(),
                        result_data=existing,
                    )
                )
                db.execute(stmt_result)

            db.commit()
    except Exception:
        logger.exception("Failed to update session %s status to %s", session_id, status)
    finally:
        engine.dispose()


def _load_prior_results(session_id: str) -> tuple[dict, dict]:
    """Load previously stored vision + audio results from the database.

    Returns
    -------
    tuple[dict, dict]
        ``(vision_result_dict, audio_result_dict)`` — may be empty dicts if not
        yet available.
    """
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
                return {}, {}
            result_data = getattr(session, "result_data", None) or {}
            # Prefer smart_vision over legacy vision when present
            vision_data = result_data.get("smart_vision") or result_data.get("vision", {})
            audio_data = result_data.get("audio", result_data)
            return vision_data, audio_data
    except Exception:
        logger.exception("Failed to load prior results for session %s", session_id)
        return {}, {}
    finally:
        engine.dispose()


def _reconstruct_vision_result(data: dict):
    """Reconstruct a VisionResult from a serialized dict.

    Returns a minimal VisionResult that FusionPipeline can consume even when
    the stored data is sparse (e.g. only session_metrics present).
    """
    from app.core.vision.data_types import (
        FrameResult,
        PersonMetrics,
        SessionMetrics,
        VisionResult,
    )

    frames = []
    for f in data.get("frames", []):
        from app.core.vision.data_types import PersonFrame

        persons = []
        for p in f.get("persons", []):
            persons.append(PersonFrame(person_id=p["person_id"], bbox=tuple(p.get("bbox", [0, 0, 0, 0]))))
        frames.append(FrameResult(
            frame_number=f.get("frame_number", 0),
            timestamp=f.get("timestamp", 0.0),
            persons=persons,
        ))

    session_metrics = None
    sm_data = data.get("session_metrics")
    if sm_data:
        person_metrics = [
            PersonMetrics(
                person_id=pm["person_id"],
                total_frames_seen=pm.get("total_frames_seen", 0),
                gaze_contact_percentage=pm.get("gaze_contact_percentage", 0.0),
                dominant_emotion=pm.get("dominant_emotion", "neutral"),
                emotion_distribution=pm.get("emotion_distribution", {}),
                average_body_orientation=pm.get("average_body_orientation", 0.0),
                gesture_counts=pm.get("gesture_counts", {}),
                attention_score=pm.get("attention_score", 0.0),
            )
            for pm in sm_data.get("per_person_metrics", [])
        ]
        session_metrics = SessionMetrics(
            total_persons=sm_data.get("total_persons", 0),
            total_frames=sm_data.get("total_frames", 0),
            duration=sm_data.get("duration", 0.0),
            per_person_metrics=person_metrics,
        )

    # Reconstruct smart-mode fields if present (guard with .get() for backward compat)
    key_moment_analyses = []
    for km_analysis in data.get("key_moment_analyses", []):
        try:
            from app.core.audio.data_types import MomentType
            from app.core.vision.data_types import MultimodalFrameAnalysis
            moment_type_val = km_analysis.get("moment_type", "speech_onset")
            try:
                moment_type = MomentType(moment_type_val)
            except (ValueError, KeyError):
                moment_type = MomentType.SPEECH_ONSET
            key_moment_analyses.append(MultimodalFrameAnalysis(
                frame_number=km_analysis.get("frame_number", 0),
                timestamp_ms=km_analysis.get("timestamp_ms", 0.0),
                speaker_id=km_analysis.get("speaker_id"),
                person_id=km_analysis.get("person_id"),
                moment_type=moment_type,
                multimodal_description=km_analysis.get("multimodal_description"),
                emotion_primary=km_analysis.get("emotion_primary"),
                emotion_confidence=km_analysis.get("emotion_confidence"),
                attention_direction=km_analysis.get("attention_direction"),
                engagement_level=km_analysis.get("engagement_level"),
                face_detected=km_analysis.get("face_detected", False),
                fallback_model=km_analysis.get("fallback_model"),
                fallback_reason=km_analysis.get("fallback_reason"),
            ))
        except Exception:
            logger.warning("Skipping malformed key_moment_analysis entry", exc_info=True)

    speaker_face_mappings = []
    for sfm_data in data.get("speaker_face_mappings", []):
        try:
            from app.core.vision.data_types import SpeakerFaceMapping
            speaker_face_mappings.append(SpeakerFaceMapping(
                speaker_id=sfm_data.get("speaker_id", ""),
                person_id=sfm_data.get("person_id"),
                confidence=sfm_data.get("confidence", 0.0),
                frame_count=sfm_data.get("frame_count", 0),
                uncertain=sfm_data.get("uncertain", True),
            ))
        except Exception:
            logger.warning("Skipping malformed speaker_face_mapping entry", exc_info=True)

    return VisionResult(
        video_path=data.get("video_path", ""),
        total_frames=data.get("total_frames", 0),
        fps_processed=data.get("fps_processed", 0.0),
        duration_seconds=data.get("duration_seconds", 0.0),
        frames=frames,
        processing_time_seconds=data.get("processing_time_seconds", 0.0),
        person_embeddings=data.get("person_embeddings", {}),
        session_metrics=session_metrics,
        smart_mode=data.get("smart_mode", False),
        key_moment_analyses=key_moment_analyses,
        speaker_face_mappings=speaker_face_mappings,
    )


def _reconstruct_audio_result(data: dict):
    """Reconstruct an AudioResult from a serialized dict."""
    from app.core.audio.data_types import (
        AudioResult,
        AudioSessionMetrics,
        Interruption,
        SpeakerMetrics,
        SpeakerSegment,
        SpeakerTurn,
        TranscriptSegment,
        WordTimestamp,
    )

    segments = [
        SpeakerSegment(
            start=s["start"],
            end=s["end"],
            speaker_id=s["speaker_id"],
            confidence=s.get("confidence"),
        )
        for s in data.get("segments", [])
    ]

    turns = [
        SpeakerTurn(
            start=t["start"],
            end=t["end"],
            speaker_id=t["speaker_id"],
            duration=t["duration"],
            segment_count=t.get("segment_count", 1),
        )
        for t in data.get("turns", [])
    ]

    transcripts = []
    for tr in data.get("transcripts", []):
        words = [
            WordTimestamp(
                word=w["word"],
                start=w["start"],
                end=w["end"],
                confidence=w["confidence"],
            )
            for w in tr.get("words", [])
        ]
        transcripts.append(
            TranscriptSegment(
                start=tr["start"],
                end=tr["end"],
                speaker_id=tr["speaker_id"],
                text=tr["text"],
                words=words,
            )
        )

    interruptions = [
        Interruption(
            time=i["time"],
            interrupter_id=i["interrupter_id"],
            interrupted_id=i["interrupted_id"],
            overlap_duration=i["overlap_duration"],
            interruption_type=i.get("interruption_type", "disruptive"),
        )
        for i in data.get("interruptions", [])
    ]

    session_metrics = None
    sm_data = data.get("session_metrics")
    if sm_data:
        per_speaker = [
            SpeakerMetrics(
                speaker_id=pm["speaker_id"],
                speaking_time_seconds=pm.get("speaking_time_seconds", 0.0),
                turn_count=pm.get("turn_count", 0),
                interruption_count=pm.get("interruption_count", 0),
                interrupted_count=pm.get("interrupted_count", 0),
                avg_turn_duration=pm.get("avg_turn_duration", 0.0),
                participation_ratio=pm.get("participation_ratio", 0.0),
                back_channel_count=pm.get("back_channel_count", 0),
            )
            for pm in sm_data.get("per_speaker_metrics", [])
        ]
        session_metrics = AudioSessionMetrics(
            total_speakers=sm_data.get("total_speakers", 0),
            duration=sm_data.get("duration", 0.0),
            total_speaking_time=sm_data.get("total_speaking_time", 0.0),
            silence_ratio=sm_data.get("silence_ratio", 0.0),
            participation_cv=sm_data.get("participation_cv", 0.0),
            turn_alternation_rate=sm_data.get("turn_alternation_rate", 0.0),
            per_speaker_metrics=per_speaker,
        )

    return AudioResult(
        video_path=data.get("video_path", ""),
        duration_seconds=data.get("duration_seconds", 0.0),
        sample_rate=data.get("sample_rate", 16000),
        segments=segments,
        turns=turns,
        transcripts=transcripts,
        interruptions=interruptions,
        processing_time_seconds=data.get("processing_time_seconds", 0.0),
        session_metrics=session_metrics,
    )


def _persist_llm_generated_fields(session_id: str, fusion_result) -> None:
    """Persist topic_description and per-student intervention_summary from the LLM output.

    Writes ``topic_description`` to ``AnalysisSession`` and each student's
    ``intervention_summary`` to the corresponding system ``RubricScore`` row
    (creating one if it does not already exist).
    """
    import uuid as _uuid_module

    from sqlalchemy import create_engine, select, update
    from sqlalchemy.orm import Session

    from app.models import AnalysisSession, RubricScore

    explanation = fusion_result.explanation
    topic_description: str | None = getattr(explanation, "topic_description", None)
    intervention_summaries: dict = getattr(explanation, "intervention_summaries", {})

    if not topic_description and not intervention_summaries:
        return

    sync_url = settings.database_url
    engine = create_engine(sync_url)

    try:
        with Session(engine) as db:
            # -- topic_description on AnalysisSession --
            if topic_description:
                db.execute(
                    update(AnalysisSession)
                    .where(AnalysisSession.id == session_id)
                    .values(topic_description=topic_description)
                )

            # -- intervention_summary on RubricScore --
            for student_id_str, summary in intervention_summaries.items():
                if not summary:
                    continue

                # student_id in intervention_summaries is a speaker/person ID string,
                # not necessarily a UUID stored in the students table. We store it
                # as a UUID if possible; otherwise we skip (the JSONB fusion data
                # already carries the value).
                try:
                    student_uuid = _uuid_module.UUID(student_id_str)
                except ValueError:
                    logger.debug(
                        "intervention_summary for non-UUID student id '%s' not persisted to RubricScore; "
                        "available in fusion JSON.",
                        student_id_str,
                    )
                    continue

                stmt = select(RubricScore).where(
                    RubricScore.session_id == session_id,
                    RubricScore.student_id == student_uuid,
                    RubricScore.evaluator_type == "system",
                )
                rubric_row = db.execute(stmt).scalar_one_or_none()

                if rubric_row is None:
                    rubric_row = RubricScore(
                        id=_uuid_module.uuid4(),
                        session_id=_uuid_module.UUID(session_id),
                        student_id=student_uuid,
                        evaluator_type="system",
                    )
                    db.add(rubric_row)

                rubric_row.intervention_summary = summary

            db.commit()
    except Exception:
        logger.exception(
            "Failed to persist LLM-generated fields for session %s", session_id
        )
    finally:
        engine.dispose()


@celery_app.task(bind=True, name="fusion.process_fusion", max_retries=1)
def process_fusion_task(
    self,
    session_id: str,
    config_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Process a session through the multimodal fusion pipeline.

    This task expects that both vision and audio analyses are already stored
    in ``AnalysisSession.result_data`` (written by Module 2.1 / 2.2 tasks).

    Parameters
    ----------
    session_id : str
        UUID of the ``AnalysisSession`` row to process and update.
    config_overrides : dict, optional
        Key/value pairs to override default ``FusionConfig`` fields.

    Returns
    -------
    dict
        Serialised ``FusionResult.to_dict()``.
    """
    from app.core.fusion.fusion_pipeline import FusionPipeline

    logger.info("Starting fusion task for session %s", session_id)
    _update_session_status(session_id, "processing")

    try:
        config = _build_fusion_config(config_overrides)

        self.update_state(
            state="PROGRESS",
            meta={"progress": 0.0, "stage": "loading_prior_results"},
        )

        vision_data, audio_data = _load_prior_results(session_id)

        if not audio_data:
            raise ValueError(
                f"No audio result found for session {session_id}. "
                "Run audio analysis before fusion."
            )

        self.update_state(
            state="PROGRESS",
            meta={"progress": 0.05, "stage": "reconstructing_results"},
        )

        vision_result = _reconstruct_vision_result(vision_data)
        audio_result = _reconstruct_audio_result(audio_data)

        pipeline = FusionPipeline(config)

        # FusionPipeline.process is async; run it in an event loop from Celery's
        # synchronous context.
        fusion_result = asyncio.run(
            pipeline.process(
                vision_result,
                audio_result,
                progress_callback=lambda pct, stage: self.update_state(
                    state="PROGRESS",
                    meta={"progress": round(0.10 + pct * 0.85, 4), "stage": stage},
                ),
            )
        )

        self.update_state(
            state="PROGRESS",
            meta={"progress": 1.0, "stage": "saving_results"},
        )

        result_dict = fusion_result.to_dict()

        # Merge fusion results into the session result_data
        _update_session_status(session_id, "completed", result_data={"fusion": result_dict})

        # Persist LLM-generated topic_description and per-student intervention_summary
        _persist_llm_generated_fields(session_id, fusion_result)

        logger.info(
            "Fusion task complete for session %s – %.1fs processing time",
            session_id,
            fusion_result.processing_time_seconds,
        )
        return result_dict

    except FileNotFoundError:
        error_msg = f"File not found during fusion for session {session_id}"
        logger.error(error_msg)
        _update_session_status(session_id, "failed", error=error_msg)
        raise

    except Exception as exc:
        error_msg = f"Fusion processing failed: {exc}\n{traceback.format_exc()}"
        logger.exception("Fusion task failed for session %s", session_id)
        _update_session_status(session_id, "failed", error=error_msg)
        raise
