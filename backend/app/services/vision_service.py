"""Vision processing service layer.

Provides a high-level API for starting, monitoring, and retrieving
vision analysis results.  All database operations are async.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import AnalysisSession
from app.tasks.vision_tasks import process_video_task

logger = logging.getLogger(__name__)


class VisionService:
    """Manage vision analysis lifecycle via Celery + database."""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def start_analysis(
        self,
        video_path: str,
        session_id: UUID,
        config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Dispatch a vision processing task to Celery.

        Parameters
        ----------
        video_path : str
            Absolute path to the video file.
        session_id : UUID
            Primary key of the ``AnalysisSession`` to update.
        config : dict, optional
            VisionConfig override values.

        Returns
        -------
        str
            Celery task ID.
        """
        # Update session status to queued
        stmt = select(AnalysisSession).where(AnalysisSession.id == session_id)
        result = await self.db.execute(stmt)
        session = result.scalar_one_or_none()

        if session is None:
            raise ValueError(f"AnalysisSession {session_id} not found")

        session.status = "queued"
        await self.db.commit()

        # Dispatch Celery task
        task = process_video_task.delay(
            video_path=video_path,
            session_id=str(session_id),
            config_overrides=config,
        )

        logger.info(
            "Vision analysis dispatched: session=%s, task=%s",
            session_id, task.id,
        )
        return task.id

    async def get_analysis_status(self, session_id: UUID) -> Dict[str, Any]:
        """Return the current status of a vision analysis.

        Returns
        -------
        dict
            ``{"session_id", "status", "duration_seconds", "processed_at"}``.
        """
        stmt = select(AnalysisSession).where(AnalysisSession.id == session_id)
        result = await self.db.execute(stmt)
        session = result.scalar_one_or_none()

        if session is None:
            raise ValueError(f"AnalysisSession {session_id} not found")

        return {
            "session_id": str(session.id),
            "status": session.status,
            "duration_seconds": session.duration_seconds,
            "processed_at": session.processed_at.isoformat() if session.processed_at else None,
        }

    async def get_analysis_results(self, session_id: UUID) -> Dict[str, Any]:
        """Retrieve the full vision analysis results.

        Results are stored as JSON in ``AnalysisSession.session_metrics`` by the
        vision task.  The stored value is the full ``VisionResult.to_dict()`` dict
        (with keys: video_path, frames, session_metrics, total_frames, etc.).
        This method flattens that dict to the top level so the response matches
        the ``VisionResult`` shape the frontend expects.

        Returns
        -------
        dict
            Flattened vision result fields (video_path, frames, session_metrics,
            total_frames, fps_processed, duration_seconds, …) plus session
            identity fields (session_id, status, processed_at).
        """
        stmt = select(AnalysisSession).where(AnalysisSession.id == session_id)
        result = await self.db.execute(stmt)
        session = result.scalar_one_or_none()

        if session is None:
            raise ValueError(f"AnalysisSession {session_id} not found")

        status_info: Dict[str, Any] = {
            "session_id": str(session.id),
            "status": session.status,
            "duration_seconds": session.duration_seconds,
            "processed_at": session.processed_at.isoformat() if session.processed_at else None,
        }

        session_metrics = getattr(session, "session_metrics", None)
        if session_metrics:
            # Flatten the stored VisionResult dict to the top level so that
            # fields like `frames`, `video_path`, and `session_metrics` are
            # directly accessible on the response object (VisionResult shape).
            # session_id / status / processed_at take precedence if there is a
            # key collision with the stored data.
            return {**session_metrics, **status_info}

        return status_info
