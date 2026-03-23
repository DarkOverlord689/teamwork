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

        The raw ``VisionResult`` is stored in the Celery result backend.
        This method retrieves the Celery result and returns it.

        Returns
        -------
        dict
            Status info plus the ``VisionResult`` dict if complete.
        """
        status_info = await self.get_analysis_status(session_id)

        if status_info["status"] != "completed":
            return status_info

        # Results are in the Celery result backend; the task returns VisionResult.to_dict()
        # The caller (API layer) can also check Celery's AsyncResult directly
        return status_info
