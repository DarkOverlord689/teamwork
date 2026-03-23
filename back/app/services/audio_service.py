"""Audio processing service layer.

Provides a high-level API for starting, monitoring, and retrieving
audio analysis results.  All database operations are async.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import AnalysisSession
from app.tasks.audio_tasks import process_audio_task

logger = logging.getLogger(__name__)


class AudioService:
    """Manage audio analysis lifecycle via Celery + database."""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def start_analysis(
        self,
        session_id: UUID,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Dispatch an audio processing task to Celery.

        Parameters
        ----------
        session_id : UUID
            Primary key of the ``AnalysisSession`` to process.
        config : dict, optional
            AudioConfig override values.

        Returns
        -------
        dict
            ``{"session_id", "task_id", "status"}``.
        """
        stmt = select(AnalysisSession).where(AnalysisSession.id == session_id)
        result = await self.db.execute(stmt)
        session = result.scalar_one_or_none()

        if session is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"AnalysisSession {session_id} not found",
            )

        if not session.video_path:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Session has no video_path set",
            )

        session.status = "queued"
        await self.db.commit()

        # Dispatch Celery task
        task = process_audio_task.delay(
            video_path=session.video_path,
            session_id=str(session_id),
            config_overrides=config,
        )

        logger.info(
            "Audio analysis dispatched: session=%s, task=%s",
            session_id,
            task.id,
        )
        return {
            "session_id": str(session_id),
            "task_id": task.id,
            "status": "queued",
        }

    async def get_analysis_status(self, session_id: UUID) -> Dict[str, Any]:
        """Return the current status of an audio analysis.

        Returns
        -------
        dict
            ``{"session_id", "task_id", "status", "progress", "error"}``.
        """
        stmt = select(AnalysisSession).where(AnalysisSession.id == session_id)
        result = await self.db.execute(stmt)
        session = result.scalar_one_or_none()

        if session is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"AnalysisSession {session_id} not found",
            )

        return {
            "session_id": str(session.id),
            "task_id": None,
            "status": session.status,
            "progress": None,
            "error": None,
        }

    async def get_analysis_results(self, session_id: UUID) -> Dict[str, Any]:
        """Retrieve the full audio analysis results.

        Results are stored as JSON in ``AnalysisSession.result_data``.
        Raises 409 if the analysis is not yet complete.

        Returns
        -------
        dict
            The serialised ``AudioResult`` dict stored at task completion.
        """
        stmt = select(AnalysisSession).where(AnalysisSession.id == session_id)
        result = await self.db.execute(stmt)
        session = result.scalar_one_or_none()

        if session is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"AnalysisSession {session_id} not found",
            )

        if session.status != "completed":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Analysis is not yet complete (status: {session.status})",
            )

        result_data = getattr(session, "result_data", None) or {}
        return result_data

    async def get_transcripts(self, session_id: UUID) -> List[Dict[str, Any]]:
        """Get just the transcript segments for a session.

        Returns
        -------
        list[dict]
            List of transcript segment dicts from the stored result_data.
        """
        results = await self.get_analysis_results(session_id)
        return results.get("transcripts", [])
