"""Fusion processing service layer (Module 2.3).

Provides a high-level API for starting, monitoring, and retrieving
multimodal fusion analysis results.  All database operations are async.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import AnalysisSession
from app.tasks.fusion_tasks import process_fusion_task

logger = logging.getLogger(__name__)


class FusionService:
    """Manage multimodal fusion analysis lifecycle via Celery + database."""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def start_analysis(
        self,
        session_id: UUID,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Dispatch a fusion processing task to Celery.

        Both the vision and audio analyses for the session must have already
        been completed and their results stored in ``result_data`` on the
        session row.

        Parameters
        ----------
        session_id : UUID
            Primary key of the ``AnalysisSession`` to process.
        config : dict, optional
            FusionConfig override values.

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

        task = process_fusion_task.delay(
            session_id=str(session_id),
            config_overrides=config,
        )

        logger.info(
            "Fusion analysis dispatched: session=%s, task=%s",
            session_id,
            task.id,
        )
        return {
            "session_id": str(session_id),
            "task_id": task.id,
            "status": "queued",
        }

    async def get_analysis_status(self, session_id: UUID) -> Dict[str, Any]:
        """Return the current status of a fusion analysis.

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
        """Retrieve the full fusion analysis results.

        Results are stored as JSON in ``AnalysisSession.result_data`` under the
        key ``"fusion"``.  Raises 409 if the analysis is not yet complete.

        Returns
        -------
        dict
            The serialised ``FusionResult`` dict stored at task completion.
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
        fusion_data = result_data.get("fusion", result_data)
        return fusion_data

    async def get_explanation(self, session_id: UUID) -> Dict[str, Any]:
        """Get just the narrative explanation for a completed fusion session.

        Returns
        -------
        dict
            The ``explanation`` sub-dict from the stored FusionResult.
        """
        results = await self.get_analysis_results(session_id)
        explanation = results.get("explanation")
        if explanation is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No explanation found in fusion results",
            )
        return explanation

    async def get_rubric_scores(self, session_id: UUID) -> Dict[str, Any]:
        """Get the UPAO rubric scores for a completed fusion session.

        Returns
        -------
        dict
            The ``rubric_scores`` sub-dict from the stored FusionResult.
        """
        results = await self.get_analysis_results(session_id)
        rubric = results.get("rubric_scores")
        if rubric is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No rubric scores found in fusion results",
            )
        return rubric
