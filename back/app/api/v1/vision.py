"""Vision processing API endpoints.

Provides REST endpoints to start vision analysis on uploaded videos,
check processing status, and retrieve results.
"""

from __future__ import annotations

import logging
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import AnalysisSession
from app.schemas.vision import VisionProcessingResponse, VisionStatusResponse
from app.services.vision_service import VisionService

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class StartVisionRequest(BaseModel):
    """Request body for starting vision processing."""
    session_id: UUID = Field(..., description="ID of the analysis session to process")
    config_overrides: Optional[dict] = Field(
        None, description="Optional VisionConfig overrides (e.g. fps, enable_emotion)"
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/process",
    response_model=VisionProcessingResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start vision analysis",
)
async def start_vision_processing(
    request: StartVisionRequest,
    db: AsyncSession = Depends(get_db),
):
    """Enqueue a vision processing task for the given analysis session.

    The session must already exist and have a valid ``video_path``.
    Returns immediately with a task ID that can be used to poll status.
    """
    # Look up the session to get the video path
    stmt = select(AnalysisSession).where(AnalysisSession.id == request.session_id)
    result = await db.execute(stmt)
    session = result.scalar_one_or_none()

    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis session {request.session_id} not found",
        )

    if not session.video_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Session has no video_path set",
        )

    service = VisionService(db)
    try:
        task_id = await service.start_analysis(
            video_path=session.video_path,
            session_id=request.session_id,
            config=request.config_overrides,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )

    return VisionProcessingResponse(
        session_id=request.session_id,
        task_id=task_id,
        status="queued",
    )


@router.get(
    "/status/{session_id}",
    response_model=VisionStatusResponse,
    summary="Check vision analysis status",
)
async def get_vision_status(
    session_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """Return the current status of a vision processing task."""
    service = VisionService(db)
    try:
        status_info = await service.get_analysis_status(session_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )

    # Try to get Celery task status for richer info
    task_id = ""
    celery_status = status_info["status"]

    return VisionStatusResponse(
        session_id=session_id,
        task_id=task_id,
        status=celery_status,
    )


@router.get(
    "/results/{session_id}",
    summary="Get vision analysis results",
)
async def get_vision_results(
    session_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """Return the full vision analysis results for a completed session."""
    service = VisionService(db)
    try:
        results = await service.get_analysis_results(session_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )

    if results.get("status") != "completed":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Analysis is not yet complete (status: {results.get('status')})",
        )

    return results
