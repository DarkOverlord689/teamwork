"""Audio processing API endpoints.

Provides REST endpoints to start audio analysis on uploaded videos,
check processing status, and retrieve results and transcripts.
"""

from __future__ import annotations

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.schemas.audio import (
    AudioProcessingResponse,
    AudioResultSchema,
    AudioStatusResponse,
    StartAudioRequest,
    TranscriptSegmentSchema,
)
from app.services.audio_service import AudioService

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/process",
    response_model=AudioProcessingResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start audio analysis",
)
async def start_audio_processing(
    request: StartAudioRequest,
    db: AsyncSession = Depends(get_db),
) -> AudioProcessingResponse:
    """Enqueue an audio processing task for the given analysis session.

    The session must already exist and have a valid ``video_path``.
    Returns immediately with a task ID that can be used to poll status.
    """
    service = AudioService(db)
    result = await service.start_analysis(
        session_id=request.session_id,
        config=request.config_overrides,
    )
    return AudioProcessingResponse(**result)


@router.get(
    "/status/{session_id}",
    response_model=AudioStatusResponse,
    summary="Check audio analysis status",
)
async def get_audio_status(
    session_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> AudioStatusResponse:
    """Return the current status of an audio processing task."""
    service = AudioService(db)
    status_info = await service.get_analysis_status(session_id)
    return AudioStatusResponse(**status_info)


@router.get(
    "/results/{session_id}",
    response_model=AudioResultSchema,
    summary="Get audio analysis results",
)
async def get_audio_results(
    session_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> AudioResultSchema:
    """Return the full audio analysis results for a completed session.

    Returns 409 if the analysis is not yet complete.
    """
    service = AudioService(db)
    result_data = await service.get_analysis_results(session_id)
    return AudioResultSchema(**result_data)


@router.get(
    "/results/{session_id}/transcripts",
    response_model=list[TranscriptSegmentSchema],
    summary="Get transcript segments",
)
async def get_audio_transcripts(
    session_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> list[TranscriptSegmentSchema]:
    """Return only the transcript segments for a completed session."""
    service = AudioService(db)
    transcripts = await service.get_transcripts(session_id)
    return [TranscriptSegmentSchema(**t) for t in transcripts]
