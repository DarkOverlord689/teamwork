"""audio.py - Endpoints del pipeline de procesamiento de audio

Proporciona rutas REST para iniciar el análisis de audio,
consultar el estado del procesamiento y obtener los resultados
y transcripciones.
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


@router.post(
    "/process",
    response_model=AudioProcessingResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Iniciar análisis de audio",
)
async def start_audio_processing(
    request: StartAudioRequest,
    db: AsyncSession = Depends(get_db),
) -> AudioProcessingResponse:
    """Encola una tarea de procesamiento de audio para la sesión indicada.

    La sesión debe existir y tener un video_path válido.
    Retorna inmediatamente con un ID de tarea para consultar el estado.
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
    summary="Estado del análisis de audio",
)
async def get_audio_status(
    session_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> AudioStatusResponse:
    """Retorna el estado actual de una tarea de procesamiento de audio."""
    service = AudioService(db)
    status_info = await service.get_analysis_status(session_id)
    return AudioStatusResponse(**status_info)


@router.get(
    "/results/{session_id}",
    response_model=AudioResultSchema,
    summary="Obtener resultados del análisis de audio",
)
async def get_audio_results(
    session_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> AudioResultSchema:
    """Retorna los resultados completos del análisis de audio.

    Retorna 409 si el análisis aún no está completo.
    """
    service = AudioService(db)
    try:
        result_data = await service.get_analysis_results(session_id)
        return AudioResultSchema.model_validate(result_data)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(
            "Error al construir AudioResultSchema para sesión %s: %s", session_id, e
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@router.get(
    "/results/{session_id}/transcripts",
    response_model=list[TranscriptSegmentSchema],
    summary="Obtener segmentos de transcripción",
)
async def get_audio_transcripts(
    session_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> list[TranscriptSegmentSchema]:
    """Retorna solo los segmentos de transcripción de una sesión completada."""
    service = AudioService(db)
    try:
        transcripts = await service.get_transcripts(session_id)
        return [TranscriptSegmentSchema.model_validate(t) for t in transcripts]
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(
            "Error al construir transcripciones para sesión %s: %s", session_id, e
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )
