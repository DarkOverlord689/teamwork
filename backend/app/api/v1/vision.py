"""vision.py - Endpoints del pipeline de procesamiento de video

Proporciona rutas REST para iniciar el análisis de video,
consultar el estado del procesamiento y obtener los resultados.
También sirve las miniaturas de los frames guardados.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import AnalysisSession
from app.schemas.vision import VisionProcessingResponse, VisionStatusResponse
from app.services.vision_service import VisionService
from app.utils.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


class StartVisionRequest(BaseModel):
    """Solicitud para iniciar el procesamiento de video."""

    session_id: UUID = Field(..., description="ID de la sesión de análisis a procesar")
    config_overrides: Optional[dict] = Field(
        None, description="Opciones de configuración opcionales"
    )


@router.post(
    "/process",
    response_model=VisionProcessingResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Iniciar análisis de video",
)
async def start_vision_processing(
    request: StartVisionRequest,
    db: AsyncSession = Depends(get_db),
):
    """Encola una tarea de procesamiento de video para la sesión indicada."""
    stmt = select(AnalysisSession).where(AnalysisSession.id == request.session_id)
    result = await db.execute(stmt)
    session = result.scalar_one_or_none()

    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Sesión {request.session_id} no encontrada",
        )

    if not session.video_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="La sesión no tiene video_path",
        )

    service = VisionService(db)
    try:
        task_id = await service.start_analysis(
            video_path=session.video_path,
            session_id=request.session_id,
            config=request.config_overrides,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))

    return VisionProcessingResponse(
        session_id=request.session_id, task_id=task_id, status="queued"
    )


@router.get(
    "/status/{session_id}",
    response_model=VisionStatusResponse,
    summary="Estado del análisis de video",
)
async def get_vision_status(
    session_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """Retorna el estado actual del procesamiento de video."""
    service = VisionService(db)
    try:
        status_info = await service.get_analysis_status(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))

    return VisionStatusResponse(
        session_id=session_id, task_id="", status=status_info["status"]
    )


@router.get("/results/{session_id}", summary="Obtener resultados del análisis de video")
async def get_vision_results(
    session_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """Retorna los resultados completos del análisis de video."""
    service = VisionService(db)
    try:
        results = await service.get_analysis_results(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))

    if results.get("session_metrics") is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Resultados de visión no disponibles aún",
        )

    return results


@router.get(
    "/frames/{session_id}/{filename}",
    summary="Servir miniatura de frame",
    response_class=FileResponse,
)
async def get_frame_thumbnail(
    session_id: UUID,
    filename: str,
):
    """Retorna una miniatura JPEG guardada para la sesión y archivo indicados.

    Solo se sirven archivos .jpg/.jpeg para prevenir path traversal.
    """
    safe_name = Path(filename).name
    if not safe_name.lower().endswith((".jpg", ".jpeg")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Solo se sirven miniaturas JPEG",
        )

    frame_path = Path(settings.frames_dir) / str(session_id) / safe_name

    if not frame_path.exists() or not frame_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Miniatura no encontrada: {safe_name}",
        )

    return FileResponse(
        path=str(frame_path), media_type="image/jpeg", filename=safe_name
    )
