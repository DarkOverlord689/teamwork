"""fusion.py - Endpoints de fusión multimodal (Módulo 2.3)

Proporciona rutas REST para iniciar el análisis de fusión completo
en una sesión que ya fue procesada por los pipelines de visión y audio,
consultar el estado y obtener resultados, puntajes de rúbrica y
explicaciones narrativas.
"""

from __future__ import annotations

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.schemas.fusion import (
    ExplanationResultSchema,
    FusionProcessingResponse,
    FusionResultSchema,
    FusionStatusResponse,
    GroupRubricScoresSchema,
    StartFusionRequest,
)
from app.services.fusion_service import FusionService

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/sessions/{session_id}/analyze",
    response_model=FusionProcessingResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Iniciar análisis de fusión multimodal",
)
async def start_fusion_analysis(
    session_id: UUID,
    request: StartFusionRequest,
    db: AsyncSession = Depends(get_db),
) -> FusionProcessingResponse:
    """Encola una tarea de fusión multimodal para la sesión indicada.

    Los pipelines de visión y audio deben haber completado su ejecución.
    La fusión alinea las características temporales, calcula métricas
    de colaboración, mapea a la rúbrica UPAO y genera una explicación.
    """
    service = FusionService(db)
    result = await service.start_analysis(
        session_id=session_id,
        config=request.config_overrides,
    )
    return FusionProcessingResponse(**result)


@router.get(
    "/sessions/{session_id}/status",
    response_model=FusionStatusResponse,
    summary="Estado del análisis de fusión",
)
async def get_fusion_status(
    session_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> FusionStatusResponse:
    """Retorna el estado actual de una tarea de fusión."""
    service = FusionService(db)
    status_info = await service.get_analysis_status(session_id)
    return FusionStatusResponse(**status_info)


@router.get(
    "/sessions/{session_id}/result",
    response_model=FusionResultSchema,
    summary="Obtener resultado completo de la fusión",
)
async def get_fusion_result(
    session_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> FusionResultSchema:
    """Retorna el resultado completo del análisis de fusión."""
    service = FusionService(db)
    result_data = await service.get_analysis_results(session_id)
    return FusionResultSchema(**result_data)


@router.get(
    "/sessions/{session_id}/explanation",
    response_model=ExplanationResultSchema,
    summary="Obtener explicación narrativa",
)
async def get_fusion_explanation(
    session_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> ExplanationResultSchema:
    """Retorna la explicación narrativa de una sesión de fusión completada.

    Incluye fortalezas, áreas de mejora y recomendaciones específicas
    generadas por el LLM.
    """
    service = FusionService(db)
    explanation = await service.get_explanation(session_id)
    return ExplanationResultSchema(**explanation)


@router.get(
    "/sessions/{session_id}/rubric",
    response_model=GroupRubricScoresSchema,
    summary="Obtener puntajes de rúbrica UPAO",
)
async def get_fusion_rubric(
    session_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> GroupRubricScoresSchema:
    """Retorna los puntajes de rúbrica UPAO para una sesión completada.

    Los puntajes van de 0 a 20 en cinco dimensiones: colaboración,
    comunicación, responsabilidad, liderazgo y contribución técnica.
    """
    service = FusionService(db)
    rubric = await service.get_rubric_scores(session_id)
    return GroupRubricScoresSchema(**rubric)
