"""fusion_service.py - Capa de servicio para la fusión multimodal (Módulo 2.3)

Proporciona una API de alto nivel para iniciar, monitorear y recuperar
los resultados del análisis de fusión multimodal. Todas las operaciones
de base de datos son asíncronas.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional
from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import AnalysisSession
from app.tasks.fusion_tasks import process_fusion_task

logger = logging.getLogger(__name__)


class FusionService:
    """Gestiona el ciclo de vida del análisis de fusión multimodal vía Celery + BD."""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def start_analysis(
        self,
        session_id: UUID,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Despacha una tarea de fusión multimodal a Celery.

        Los análisis de visión y audio deben haberse completado primero
        y sus resultados deben estar almacenados en result_data.
        """
        stmt = select(AnalysisSession).where(AnalysisSession.id == session_id)
        result = await self.db.execute(stmt)
        session = result.scalar_one_or_none()

        if session is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Sesión {session_id} no encontrada",
            )

        if not session.video_path:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="La sesión no tiene video_path",
            )

        session.status = "queued"
        await self.db.commit()

        task = process_fusion_task.delay(
            session_id=str(session_id),
            config_overrides=config,
        )

        logger.info(
            "Fusión multimodal despachada: sesión=%s, tarea=%s", session_id, task.id
        )
        return {"session_id": str(session_id), "task_id": task.id, "status": "queued"}

    async def get_analysis_status(self, session_id: UUID) -> Dict[str, Any]:
        """Retorna el estado actual del análisis de fusión."""
        stmt = select(AnalysisSession).where(AnalysisSession.id == session_id)
        result = await self.db.execute(stmt)
        session = result.scalar_one_or_none()

        if session is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Sesión {session_id} no encontrada",
            )

        return {
            "session_id": str(session.id),
            "task_id": None,
            "status": session.status,
            "progress": None,
            "error": None,
        }

    async def get_analysis_results(self, session_id: UUID) -> Dict[str, Any]:
        """Recupera los resultados completos del análisis de fusión.

        Los resultados se guardan como JSON en AnalysisSession.result_data
        bajo la clave 'fusion'.
        """
        stmt = select(AnalysisSession).where(AnalysisSession.id == session_id)
        result = await self.db.execute(stmt)
        session = result.scalar_one_or_none()

        if session is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Sesión {session_id} no encontrada",
            )

        if session.status != "completed":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Análisis no completado (estado: {session.status})",
            )

        result_data = getattr(session, "result_data", None) or {}
        fusion_data = result_data.get("fusion", result_data)
        return fusion_data

    async def get_explanation(self, session_id: UUID) -> Dict[str, Any]:
        """Obtiene solo la explicación narrativa de una sesión de fusión completada."""
        results = await self.get_analysis_results(session_id)
        explanation = results.get("explanation")
        if explanation is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No se encontró explicación en los resultados de fusión",
            )
        return explanation

    async def get_rubric_scores(self, session_id: UUID) -> Dict[str, Any]:
        """Obtiene los puntajes de rúbrica UPAO de una sesión de fusión completada."""
        results = await self.get_analysis_results(session_id)
        rubric = results.get("rubric_scores")
        if rubric is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No se encontraron puntajes de rúbrica en los resultados de fusión",
            )
        return rubric
