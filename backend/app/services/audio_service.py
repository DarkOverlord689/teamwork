"""audio_service.py - Capa de servicio para el procesamiento de audio

Proporciona una API de alto nivel para iniciar, monitorear y recuperar
los resultados del análisis de audio. Todas las operaciones de base de
datos son asíncronas.
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
    """Gestiona el ciclo de vida del análisis de audio vía Celery + BD."""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def start_analysis(
        self,
        session_id: UUID,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Despacha una tarea de procesamiento de audio a Celery."""
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

        task = process_audio_task.delay(
            video_path=session.video_path,
            session_id=str(session_id),
            config_overrides=config,
        )

        logger.info(
            "Análisis de audio despachado: sesión=%s, tarea=%s", session_id, task.id
        )
        return {"session_id": str(session_id), "task_id": task.id, "status": "queued"}

    async def get_analysis_status(self, session_id: UUID) -> Dict[str, Any]:
        """Retorna el estado actual del análisis de audio."""
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
        """Recupera los resultados completos del análisis de audio.

        Los resultados se guardan como JSON en AnalysisSession.result_data.
        """
        stmt = select(AnalysisSession).where(AnalysisSession.id == session_id)
        result = await self.db.execute(stmt)
        session = result.scalar_one_or_none()

        if session is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Sesión {session_id} no encontrada",
            )

        result_data = getattr(session, "result_data", None)
        audio_data = result_data.get("audio") if result_data else None
        if not audio_data and not (result_data and "segments" in result_data):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Resultados de audio no disponibles (estado: {session.status})",
            )

        return result_data.get("audio", result_data)

    async def get_transcripts(self, session_id: UUID) -> List[Dict[str, Any]]:
        """Obtiene solo los segmentos de transcripción de una sesión."""
        results = await self.get_analysis_results(session_id)
        return results.get("transcripts", [])
