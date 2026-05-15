"""vision_service.py - Capa de servicio para el procesamiento de video

Proporciona una API de alto nivel para iniciar, monitorear y recuperar
los resultados del análisis de video. Todas las operaciones de base de
datos son asíncronas.
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
    """Gestiona el ciclo de vida del análisis de video vía Celery + BD."""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def start_analysis(
        self,
        video_path: str,
        session_id: UUID,
        config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Despacha una tarea de procesamiento de video a Celery."""
        stmt = select(AnalysisSession).where(AnalysisSession.id == session_id)
        result = await self.db.execute(stmt)
        session = result.scalar_one_or_none()

        if session is None:
            raise ValueError(f"Sesión {session_id} no encontrada")

        session.status = "queued"
        await self.db.commit()

        task = process_video_task.delay(
            video_path=video_path,
            session_id=str(session_id),
            config_overrides=config,
        )

        logger.info(
            "Análisis de video despachado: sesión=%s, tarea=%s", session_id, task.id
        )
        return task.id

    async def get_analysis_status(self, session_id: UUID) -> Dict[str, Any]:
        """Retorna el estado actual del análisis de video."""
        stmt = select(AnalysisSession).where(AnalysisSession.id == session_id)
        result = await self.db.execute(stmt)
        session = result.scalar_one_or_none()

        if session is None:
            raise ValueError(f"Sesión {session_id} no encontrada")

        return {
            "session_id": str(session.id),
            "status": session.status,
            "duration_seconds": session.duration_seconds,
            "processed_at": session.processed_at.isoformat()
            if session.processed_at
            else None,
        }

    async def get_analysis_results(self, session_id: UUID) -> Dict[str, Any]:
        """Recupera los resultados completos del análisis de video.

        Los resultados se guardan como JSON en AnalysisSession.session_metrics.
        """
        stmt = select(AnalysisSession).where(AnalysisSession.id == session_id)
        result = await self.db.execute(stmt)
        session = result.scalar_one_or_none()

        if session is None:
            raise ValueError(f"Sesión {session_id} no encontrada")

        status_info: Dict[str, Any] = {
            "session_id": str(session.id),
            "status": session.status,
            "duration_seconds": session.duration_seconds,
            "processed_at": session.processed_at.isoformat()
            if session.processed_at
            else None,
        }

        session_metrics = getattr(session, "session_metrics", None)
        if session_metrics:
            return {**session_metrics, **status_info}

        return status_info
