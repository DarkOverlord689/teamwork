"""reports.py - Endpoints de reportes (Módulo 2.4)

Permite listar los reportes de un grupo y descargarlos en PDF o JSON.
La generación de PDF usa ReportLab a través de PDFReportGenerator.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import Response, StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.reports.pdf_generator import PDFReportGenerator
from app.database import get_db
from app.models import AnalysisSession, Group

router = APIRouter()


@router.get("/{group_id}", summary="Listar reportes de un grupo")
async def get_reports(
    group_id: str,
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """Retorna metadatos de todas las sesiones de análisis de un grupo."""
    try:
        group_uuid = UUID(group_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"group_id inválido: {group_id}",
        )

    stmt = (
        select(AnalysisSession)
        .where(AnalysisSession.group_id == group_uuid)
        .order_by(AnalysisSession.processed_at.desc().nullslast())
    )
    result = await db.execute(stmt)
    sessions: List[AnalysisSession] = list(result.scalars().all())

    group_stmt = select(Group).where(Group.id == group_uuid)
    group_result = await db.execute(group_stmt)
    group = group_result.scalar_one_or_none()
    group_name = group.name if group else group_id

    reports = [
        {
            "id": str(s.id),
            "group_id": str(s.group_id),
            "group_name": group_name,
            "status": s.status,
            "created_at": s.processed_at.isoformat() if s.processed_at else None,
            "duration_seconds": s.duration_seconds,
        }
        for s in sessions
    ]

    return {"group_id": group_id, "group_name": group_name, "reports": reports}


@router.get("/{report_id}/download", summary="Descargar reporte en PDF o JSON")
async def download_report(
    report_id: str,
    format: str = "pdf",
    db: AsyncSession = Depends(get_db),
) -> Response:
    """Descarga el reporte de una sesión de análisis.

    - format=json: retorna el JSON crudo del resultado de fusión.
    - format=pdf: genera un PDF con ReportLab y lo envía.
    """
    try:
        session_uuid = UUID(report_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"report_id inválido: {report_id}",
        )

    stmt = select(AnalysisSession).where(AnalysisSession.id == session_uuid)
    result = await db.execute(stmt)
    session: AnalysisSession | None = result.scalar_one_or_none()

    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Sesión {report_id} no encontrada",
        )

    if session.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Sesión {report_id} no completada (estado: {session.status})",
        )

    group_stmt = select(Group).where(Group.id == session.group_id)
    group_result = await db.execute(group_stmt)
    group = group_result.scalar_one_or_none()
    group_name = group.name if group else str(session.group_id)

    result_data: Dict[str, Any] = getattr(session, "result_data", None) or {}
    fusion_result: Dict[str, Any] = result_data.get("fusion", result_data)

    date_str = (
        session.processed_at.strftime("%Y-%m-%d")
        if session.processed_at
        else datetime.utcnow().strftime("%Y-%m-%d")
    )

    if format.lower() == "json":
        payload = json.dumps(fusion_result, ensure_ascii=False, indent=2).encode(
            "utf-8"
        )
        filename = f"report_{group_name}_{date_str}.json".replace(" ", "_")
        return Response(
            content=payload,
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    analysis_data: Dict[str, Any] = {
        "group_name": group_name,
        "course": "",
        "date": date_str,
        **fusion_result,
    }

    generator = PDFReportGenerator(analysis_data)
    pdf_bytes = generator.generate()

    filename = f"report_{group_name}_{date_str}.pdf".replace(" ", "_")

    return StreamingResponse(
        iter([pdf_bytes]),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
