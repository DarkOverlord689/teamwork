"""teacher.py - Endpoints del docente (Módulo 2.4)

Proporciona estadísticas agregadas para el dashboard del docente
y la comparativa del progreso de un grupo a través de múltiples sesiones.
"""

from __future__ import annotations

from typing import Any, Dict, List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import AnalysisSession, Group

router = APIRouter()


@router.get("/dashboard", summary="Estadísticas del dashboard del docente")
async def get_teacher_dashboard(
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """Retorna estadísticas agregadas para el dashboard principal del docente.

    Incluye: total de grupos, sesiones completadas, CV de participación
    promedio, puntaje de rúbrica promedio, desglose por estado y las
    10 sesiones completadas más recientes.
    """
    groups_stmt = select(func.count()).select_from(Group)
    total_groups: int = (await db.execute(groups_stmt)).scalar_one()

    status_stmt = select(AnalysisSession.status, func.count().label("cnt")).group_by(
        AnalysisSession.status
    )
    status_rows = (await db.execute(status_stmt)).all()
    groups_by_status: Dict[str, int] = {
        "completed": 0,
        "processing": 0,
        "pending": 0,
        "error": 0,
    }
    for row in status_rows:
        key = row.status if row.status in groups_by_status else "error"
        groups_by_status[key] = row.cnt

    completed_sessions: int = groups_by_status.get("completed", 0)

    avg_cv = 0.0
    avg_rubric_overall = 0.0
    try:
        cv_stmt = select(AnalysisSession).where(AnalysisSession.status == "completed")
        cv_result = await db.execute(cv_stmt)
        completed_rows: List[AnalysisSession] = list(cv_result.scalars().all())
        cv_values = []
        rubric_values = []
        for row in completed_rows:
            rd = getattr(row, "result_data", None) or {}
            fusion = rd.get("fusion", rd)
            gm = fusion.get("group_metrics", {})
            cv = gm.get("participation_cv")
            if cv is not None:
                cv_values.append(float(cv))
            # Get overall_score from rubric_scores in result_data
            rubric = fusion.get("rubric_scores", {})
            overall = rubric.get("overall_score")
            if overall is not None:
                rubric_values.append(float(overall))
        if cv_values:
            avg_cv = sum(cv_values) / len(cv_values)
        if rubric_values:
            avg_rubric_overall = sum(rubric_values) / len(rubric_values)
    except Exception:
        avg_cv = 0.0
        avg_rubric_overall = 0.0

    recent_stmt = (
        select(AnalysisSession, Group.name.label("group_name"))
        .join(Group, AnalysisSession.group_id == Group.id)
        .where(AnalysisSession.status == "completed")
        .order_by(AnalysisSession.processed_at.desc().nullslast())
        .limit(10)
    )
    recent_result = await db.execute(recent_stmt)
    recent_rows = recent_result.all()

    recent_sessions = []
    for row in recent_rows:
        session: AnalysisSession = row[0]
        g_name: str = row[1]

        # Get overall_score from result_data fusion rubric_scores
        rd = getattr(session, "result_data", None) or {}
        fusion = rd.get("fusion", rd)
        rubric = fusion.get("rubric_scores", {})
        overall = rubric.get("overall_score", 0.0)

        recent_sessions.append(
            {
                "session_id": str(session.id),
                "group_name": g_name,
                "status": session.status,
                "processed_at": session.processed_at.isoformat()
                if session.processed_at
                else None,
                "overall_score": round(float(overall), 2),
            }
        )

    return {
        "total_groups": total_groups,
        "completed_sessions": completed_sessions,
        "avg_participation_cv": round(avg_cv, 4),
        "avg_rubric_overall": round(float(avg_rubric_overall), 2),
        "groups_by_status": groups_by_status,
        "recent_sessions": recent_sessions,
    }


@router.get(
    "/groups/{group_id}/comparison",
    summary="Progreso del grupo a través de las sesiones",
)
async def get_group_comparison(
    group_id: str,
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """Retorna todas las sesiones de un grupo ordenadas por fecha con sus puntajes de rúbrica.

    Útil para graficar el progreso del grupo a lo largo del tiempo.
    """
    try:
        group_uuid = UUID(group_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"group_id inválido: {group_id}",
        )

    group_stmt = select(Group).where(Group.id == group_uuid)
    group_result = await db.execute(group_stmt)
    group = group_result.scalar_one_or_none()

    if group is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Grupo {group_id} no encontrado",
        )

    sessions_stmt = (
        select(AnalysisSession)
        .where(AnalysisSession.group_id == group_uuid)
        .order_by(AnalysisSession.processed_at.asc().nullslast())
    )
    sessions_result = await db.execute(sessions_stmt)
    sessions: List[AnalysisSession] = list(sessions_result.scalars().all())

    history = []
    for session in sessions:
        # Get rubric scores from result_data fusion
        rd = getattr(session, "result_data", None) or {}
        fusion = rd.get("fusion", rd)
        rubric = fusion.get("rubric_scores", {})

        def _safe(val: Any) -> float:
            return round(float(val), 2) if val is not None else 0.0

        history.append(
            {
                "session_id": str(session.id),
                "status": session.status,
                "processed_at": session.processed_at.isoformat()
                if session.processed_at
                else None,
                "duration_seconds": session.duration_seconds,
                "rubric_scores": {
                    "contributes_to_team_meetings": _safe(
                        rubric.get("contributes_to_team_meetings")
                    ),
                    "facilitates_contributions": _safe(
                        rubric.get("facilitates_contributions")
                    ),
                    "fosters_constructive_climate": _safe(
                        rubric.get("fosters_constructive_climate")
                    ),
                    "responds_to_conflict": _safe(rubric.get("responds_to_conflict")),
                    "individual_contributions_outside": _safe(
                        rubric.get("individual_contributions_outside")
                    ),
                    "overall": _safe(rubric.get("overall_score")),
                },
            }
        )

    return {
        "group_id": group_id,
        "group_name": group.name,
        "session_count": len(history),
        "history": history,
    }
