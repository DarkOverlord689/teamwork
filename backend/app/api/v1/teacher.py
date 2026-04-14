"""Teacher-specific dashboard and comparison endpoints (Module 2.4).

Provides aggregate statistics for the teacher dashboard and group progress
comparison across sessions over time.
"""

from __future__ import annotations

from typing import Any, Dict, List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import AnalysisSession, Group, RubricScore

router = APIRouter()


# ---------------------------------------------------------------------------
# GET /teacher/dashboard
# ---------------------------------------------------------------------------


@router.get("/dashboard", summary="Teacher aggregate dashboard statistics")
async def get_teacher_dashboard(
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """Return aggregate stats used to populate the teacher's main dashboard.

    Includes total groups, completed sessions, average participation CV,
    average rubric overall score, groups_by_status breakdown, and the 10
    most recent completed sessions.
    """
    # Total groups
    groups_stmt = select(func.count()).select_from(Group)
    total_groups: int = (await db.execute(groups_stmt)).scalar_one()

    # Sessions aggregated by status
    status_stmt = select(AnalysisSession.status, func.count().label("cnt")).group_by(AnalysisSession.status)
    status_rows = (await db.execute(status_stmt)).all()
    groups_by_status: Dict[str, int] = {"completed": 0, "processing": 0, "pending": 0, "error": 0}
    for row in status_rows:
        key = row.status if row.status in groups_by_status else "error"
        groups_by_status[key] = row.cnt

    completed_sessions: int = groups_by_status.get("completed", 0)

    # Average participation CV — stored in result_data->fusion->group_metrics->participation_cv
    # Fall back to 0.0 if the column does not exist or data is missing.
    avg_cv = 0.0
    try:
        cv_stmt = select(AnalysisSession).where(AnalysisSession.status == "completed")
        cv_result = await db.execute(cv_stmt)
        completed_rows: List[AnalysisSession] = list(cv_result.scalars().all())
        cv_values = []
        for row in completed_rows:
            rd = getattr(row, "result_data", None) or {}
            fusion = rd.get("fusion", rd)
            gm = fusion.get("group_metrics", {})
            cv = gm.get("participation_cv")
            if cv is not None:
                cv_values.append(float(cv))
        if cv_values:
            avg_cv = sum(cv_values) / len(cv_values)
    except Exception:
        avg_cv = 0.0

    # Average rubric overall score (system evaluator)
    rubric_stmt = (
        select(func.avg(RubricScore.overall_score))
        .where(
            RubricScore.evaluator_type == "system",
            RubricScore.overall_score.isnot(None),
        )
    )
    avg_rubric_overall: float = (await db.execute(rubric_stmt)).scalar_one() or 0.0

    # Recent sessions (last 10 completed)
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

        # Extract overall rubric score from system rubric rows
        rubric_row_stmt = (
            select(func.avg(RubricScore.overall_score))
            .where(
                RubricScore.session_id == session.id,
                RubricScore.evaluator_type == "system",
                RubricScore.overall_score.isnot(None),
            )
        )
        overall = (await db.execute(rubric_row_stmt)).scalar_one() or 0.0

        recent_sessions.append(
            {
                "session_id": str(session.id),
                "group_name": g_name,
                "status": session.status,
                "processed_at": session.processed_at.isoformat() if session.processed_at else None,
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


# ---------------------------------------------------------------------------
# GET /teacher/groups/{group_id}/comparison
# ---------------------------------------------------------------------------


@router.get("/groups/{group_id}/comparison", summary="Group progress across sessions")
async def get_group_comparison(
    group_id: str,
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """Return all sessions for *group_id* ordered by date, with rubric scores per session.

    Use this to plot progress over time for a group.
    """
    try:
        group_uuid = UUID(group_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid group_id: {group_id}",
        )

    group_stmt = select(Group).where(Group.id == group_uuid)
    group_result = await db.execute(group_stmt)
    group = group_result.scalar_one_or_none()

    if group is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Group {group_id} not found",
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
        rubric_stmt = (
            select(
                func.avg(RubricScore.collaboration_score).label("collaboration"),
                func.avg(RubricScore.communication_score).label("communication"),
                func.avg(RubricScore.responsibility_score).label("responsibility"),
                func.avg(RubricScore.leadership_score).label("leadership"),
                func.avg(RubricScore.technical_contribution_score).label("technical_contribution"),
                func.avg(RubricScore.overall_score).label("overall"),
            )
            .where(
                RubricScore.session_id == session.id,
                RubricScore.evaluator_type == "system",
            )
        )
        rubric_result = await db.execute(rubric_stmt)
        rubric_row = rubric_result.one()

        def _safe(val: Any) -> float:
            return round(float(val), 2) if val is not None else 0.0

        history.append(
            {
                "session_id": str(session.id),
                "status": session.status,
                "processed_at": session.processed_at.isoformat() if session.processed_at else None,
                "duration_seconds": session.duration_seconds,
                "rubric_scores": {
                    "collaboration": _safe(rubric_row.collaboration),
                    "communication": _safe(rubric_row.communication),
                    "responsibility": _safe(rubric_row.responsibility),
                    "leadership": _safe(rubric_row.leadership),
                    "technical_contribution": _safe(rubric_row.technical_contribution),
                    "overall": _safe(rubric_row.overall),
                },
            }
        )

    return {
        "group_id": group_id,
        "group_name": group.name,
        "session_count": len(history),
        "history": history,
    }
