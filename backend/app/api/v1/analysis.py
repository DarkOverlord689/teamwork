import uuid as _uuid
from typing import Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import Group, Student, AnalysisSession, RubricScore
from app.schemas.group import GroupCreate, GroupResponse
from app.schemas.analysis import AnalysisResponse


router = APIRouter()


# ---------------------------------------------------------------------------
# Request schemas for topic/intervention updates
# ---------------------------------------------------------------------------


class TopicDescriptionRequest(BaseModel):
    topic_description: str | None = None


class InterventionSummaryEntry(BaseModel):
    student_id: str
    intervention_summary: str | None = None


class InterventionSummariesRequest(BaseModel):
    interventions: List[InterventionSummaryEntry]


@router.post("/groups", response_model=GroupResponse, status_code=status.HTTP_201_CREATED)
async def create_group(group: GroupCreate, db: AsyncSession = Depends(get_db)):
    db_group = Group(
        course_id=group.course_id,
        name=group.name,
    )
    db.add(db_group)
    await db.commit()
    await db.refresh(db_group)
    return db_group


@router.get("/groups", response_model=list[GroupResponse])
async def list_groups(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Group).order_by(Group.created_at.desc()))
    return result.scalars().all()


@router.get("/groups/{group_id}/analysis", response_model=list[AnalysisResponse])
async def get_group_analysis(group_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(AnalysisSession).where(AnalysisSession.group_id == group_id)
    )
    sessions = result.scalars().all()

    responses = []
    for session in sessions:
        fusion_data = {}
        result_data = getattr(session, "result_data", None) or {}
        if isinstance(result_data.get("fusion"), dict):
            fusion_data = result_data["fusion"]

        # Merge per-student intervention_summary from RubricScore (system evaluator)
        # into group_metrics.per_student_metrics so the frontend receives them together.
        group_metrics = fusion_data.get("group_metrics")
        if group_metrics and isinstance(group_metrics.get("per_student_metrics"), list):
            rubric_result = await db.execute(
                select(RubricScore).where(
                    RubricScore.session_id == session.id,
                    RubricScore.evaluator_type == "system",
                )
            )
            rubric_rows: Dict[str, str] = {
                str(row.student_id): row.intervention_summary
                for row in rubric_result.scalars().all()
                if row.intervention_summary
            }
            if rubric_rows:
                for sm in group_metrics["per_student_metrics"]:
                    sid = sm.get("student_id", "")
                    if sid in rubric_rows:
                        sm["intervention_summary"] = rubric_rows[sid]

        responses.append(
            AnalysisResponse(
                id=session.id,
                group_id=session.group_id,
                video_path=session.video_path,
                duration_seconds=session.duration_seconds,
                processed_at=session.processed_at,
                status=session.status,
                topic_description=session.topic_description,
                group_metrics=group_metrics,
                rubric_scores=fusion_data.get("rubric_scores"),
                explanation=fusion_data.get("explanation"),
            )
        )
    return responses


# ---------------------------------------------------------------------------
# PATCH /analysis/sessions/{session_id}/topic
# ---------------------------------------------------------------------------


@router.patch("/sessions/{session_id}/topic", response_model=AnalysisResponse)
async def update_topic_description(
    session_id: str,
    payload: TopicDescriptionRequest,
    db: AsyncSession = Depends(get_db),
) -> AnalysisResponse:
    """Set or update the topic description for a session (exposition theme)."""
    try:
        session_uuid = UUID(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))

    result = await db.execute(select(AnalysisSession).where(AnalysisSession.id == session_uuid))
    session = result.scalar_one_or_none()
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis session {session_id} not found",
        )

    session.topic_description = payload.topic_description
    await db.commit()
    await db.refresh(session)

    fusion_data = {}
    result_data = getattr(session, "result_data", None) or {}
    if isinstance(result_data.get("fusion"), dict):
        fusion_data = result_data["fusion"]

    return AnalysisResponse(
        id=session.id,
        group_id=session.group_id,
        video_path=session.video_path,
        duration_seconds=session.duration_seconds,
        processed_at=session.processed_at,
        status=session.status,
        topic_description=session.topic_description,
        group_metrics=fusion_data.get("group_metrics"),
        rubric_scores=fusion_data.get("rubric_scores"),
        explanation=fusion_data.get("explanation"),
    )


# ---------------------------------------------------------------------------
# PATCH /analysis/sessions/{session_id}/interventions
# ---------------------------------------------------------------------------


@router.patch("/sessions/{session_id}/interventions")
async def update_intervention_summaries(
    session_id: str,
    payload: InterventionSummariesRequest,
    db: AsyncSession = Depends(get_db),
) -> Dict:
    """Set or update per-student intervention summaries for a session.

    Each entry targets the system RubricScore row for the given student.
    If no system RubricScore row exists for a student, a new one is created.
    """
    try:
        session_uuid = UUID(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))

    result = await db.execute(select(AnalysisSession).where(AnalysisSession.id == session_uuid))
    session = result.scalar_one_or_none()
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis session {session_id} not found",
        )

    updated = []
    for entry in payload.interventions:
        try:
            student_uuid = UUID(entry.student_id)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid student_id: {entry.student_id}",
            )

        sys_stmt = select(RubricScore).where(
            RubricScore.session_id == session_uuid,
            RubricScore.student_id == student_uuid,
            RubricScore.evaluator_type == "system",
        )
        sys_result = await db.execute(sys_stmt)
        rubric_row = sys_result.scalar_one_or_none()

        if rubric_row is None:
            rubric_row = RubricScore(
                id=_uuid.uuid4(),
                session_id=session_uuid,
                student_id=student_uuid,
                evaluator_type="system",
            )
            db.add(rubric_row)

        rubric_row.intervention_summary = entry.intervention_summary
        updated.append(entry.student_id)

    await db.commit()
    return {"session_id": session_id, "updated_students": updated, "status": "ok"}