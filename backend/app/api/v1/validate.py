"""validate.py - Endpoints de validación docente (Módulo 2.4)

Permite al docente revisar los puntajes de rúbrica generados por el sistema,
enviar correcciones y agregar notas por estudiante.

Las correcciones del docente se guardan en una fila separada
(evaluator_type='teacher') sin modificar las puntuaciones originales del sistema,
permitiendo la comparación entre ambas.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import AnalysisSession, RubricScore

router = APIRouter()


class RubricCorrections(BaseModel):
    """Correcciones del docente para cada criterio de rúbrica VALUE (0-20)."""

    contributes_to_team_meetings: float | None = Field(None, ge=0, le=20)
    facilitates_contributions: float | None = Field(None, ge=0, le=20)
    fosters_constructive_climate: float | None = Field(None, ge=0, le=20)
    responds_to_conflict: float | None = Field(None, ge=0, le=20)
    individual_contributions_outside: float | None = Field(None, ge=0, le=20)


class ValidationRequest(BaseModel):
    """Solicitud de validación: ID del estudiante, correcciones y nota del docente."""

    student_id: str
    rubric_corrections: RubricCorrections
    teacher_note: str = ""


@router.post("/{session_id}", summary="Enviar correcciones del docente para una sesión")
async def submit_corrections(
    session_id: str,
    payload: ValidationRequest,
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """Guarda las correcciones del docente como una nueva fila RubricScore.

    Las puntuaciones originales del sistema NO se modifican.
    Las correcciones del docente se almacenan por separado.
    """
    try:
        session_uuid = UUID(session_id)
        student_uuid = UUID(payload.student_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        )

    stmt = select(AnalysisSession).where(AnalysisSession.id == session_uuid)
    result = await db.execute(stmt)
    session = result.scalar_one_or_none()

    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Sesión {session_id} no encontrada",
        )

    sys_stmt = select(RubricScore).where(
        RubricScore.session_id == session_uuid,
        RubricScore.student_id == student_uuid,
        RubricScore.evaluator_type == "system",
    )
    sys_result = await db.execute(sys_stmt)
    system_score: RubricScore | None = sys_result.scalar_one_or_none()

    corr = payload.rubric_corrections

    def _resolve(teacher_val: float | None, system_attr: str) -> float | None:
        if teacher_val is not None:
            return teacher_val
        if system_score is not None:
            return getattr(system_score, f"{system_attr}_score", None)
        return None

    contributes = _resolve(
        corr.contributes_to_team_meetings, "contributes_to_team_meetings"
    )
    facilitates = _resolve(corr.facilitates_contributions, "facilitates_contributions")
    climate = _resolve(
        corr.fosters_constructive_climate, "fosters_constructive_climate"
    )
    conflict = _resolve(corr.responds_to_conflict, "responds_to_conflict")
    outside = _resolve(
        corr.individual_contributions_outside, "individual_contributions_outside"
    )

    scores = [
        s
        for s in [
            contributes,
            facilitates,
            climate,
            conflict,
            outside,
        ]
        if s is not None
    ]
    overall = sum(scores) / len(scores) if scores else None

    teacher_row = RubricScore(
        id=uuid.uuid4(),
        session_id=session_uuid,
        student_id=student_uuid,
        collaboration_score=contributes,
        communication_score=facilitates,
        responsibility_score=climate,
        leadership_score=conflict,
        technical_contribution_score=outside,
        overall_score=overall,
        evaluator_type="teacher",
    )
    db.add(teacher_row)
    await db.commit()
    await db.refresh(teacher_row)

    return {
        "session_id": session_id,
        "student_id": payload.student_id,
        "teacher_note": payload.teacher_note,
        "status": "validated",
        "teacher_scores": {
            "contributes_to_team_meetings": teacher_row.collaboration_score,
            "facilitates_contributions": teacher_row.communication_score,
            "fosters_constructive_climate": teacher_row.responsibility_score,
            "responds_to_conflict": teacher_row.leadership_score,
            "individual_contributions_outside": teacher_row.technical_contribution_score,
            "overall": teacher_row.overall_score,
        },
    }


@router.get(
    "/{session_id}", summary="Obtener puntajes del sistema y del docente lado a lado"
)
async def get_validation(
    session_id: str,
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """Retorna los puntajes de rúbrica del sistema y del docente para comparación."""
    try:
        session_uuid = UUID(session_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        )

    stmt = select(AnalysisSession).where(AnalysisSession.id == session_uuid)
    result = await db.execute(stmt)
    session = result.scalar_one_or_none()

    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Sesión {session_id} no encontrada",
        )

    scores_stmt = select(RubricScore).where(RubricScore.session_id == session_uuid)
    scores_result = await db.execute(scores_stmt)
    all_scores: List[RubricScore] = list(scores_result.scalars().all())

    def _serialize(row: RubricScore) -> Dict[str, Any]:
        return {
            "id": str(row.id),
            "student_id": str(row.student_id),
            "evaluator_type": row.evaluator_type,
            "contributes_to_team_meetings": row.collaboration_score,
            "facilitates_contributions": row.communication_score,
            "fosters_constructive_climate": row.responsibility_score,
            "responds_to_conflict": row.leadership_score,
            "individual_contributions_outside": row.technical_contribution_score,
            "overall": row.overall_score,
        }

    system_scores = [_serialize(s) for s in all_scores if s.evaluator_type == "system"]
    teacher_scores = [
        _serialize(s) for s in all_scores if s.evaluator_type == "teacher"
    ]

    return {
        "session_id": session_id,
        "system_scores": system_scores,
        "teacher_scores": teacher_scores,
    }
