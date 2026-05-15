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
    """Correcciones del docente para cada criterio de rúbrica (0-20)."""

    collaboration: float | None = Field(None, ge=0, le=20)
    communication: float | None = Field(None, ge=0, le=20)
    responsibility: float | None = Field(None, ge=0, le=20)
    leadership: float | None = Field(None, ge=0, le=20)
    technical_contribution: float | None = Field(None, ge=0, le=20)


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

    collaboration = _resolve(corr.collaboration, "collaboration")
    communication = _resolve(corr.communication, "communication")
    responsibility = _resolve(corr.responsibility, "responsibility")
    leadership = _resolve(corr.leadership, "leadership")
    technical_contribution = _resolve(
        corr.technical_contribution, "technical_contribution"
    )

    scores = [
        s
        for s in [
            collaboration,
            communication,
            responsibility,
            leadership,
            technical_contribution,
        ]
        if s is not None
    ]
    overall = sum(scores) / len(scores) if scores else None

    teacher_row = RubricScore(
        id=uuid.uuid4(),
        session_id=session_uuid,
        student_id=student_uuid,
        collaboration_score=collaboration,
        communication_score=communication,
        responsibility_score=responsibility,
        leadership_score=leadership,
        technical_contribution_score=technical_contribution,
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
            "collaboration": teacher_row.collaboration_score,
            "communication": teacher_row.communication_score,
            "responsibility": teacher_row.responsibility_score,
            "leadership": teacher_row.leadership_score,
            "technical_contribution": teacher_row.technical_contribution_score,
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
            "collaboration": row.collaboration_score,
            "communication": row.communication_score,
            "responsibility": row.responsibility_score,
            "leadership": row.leadership_score,
            "technical_contribution": row.technical_contribution_score,
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
