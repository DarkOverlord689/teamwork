from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import Group, Student, AnalysisSession
from app.schemas.group import GroupCreate, GroupResponse
from app.schemas.analysis import AnalysisResponse


router = APIRouter()


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
    result = await db.execute("SELECT * FROM groups ORDER BY created_at DESC")
    return result.scalars().all()


@router.get("/groups/{group_id}/analysis", response_model=list[AnalysisResponse])
async def get_group_analysis(group_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        "SELECT * FROM analysis_sessions WHERE group_id = :group_id",
        {"group_id": group_id}
    )
    return result.scalars().all()