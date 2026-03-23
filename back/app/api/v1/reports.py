from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db

router = APIRouter()


@router.get("/reports/{group_id}")
async def get_reports(group_id: str, db: AsyncSession = Depends(get_db)):
    return {"group_id": group_id, "reports": []}


@router.get("/{report_id}/download")
async def download_report(report_id: str, format: str = "pdf"):
    return {"download_url": f"/api/v1/reports/{report_id}/download/{format}"}