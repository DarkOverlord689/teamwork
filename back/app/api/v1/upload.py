from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db

router = APIRouter()


@router.post("/", status_code=status.HTTP_201_CREATED)
async def upload_video(
    file: UploadFile = File(...),
    group_id: str = None,
    db: AsyncSession = Depends(get_db)
):
    if file.content_type not in ["video/mp4", "video/avi", "video/quicktime"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Formato no soportado. Use MP4, AVI o MOV"
        )
    
    return {"message": "Video recibido", "filename": file.filename, "group_id": group_id}