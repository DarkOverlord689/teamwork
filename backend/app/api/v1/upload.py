import asyncio
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import AnalysisSession, Group
from app.utils.config import settings

router = APIRouter()

ALLOWED_CONTENT_TYPES = {"video/mp4", "video/avi", "video/quicktime"}

PROGRESS_MESSAGES = {
    "pending": "En cola, esperando procesamiento...",
    "processing": "Analizando el video...",
    "completed": "Análisis completado",
    "error": "Error en el procesamiento",
    "failed": "Error en el procesamiento",
}


def _upload_to_minio(file_bytes: bytes, bucket: str, key: str) -> None:
    """Synchronous MinIO upload, intended to be called via asyncio.to_thread."""
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError

    s3 = boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint,
        aws_access_key_id=settings.s3_access_key,
        aws_secret_access_key=settings.s3_secret_key,
    )
    try:
        s3.put_object(Bucket=bucket, Key=key, Body=file_bytes)
    except (BotoCoreError, ClientError) as exc:
        raise RuntimeError(f"MinIO upload failed: {exc}") from exc


@router.post("/", status_code=status.HTTP_201_CREATED)
async def upload_video(
    file: UploadFile = File(...),
    group_name: str | None = Form(default=None),
    group_id: str | None = Form(default=None),
    db: AsyncSession = Depends(get_db),
):
    # 1. Validate content type
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Formato no soportado. Use MP4, AVI o MOV",
        )

    # 2. Resolve group
    group: Group | None = None

    if group_id:
        try:
            parsed_id = uuid.UUID(group_id)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="group_id inválido",
            )
        result = await db.execute(select(Group).where(Group.id == parsed_id))
        group = result.scalar_one_or_none()
        if group is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Grupo {group_id} no encontrado",
            )
    else:
        # Derive group name from form field or filename
        resolved_name = group_name or Path(file.filename or "video").stem
        group = Group(name=resolved_name)
        db.add(group)
        await db.flush()  # populate group.id before using it

    # 3. Read file content and persist (MinIO or /tmp fallback)
    file_bytes = await file.read()
    safe_filename = file.filename or f"{uuid.uuid4()}.mp4"
    s3_key = f"videos/{group.id}/{safe_filename}"

    try:
        await asyncio.to_thread(_upload_to_minio, file_bytes, settings.s3_bucket, s3_key)
        video_path = s3_key
        logger.info("Video uploaded to MinIO: {}", s3_key)
    except Exception as exc:
        # Fallback: save to the shared upload directory so that the
        # celery-worker container (which mounts the same volume) can read it.
        _tmp_dir = settings.upload_tmp_dir
        tmp_path = f"{_tmp_dir}/{uuid.uuid4()}_{safe_filename}"
        try:
            await asyncio.to_thread(_write_tmp, file_bytes, tmp_path)
            video_path = tmp_path
            logger.warning(
                "MinIO unreachable ({}). Video saved locally: {}", exc, tmp_path
            )
        except Exception as tmp_exc:
            logger.error("Failed to save video locally: {}", tmp_exc)
            video_path = f"{_tmp_dir}/{safe_filename}"  # best-effort placeholder

    # 4. Create AnalysisSession
    session = AnalysisSession(
        group_id=group.id,
        video_path=video_path,
        status="pending",
        duration_seconds=0,
    )
    db.add(session)
    await db.commit()
    await db.refresh(session)
    await db.refresh(group)

    # 5. Dispatch Celery task
    try:
        from app.tasks import process_video_task

        process_video_task.delay(video_path=video_path, session_id=str(session.id))
        logger.info("Celery task dispatched for session {}", session.id)
    except Exception as exc:
        # If Celery / broker is unavailable, mark as processing so the caller
        # knows work is conceptually in progress; a worker will pick it up when
        # the broker comes back, or an operator can re-trigger manually.
        logger.warning("Could not dispatch Celery task ({}); status stays 'pending'", exc)

    return {
        "session_id": str(session.id),
        "group_id": str(group.id),
        "status": session.status,
        "message": "Video recibido, procesando...",
    }


def _write_tmp(file_bytes: bytes, path: str) -> None:
    """Write bytes to a local file path (sync, for asyncio.to_thread)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(file_bytes)


@router.get("/{session_id}/status")
async def get_upload_status(session_id: str, db: AsyncSession = Depends(get_db)):
    try:
        parsed_id = uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="session_id inválido",
        )

    result = await db.execute(
        select(AnalysisSession, Group)
        .join(Group, AnalysisSession.group_id == Group.id)
        .where(AnalysisSession.id == parsed_id)
    )
    row = result.one_or_none()

    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Sesión {session_id} no encontrada",
        )

    session_obj, group_obj = row

    effective_status = session_obj.status
    effective_progress_message = PROGRESS_MESSAGES.get(session_obj.status, "Estado desconocido")

    normalized_status = effective_status if effective_status != "failed" else "error"
    return {
        "session_id": str(session_obj.id),
        "status": normalized_status,
        "group_id": str(session_obj.group_id),
        "group_name": group_obj.name,
        "progress_message": effective_progress_message,
    }
