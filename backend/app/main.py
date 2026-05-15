"""main.py - Punto de entrada del servidor FastAPI SMATC-UPAO

Configura e inicia la aplicación con:
- Conexión a base de datos (creación automática de tablas)
- Middleware CORS para permitir peticiones del frontend
- Manejador global de excepciones no capturadas
- Todos los routers del API agrupados por módulo
- Endpoint de salud /health para monitoreo
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1 import (
    auth,
    upload,
    analysis,
    reports,
    validate,
    vision,
    audio,
    fusion,
    teacher,
)
from app.database import engine, Base
from app.utils.config import settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ciclo de vida de la aplicación: crea las tablas al iniciar."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield


app = FastAPI(
    title="SMATC-UPAO API",
    description="Sistema Multimodal de Análisis de Trabajo Colaborativo",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Captura errores no manejados y retorna un 500 genérico."""
    logger.exception(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# Registro de routers del API
app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(upload.router, prefix="/api/v1/upload", tags=["upload"])
app.include_router(analysis.router, prefix="/api/v1/analysis", tags=["analysis"])
app.include_router(reports.router, prefix="/api/v1/reports", tags=["reports"])
app.include_router(validate.router, prefix="/api/v1/validate", tags=["validate"])
app.include_router(vision.router, prefix="/api/v1/vision", tags=["vision"])
app.include_router(audio.router, prefix="/api/v1/audio", tags=["audio"])
app.include_router(fusion.router, prefix="/api/v1/fusion", tags=["fusion"])
app.include_router(teacher.router, prefix="/api/v1/teacher", tags=["teacher"])


@app.get("/health")
async def health_check():
    """Endpoint de verificación de salud del servidor."""
    return {"status": "healthy", "version": "1.0.0"}
