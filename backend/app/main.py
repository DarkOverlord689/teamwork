import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1 import auth, upload, analysis, reports, validate, vision, audio, fusion, teacher
from app.database import engine, Base
from app.utils.config import settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield


app = FastAPI(
    title="SMATC-UPAO API",
    description="Sistema Multimodal de Análisis de Trabajo Colaborativo",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS must be added BEFORE the exception handler so that error responses
# also carry Access-Control-Allow-Origin headers and are not blocked by
# the browser's CORS policy.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )

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
    return {"status": "healthy", "version": "1.0.0"}