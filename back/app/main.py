from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1 import auth, upload, analysis, reports, validate, vision, audio
from app.database import engine, Base
from app.utils.config import settings


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

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(upload.router, prefix="/api/v1/upload", tags=["upload"])
app.include_router(analysis.router, prefix="/api/v1/analysis", tags=["analysis"])
app.include_router(reports.router, prefix="/api/v1/reports", tags=["reports"])
app.include_router(validate.router, prefix="/api/v1/validate", tags=["validate"])
app.include_router(vision.router, prefix="/api/v1/vision", tags=["vision"])
app.include_router(audio.router, prefix="/api/v1/audio", tags=["audio"])


@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}