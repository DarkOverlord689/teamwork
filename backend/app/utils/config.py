from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    database_url: str = "postgresql://user:pass@localhost:5432/smatc"
    redis_url: str = "redis://localhost:6379/0"

    s3_endpoint: str = "http://localhost:9000"
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"
    s3_bucket: str = "videos"

    secret_key: str = "your-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    cors_origins: list[str] = [
        "http://localhost",
        "http://localhost:80",
        "http://localhost:3000",
        "http://localhost:8000",
    ]

    llama_model_path: str = "/models/llama-3.1-8b"
    whisper_model: str = "large-v3"

    max_workers: int = 4
    video_fps: int = 5

    # Vision pipeline settings
    vision_frame_fps: float = 3.0
    vision_max_frames: int = 500
    vision_face_confidence: float = 0.5
    vision_gaze_threshold: float = 15.0
    vision_gesture_window: int = 10
    vision_clip_model: str = "openai/clip-vit-base-patch32"
    vision_device: str = "auto"
    vision_enable_emotion: bool = True
    vision_enable_gaze: bool = True
    vision_enable_gesture: bool = True
    vision_enable_pose: bool = True
    vision_enable_tracking: bool = True

    # Audio Processing (Module 2.2)
    audio_sample_rate: int = Field(default=16000)
    audio_device: str = Field(default="auto")
    pyannote_auth_token: str = Field(default="")
    whisper_model_size: str = Field(default="medium")
    whisper_language: str = Field(default="es")
    enable_audio_diarization: bool = Field(default=True)
    enable_audio_transcription: bool = Field(default=True)

    # Shared upload directory (must match the volume mounted in both backend and
    # celery-worker containers so that fallback-to-disk uploads are accessible
    # from both services without a FileNotFoundError).
    upload_tmp_dir: str = Field(default="/tmp/uploads")

    # Directory where key frame thumbnails are saved during vision processing.
    # Must be on the shared-uploads volume so both backend and celery-worker
    # can write/read the same files.
    frames_dir: str = Field(default="/shared/uploads/frames")

    # Multimodal Fusion (Module 2.3)
    openai_api_key: str = Field(default="")
    fusion_llm_model: str = Field(default="gpt-4o-mini")
    fusion_max_explanation_tokens: int = Field(default=1024)
    fusion_enable_explanation: bool = Field(default=True)

    # Smart frame selection
    sfs_enabled: bool = True
    use_multimodal: bool = True
    multimodal_model: str = "qwen3-vl:8b"
    multimodal_device: str = "auto"
    multimodal_timeout_seconds: float = 30.0

    # Ollama Cloud vision backend
    ollama_base_url: str = Field(default="https://api.ollama.com")
    ollama_api_key: str = Field(default="")
    ollama_vision_model: str = Field(default="qwen3-vl:8b")

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings():
    return Settings()


settings = get_settings()