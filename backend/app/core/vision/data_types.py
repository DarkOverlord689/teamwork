"""Clases de datos para el pipeline de procesamiento de visión.

Estas clases definen la estructura de salida en cada nivel del pipeline:
datos por rostro por frame (PersonFrame), agregación por frame (FrameResult),
y resultados completos del video (VisionResult).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.core.audio.data_types import MomentType


class GazeCategory(str, Enum):
    """Clasificación de dirección de mirada."""

    CAMERA = "camera"
    SCREEN = "screen"
    PEER = "peer"
    AWAY = "away"
    UNKNOWN = "unknown"


class GestureType(str, Enum):
    """Tipos de gestos reconocidos de cabeza/rostro."""

    NOD = "nod"
    SHAKE = "shake"
    FROWN = "frown"
    NEUTRAL = "neutral"


@dataclass
class GazeData:
    """Resultado de estimación de mirada para un rostro en un frame."""

    direction: Tuple[float, float, float]
    is_looking_at_camera: bool
    confidence: float
    category: str

    def to_dict(self) -> dict:
        return {
            "direction": list(self.direction),
            "is_looking_at_camera": self.is_looking_at_camera,
            "confidence": self.confidence,
            "category": self.category,
        }


@dataclass
class GestureData:
    """Resultado de detección de gesto para un rostro en un frame."""

    gesture_type: str
    confidence: float
    intensity: float

    def to_dict(self) -> dict:
        return {
            "gesture_type": self.gesture_type,
            "confidence": self.confidence,
            "intensity": self.intensity,
        }


@dataclass
class PoseData:
    """Resultado de estimación de postura corporal para una persona en un frame."""

    body_orientation: float
    shoulder_angle: float
    confidence: float

    def to_dict(self) -> dict:
        return {
            "body_orientation": self.body_orientation,
            "shoulder_angle": self.shoulder_angle,
            "confidence": self.confidence,
        }


@dataclass
class EmotionData:
    """Resultado de clasificación de emoción para un rostro en un frame."""

    primary_emotion: str
    confidence: float
    all_emotions: Dict[str, float]

    def to_dict(self) -> dict:
        return {
            "primary_emotion": self.primary_emotion,
            "confidence": self.confidence,
            "all_emotions": dict(self.all_emotions),
        }


@dataclass
class PersonFrame:
    """Todos los datos de visión para una persona en un frame."""

    person_id: str
    bbox: Tuple[int, int, int, int]
    landmarks: Optional[np.ndarray] = None
    gaze: Optional[GazeData] = None
    gesture: Optional[GestureData] = None
    pose: Optional[PoseData] = None
    emotion: Optional[EmotionData] = None
    face_embedding: Optional[np.ndarray] = None

    def to_dict(self) -> dict:
        result: dict = {
            "person_id": self.person_id,
            "bbox": list(self.bbox),
        }
        if self.landmarks is not None:
            result["landmarks"] = self.landmarks.tolist()
        if self.gaze is not None:
            result["gaze"] = self.gaze.to_dict()
        if self.gesture is not None:
            result["gesture"] = self.gesture.to_dict()
        if self.pose is not None:
            result["pose"] = self.pose.to_dict()
        if self.emotion is not None:
            result["emotion"] = self.emotion.to_dict()
        if self.face_embedding is not None:
            result["face_embedding"] = self.face_embedding.tolist()
        return result


@dataclass
class FrameResult:
    """Todos los resultados para un frame muestreado."""

    frame_number: int
    timestamp: float
    persons: List[PersonFrame] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "frame_number": self.frame_number,
            "timestamp": self.timestamp,
            "persons": [p.to_dict() for p in self.persons],
        }


@dataclass
class PersonMetrics:
    """Métricas agregadas para una persona seguida durante toda la sesión."""

    person_id: str
    total_frames_seen: int
    gaze_contact_percentage: float
    dominant_emotion: str
    emotion_distribution: Dict[str, float]
    average_body_orientation: float
    gesture_counts: Dict[str, int]
    attention_score: float

    def to_dict(self) -> dict:
        return {
            "person_id": self.person_id,
            "total_frames_seen": self.total_frames_seen,
            "gaze_contact_percentage": self.gaze_contact_percentage,
            "dominant_emotion": self.dominant_emotion,
            "emotion_distribution": dict(self.emotion_distribution),
            "average_body_orientation": self.average_body_orientation,
            "gesture_counts": dict(self.gesture_counts),
            "attention_score": self.attention_score,
        }


@dataclass
class SessionMetrics:
    """Métricas agregadas para toda la sesión de análisis de video."""

    total_persons: int
    total_frames: int
    duration: float
    per_person_metrics: List[PersonMetrics] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_persons": self.total_persons,
            "total_frames": self.total_frames,
            "duration": self.duration,
            "per_person_metrics": [m.to_dict() for m in self.per_person_metrics],
        }


@dataclass
class MultimodalFrameAnalysis:
    """Resultado del análisis multimodal para un frame clave."""

    frame_number: int
    timestamp_ms: float
    speaker_id: str | None
    person_id: str | None
    moment_type: MomentType
    multimodal_description: str | None
    emotion_primary: str | None
    emotion_confidence: float | None
    attention_direction: str | None
    engagement_level: str | None
    face_detected: bool
    fallback_model: str | None
    fallback_reason: str | None
    image_url: str | None = None
    timestamp: float | None = None

    def to_dict(self) -> dict:
        return {
            "frame_number": self.frame_number,
            "timestamp_ms": self.timestamp_ms,
            "timestamp": self.timestamp
            if self.timestamp is not None
            else self.timestamp_ms / 1000.0,
            "speaker_id": self.speaker_id,
            "person_id": self.person_id,
            "moment_type": self.moment_type.value
            if self.moment_type is not None
            else None,
            "multimodal_description": self.multimodal_description,
            "emotion_primary": self.emotion_primary,
            "emotion_confidence": self.emotion_confidence,
            "attention_direction": self.attention_direction,
            "engagement_level": self.engagement_level,
            "face_detected": self.face_detected,
            "fallback_model": self.fallback_model,
            "fallback_reason": self.fallback_reason,
            "image_url": self.image_url,
        }


@dataclass
class SpeakerFaceMapping:
    """Mapeo entre un hablante diarizado y una cara/persona seguida."""

    speaker_id: str
    person_id: str | None
    confidence: float
    frame_count: int
    uncertain: bool

    def to_dict(self) -> dict:
        return {
            "speaker_id": self.speaker_id,
            "person_id": self.person_id,
            "confidence": self.confidence,
            "frame_count": self.frame_count,
            "uncertain": self.uncertain,
        }


@dataclass
class VisionResult:
    """Resultado completo del pipeline de visión para un video."""

    video_path: str
    total_frames: int
    fps_processed: float
    duration_seconds: float
    frames: List[FrameResult] = field(default_factory=list)
    processing_time_seconds: float = 0.0
    person_embeddings: Dict[str, Any] = field(default_factory=dict)
    session_metrics: Optional[SessionMetrics] = None
    smart_mode: bool = False
    key_moment_analyses: List["MultimodalFrameAnalysis"] = field(default_factory=list)
    speaker_face_mappings: List["SpeakerFaceMapping"] = field(default_factory=list)
    frame_thumbnails: Dict[float, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        result = {
            "video_path": self.video_path,
            "total_frames": self.total_frames,
            "fps_processed": self.fps_processed,
            "duration_seconds": self.duration_seconds,
            "processing_time_seconds": self.processing_time_seconds,
            "frames": [f.to_dict() for f in self.frames],
            "person_embeddings": {
                pid: emb.tolist() if isinstance(emb, np.ndarray) else emb
                for pid, emb in self.person_embeddings.items()
            },
            "smart_mode": self.smart_mode,
            "frame_thumbnails": {str(k): v for k, v in self.frame_thumbnails.items()},
        }
        if self.session_metrics is not None:
            result["session_metrics"] = self.session_metrics.to_dict()
        if self.key_moment_analyses:
            result["key_moment_analyses"] = [
                a.to_dict() for a in self.key_moment_analyses
            ]
        if self.speaker_face_mappings:
            result["speaker_face_mappings"] = [
                m.to_dict() for m in self.speaker_face_mappings
            ]
        return result
