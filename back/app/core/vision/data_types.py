"""Data classes for the vision processing pipeline.

These dataclasses define the structured output at every level of the pipeline:
per-face per-frame data (PersonFrame), per-frame aggregation (FrameResult),
and full-video results (VisionResult).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class GazeCategory(str, Enum):
    """Discrete gaze direction classification."""
    CAMERA = "camera"
    SCREEN = "screen"
    PEER = "peer"
    AWAY = "away"
    UNKNOWN = "unknown"


class GestureType(str, Enum):
    """Recognised head/facial gesture types."""
    NOD = "nod"
    SHAKE = "shake"
    FROWN = "frown"
    NEUTRAL = "neutral"


# ---------------------------------------------------------------------------
# Per-attribute data
# ---------------------------------------------------------------------------

@dataclass
class GazeData:
    """Gaze estimation result for a single face in a single frame."""
    direction: Tuple[float, float, float]  # yaw, pitch, roll (degrees)
    is_looking_at_camera: bool
    confidence: float
    category: str  # one of GazeCategory values

    def to_dict(self) -> dict:
        return {
            "direction": list(self.direction),
            "is_looking_at_camera": self.is_looking_at_camera,
            "confidence": self.confidence,
            "category": self.category,
        }


@dataclass
class GestureData:
    """Gesture detection result for a single face in a single frame."""
    gesture_type: str  # one of GestureType values
    confidence: float
    intensity: float  # 0.0 to 1.0

    def to_dict(self) -> dict:
        return {
            "gesture_type": self.gesture_type,
            "confidence": self.confidence,
            "intensity": self.intensity,
        }


@dataclass
class PoseData:
    """Body pose estimation result for a single person in a single frame."""
    body_orientation: float  # degrees, 0 = facing camera
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
    """Emotion classification result for a single face in a single frame."""
    primary_emotion: str
    confidence: float
    all_emotions: Dict[str, float]  # emotion label -> probability

    def to_dict(self) -> dict:
        return {
            "primary_emotion": self.primary_emotion,
            "confidence": self.confidence,
            "all_emotions": dict(self.all_emotions),
        }


# ---------------------------------------------------------------------------
# Per-person, per-frame composite
# ---------------------------------------------------------------------------

@dataclass
class PersonFrame:
    """All vision data for one person in one frame."""
    person_id: str
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    landmarks: Optional[np.ndarray] = None  # 468x3 face landmarks
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


# ---------------------------------------------------------------------------
# Per-frame aggregate
# ---------------------------------------------------------------------------

@dataclass
class FrameResult:
    """All results for one sampled frame."""
    frame_number: int
    timestamp: float  # seconds from start of video
    persons: List[PersonFrame] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "frame_number": self.frame_number,
            "timestamp": self.timestamp,
            "persons": [p.to_dict() for p in self.persons],
        }


# ---------------------------------------------------------------------------
# Per-person aggregation metrics (computed after all frames)
# ---------------------------------------------------------------------------

@dataclass
class PersonMetrics:
    """Aggregated metrics for one tracked person across the entire session."""
    person_id: str
    total_frames_seen: int
    gaze_contact_percentage: float  # 0.0 - 100.0
    dominant_emotion: str
    emotion_distribution: Dict[str, float]  # emotion label -> proportion
    average_body_orientation: float  # degrees
    gesture_counts: Dict[str, int]  # gesture_type -> count
    attention_score: float  # 0.0 - 1.0 weighted composite

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
    """Aggregated metrics for the entire video analysis session."""
    total_persons: int
    total_frames: int
    duration: float  # seconds
    per_person_metrics: List[PersonMetrics] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_persons": self.total_persons,
            "total_frames": self.total_frames,
            "duration": self.duration,
            "per_person_metrics": [m.to_dict() for m in self.per_person_metrics],
        }


# ---------------------------------------------------------------------------
# Full-video result
# ---------------------------------------------------------------------------

@dataclass
class VisionResult:
    """Complete output of the vision pipeline for one video."""
    video_path: str
    total_frames: int
    fps_processed: float
    duration_seconds: float
    frames: List[FrameResult] = field(default_factory=list)
    processing_time_seconds: float = 0.0
    person_embeddings: Dict[str, Any] = field(default_factory=dict)
    session_metrics: Optional[SessionMetrics] = None

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict for Celery/API transport.

        numpy arrays inside PersonFrame are converted to plain lists so the
        result can be passed through ``json.dumps`` without a custom encoder.
        """
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
        }
        if self.session_metrics is not None:
            result["session_metrics"] = self.session_metrics.to_dict()
        return result
