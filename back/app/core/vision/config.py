"""Configuration dataclass for the vision processing pipeline.

``VisionConfig`` centralises all tuneable parameters so they can be
overridden from environment variables (via ``app.utils.config.Settings``)
or passed directly in code / tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class VisionConfig:
    """All configuration knobs for the vision pipeline and its sub-processors."""

    # ------------------------------------------------------------------
    # Frame extraction
    # ------------------------------------------------------------------
    fps: float = 3.0
    max_frames: int = 500

    # ------------------------------------------------------------------
    # Face detection (MediaPipe Face Mesh)
    # ------------------------------------------------------------------
    max_faces: int = 10
    min_face_confidence: float = 0.5

    # ------------------------------------------------------------------
    # Emotion classification (CLIP zero-shot)
    # ------------------------------------------------------------------
    emotion_labels: List[str] = field(default_factory=lambda: [
        "happy",
        "sad",
        "angry",
        "surprised",
        "fearful",
        "disgusted",
        "contemptuous",
        "neutral",
        "attentive",
    ])
    clip_model: str = "openai/clip-vit-base-patch32"

    # ------------------------------------------------------------------
    # Gaze estimation (geometric, from iris landmarks)
    # ------------------------------------------------------------------
    gaze_camera_threshold: float = 15.0  # degrees from centre to count as "camera"

    # ------------------------------------------------------------------
    # Gesture detection (temporal landmark displacement)
    # ------------------------------------------------------------------
    gesture_window_size: int = 10  # number of frames in sliding window
    nod_threshold: float = 0.03

    # ------------------------------------------------------------------
    # Pose estimation (MediaPipe Pose)
    # ------------------------------------------------------------------
    min_pose_confidence: float = 0.5

    # ------------------------------------------------------------------
    # Person tracking (face embeddings)
    # ------------------------------------------------------------------
    embedding_similarity_threshold: float = 0.6

    # ------------------------------------------------------------------
    # Device selection
    # ------------------------------------------------------------------
    device: str = "auto"  # "auto", "cpu", "cuda"

    # ------------------------------------------------------------------
    # Module toggles — allow disabling individual sub-processors
    # ------------------------------------------------------------------
    enable_emotion: bool = True
    enable_gaze: bool = True
    enable_gesture: bool = True
    enable_pose: bool = True
    enable_tracking: bool = True
