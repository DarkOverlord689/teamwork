"""Vision processing module for SMATC-UPAO visual analysis pipeline."""

from .data_types import (
    GazeCategory,
    GazeData,
    GestureType,
    GestureData,
    PoseData,
    EmotionData,
    PersonFrame,
    FrameResult,
    PersonMetrics,
    SessionMetrics,
    VisionResult,
)
from .base_processor import BaseProcessor
from .config import VisionConfig
from .frame_extractor import FrameExtractor
from .face_detector import FaceDetector
from .emotion_classifier import EmotionClassifier
from .gaze_estimator import GazeEstimator
from .gesture_analyzer import GestureAnalyzer
from .pose_estimator import PoseEstimator
from .person_tracker import PersonTracker
from .pipeline import VisionPipeline

__all__ = [
    # Data types
    "GazeCategory",
    "GazeData",
    "GestureType",
    "GestureData",
    "PoseData",
    "EmotionData",
    "PersonFrame",
    "FrameResult",
    "PersonMetrics",
    "SessionMetrics",
    "VisionResult",
    # Base
    "BaseProcessor",
    "VisionConfig",
    # Processors
    "FrameExtractor",
    "FaceDetector",
    "EmotionClassifier",
    "GazeEstimator",
    "GestureAnalyzer",
    "PoseEstimator",
    "PersonTracker",
    # Pipeline
    "VisionPipeline",
]
