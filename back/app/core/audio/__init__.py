"""Audio processing module for SMATC-UPAO auditory analysis pipeline (Module 2.2)."""

from app.core.audio.base_processor import AudioBaseProcessor
from app.core.audio.config import AudioConfig
from app.core.audio.data_types import (
    AudioResult,
    AudioSessionMetrics,
    Interruption,
    InterruptionType,
    SpeakerMetrics,
    SpeakerRole,
    SpeakerSegment,
    SpeakerTurn,
    TranscriptSegment,
    WordTimestamp,
)

from app.core.audio.pipeline import AudioPipeline

__all__ = [
    # Base
    "AudioBaseProcessor",
    "AudioConfig",
    # Data types — enums
    "SpeakerRole",
    "InterruptionType",
    # Data types — primitives
    "WordTimestamp",
    "SpeakerSegment",
    "SpeakerTurn",
    "TranscriptSegment",
    "Interruption",
    "SpeakerMetrics",
    "AudioSessionMetrics",
    "AudioResult",
    # Pipeline (available after Phase 2)
    "AudioPipeline",
]
