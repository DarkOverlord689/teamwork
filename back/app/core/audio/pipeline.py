"""Audio pipeline orchestrator.

``AudioPipeline`` manages the full lifecycle of audio analysis: extraction,
diarization, transcription, turn analysis, interruption detection, and
participation aggregation.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Callable

from app.core.audio.audio_extractor import AudioExtractor
from app.core.audio.config import AudioConfig
from app.core.audio.data_types import AudioResult, SpeakerSegment, TranscriptSegment
from app.core.audio.diarizer import Diarizer
from app.core.audio.interruption_detector import InterruptionDetector
from app.core.audio.participation_aggregator import ParticipationAggregator
from app.core.audio.transcriber import Transcriber
from app.core.audio.turn_analyzer import TurnAnalyzer

logger = logging.getLogger(__name__)


class AudioPipeline:
    """Full audio analysis pipeline.

    Use as a context manager to handle automatic model loading and unloading:

    .. code-block:: python

        with AudioPipeline(config) as pipeline:
            result = pipeline.process_audio(video_path)

    Parameters
    ----------
    config : AudioConfig, optional
        Pipeline-wide configuration.  Defaults to ``AudioConfig()`` when not
        provided.
    """

    def __init__(self, config: AudioConfig | None = None) -> None:
        self.config = config or AudioConfig()

        self._extractor = AudioExtractor(self.config)
        self._diarizer = Diarizer(self.config) if self.config.enable_diarization else None
        self._transcriber = Transcriber(self.config) if self.config.enable_transcription else None
        self._turn_analyzer = TurnAnalyzer(self.config)
        self._interruption_detector = (
            InterruptionDetector(self.config) if self.config.enable_interruption else None
        )
        self._aggregator = (
            ParticipationAggregator(self.config) if self.config.enable_participation else None
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load_all(self) -> None:
        """Load all ML models into memory."""
        logger.info("Loading audio pipeline processors...")
        self._extractor.load()
        if self._diarizer is not None:
            self._diarizer.load()
        if self._transcriber is not None:
            self._transcriber.load()
        logger.info("Audio pipeline processors loaded.")

    def unload_all(self) -> None:
        """Release all model resources."""
        logger.info("Unloading audio pipeline processors...")
        self._extractor.unload()
        if self._diarizer is not None:
            self._diarizer.unload()
        if self._transcriber is not None:
            self._transcriber.unload()
        logger.info("Audio pipeline processors unloaded.")

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "AudioPipeline":
        self.load_all()
        return self

    def __exit__(self, *args: object) -> None:
        self.unload_all()

    # ------------------------------------------------------------------
    # Main processing
    # ------------------------------------------------------------------

    def process_audio(
        self,
        video_path: str,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> AudioResult:
        """Run the full pipeline on *video_path* and return an ``AudioResult``.

        Parameters
        ----------
        video_path : str
            Path to the video (or audio) file on disk.
        progress_callback : callable, optional
            Called with ``(fraction: float, message: str)`` as each stage
            completes.  ``fraction`` increases monotonically from 0.0 to 1.0.

        Returns
        -------
        AudioResult
        """
        start_time = time.time()

        def _progress(p: float, msg: str) -> None:
            if progress_callback is not None:
                progress_callback(p, msg)

        # ------------------------------------------------------------------
        # Step 1 — Audio extraction
        # ------------------------------------------------------------------
        _progress(0.0, "Starting audio extraction")
        waveform, sample_rate, duration = self._extractor.process(video_path)
        temp_wav_path = self._extractor.write_temp_wav(waveform, sample_rate)
        _progress(0.10, "Audio extracted")

        # ------------------------------------------------------------------
        # Step 2 — Speaker diarization
        # ------------------------------------------------------------------
        segments: list[SpeakerSegment] = []
        if self._diarizer is not None:
            _progress(0.15, "Starting diarization")
            segments = self._diarizer.process(temp_wav_path)
            _progress(0.40, "Diarization complete")

        # ------------------------------------------------------------------
        # Step 3 — Transcription
        # ------------------------------------------------------------------
        transcripts: list[TranscriptSegment] = []
        if self._transcriber is not None and segments:
            _progress(0.45, "Starting transcription")
            transcripts = self._transcriber.process(waveform, segments)
            _progress(0.75, "Transcription complete")

        # ------------------------------------------------------------------
        # Step 4 — Turn analysis
        # ------------------------------------------------------------------
        _progress(0.80, "Analyzing turns")
        turn_result = self._turn_analyzer.process(segments, total_duration=duration)

        # ------------------------------------------------------------------
        # Step 5 — Interruption detection
        # ------------------------------------------------------------------
        interruptions = []
        if self._interruption_detector is not None:
            _progress(0.85, "Detecting interruptions")
            interruptions = self._interruption_detector.process(
                turn_result.turns,
                transcripts,
                turn_result.overlaps,
            )

        # ------------------------------------------------------------------
        # Step 6 — Participation aggregation
        # ------------------------------------------------------------------
        session_metrics = None
        if self._aggregator is not None:
            _progress(0.90, "Aggregating metrics")
            session_metrics = self._aggregator.process(
                turn_result,
                interruptions,
                transcripts,
                duration,
            )

        # ------------------------------------------------------------------
        # Cleanup temp WAV
        # ------------------------------------------------------------------
        try:
            os.unlink(temp_wav_path)
        except Exception:
            pass

        processing_time = time.time() - start_time
        _progress(1.0, "Audio analysis complete")

        logger.info(
            "Audio analysis complete: %.1fs processing time (video duration %.1fs)",
            processing_time,
            duration,
        )

        return AudioResult(
            video_path=video_path,
            duration_seconds=duration,
            sample_rate=sample_rate,
            segments=segments,
            turns=turn_result.turns,
            transcripts=transcripts,
            interruptions=interruptions,
            processing_time_seconds=processing_time,
            session_metrics=session_metrics,
        )
