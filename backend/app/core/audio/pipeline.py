"""Pipeline de audio.

AudioPipeline gestiona el ciclo completo del análisis de audio:
extracción, diarización, transcripción, análisis de turnos,
detección de interrupciones y agregación de participación.
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
    def __init__(self, config: AudioConfig | None = None) -> None:
        self.config = config or AudioConfig()

        self._extractor = AudioExtractor(self.config)
        self._diarizer = (
            Diarizer(self.config) if self.config.enable_diarization else None
        )
        self._transcriber = (
            Transcriber(self.config) if self.config.enable_transcription else None
        )
        self._turn_analyzer = TurnAnalyzer(self.config)
        self._interruption_detector = (
            InterruptionDetector(self.config)
            if self.config.enable_interruption
            else None
        )
        self._aggregator = (
            ParticipationAggregator(self.config)
            if self.config.enable_participation
            else None
        )

    def load_all(self) -> None:
        """Carga todos los modelos de ML en memoria."""
        logger.info("Loading audio pipeline processors...")
        self._extractor.load()
        if self._diarizer is not None:
            self._diarizer.load()
        if self._transcriber is not None:
            self._transcriber.load()
        logger.info("Audio pipeline processors loaded.")

    def unload_all(self) -> None:
        """Libera los recursos de los modelos."""
        logger.info("Unloading audio pipeline processors...")
        self._extractor.unload()
        if self._diarizer is not None:
            self._diarizer.unload()
        if self._transcriber is not None:
            self._transcriber.unload()
        logger.info("Audio pipeline processors unloaded.")

    def __enter__(self) -> "AudioPipeline":
        self.load_all()
        return self

    def __exit__(self, *args: object) -> None:
        self.unload_all()

    def process_audio(
        self,
        video_path: str,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> AudioResult:
        """Ejecuta el pipeline completo en el video y retorna un AudioResult."""
        start_time = time.time()

        def _progress(p: float, msg: str) -> None:
            if progress_callback is not None:
                progress_callback(p, msg)

        _progress(0.0, "Starting audio extraction")
        waveform, sample_rate, duration = self._extractor.process(video_path)
        temp_wav_path = self._extractor.write_temp_wav(waveform, sample_rate)
        _progress(0.10, "Audio extracted")

        segments: list[SpeakerSegment] = []
        if self._diarizer is not None:
            _progress(0.15, "Starting diarization")
            segments = self._diarizer.process(temp_wav_path)
            _progress(0.40, "Diarization complete")

        import gc

        if self._diarizer is not None:
            self._diarizer.unload()
        gc.collect()

        transcripts: list[TranscriptSegment] = []
        if self._transcriber is not None and segments:
            _progress(0.45, "Starting transcription")
            transcripts = self._transcriber.process(waveform, segments)
            _progress(0.75, "Transcription complete")

        _progress(0.80, "Analyzing turns")
        turn_result = self._turn_analyzer.process(segments, total_duration=duration)

        interruptions = []
        if self._interruption_detector is not None:
            _progress(0.85, "Detecting interruptions")
            interruptions = self._interruption_detector.process(
                turn_result.turns,
                transcripts,
                turn_result.overlaps,
            )

        session_metrics = None
        if self._aggregator is not None:
            _progress(0.90, "Aggregating metrics")
            session_metrics = self._aggregator.process(
                turn_result,
                interruptions,
                transcripts,
                duration,
            )

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
