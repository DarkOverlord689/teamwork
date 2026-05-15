from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List


class SpeakerRole(str, Enum):
    """Rol de participación inferido para un hablante durante la sesión."""

    UNKNOWN = "unknown"
    LEADER = "leader"
    CONTRIBUTOR = "contributor"
    PASSIVE = "passive"


class InterruptionType(str, Enum):
    """Clasificación semántica de un evento de interrupción detectado."""

    DISRUPTIVE = "disruptive"
    BACK_CHANNEL = "back_channel"
    COOPERATIVE = "cooperative"


class MomentType(str, Enum):
    """Tipo de momento clave para la selección inteligente de frames."""

    SPEECH_ONSET = "speech_onset"
    RECEIVING_QUESTION = "receiving_question"
    HESITATION = "hesitation"
    INTERRUPTION_RECEIVED = "interruption_received"
    POST_QUESTION_SILENCE = "post_question_silence"
    END_OF_LONG_TURN = "end_of_long_turn"
    BACK_CHANNEL = "back_channel"


@dataclass
class KeyMoment:
    """Un momento clave de conversación seleccionado para capturar frames."""

    timestamp_ms: float
    speaker_id: str
    moment_type: MomentType
    priority: int
    context_text: str


@dataclass
class WordTimestamp:
    """Tiempo y confianza para una sola palabra reconocida."""

    word: str
    start: float
    end: float
    confidence: float

    def to_dict(self) -> dict:
        return {
            "word": self.word,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
        }


@dataclass
class SpeakerSegment:
    """Un segmento de voz continuo asignado a un solo hablante."""

    start: float
    end: float
    speaker_id: str
    confidence: float | None = None

    def to_dict(self) -> dict:
        result: dict = {
            "start": self.start,
            "end": self.end,
            "speaker_id": self.speaker_id,
        }
        if self.confidence is not None:
            result["confidence"] = self.confidence
        return result


@dataclass
class SpeakerTurn:
    """Un turno de palabra - uno o más segmentos consecutivos del mismo hablante."""

    start: float
    end: float
    speaker_id: str
    duration: float
    segment_count: int = 1

    def to_dict(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "speaker_id": self.speaker_id,
            "duration": self.duration,
            "segment_count": self.segment_count,
        }


@dataclass
class TranscriptSegment:
    """Transcripción de Whisper alineada a un segmento de hablante."""

    start: float
    end: float
    speaker_id: str
    text: str
    words: List[WordTimestamp] = field(default_factory=list)
    language: str = "es"
    no_speech_prob: float = 0.0

    def to_dict(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "speaker_id": self.speaker_id,
            "text": self.text,
            "words": [w.to_dict() for w in self.words],
            "language": self.language,
            "no_speech_prob": self.no_speech_prob,
        }


@dataclass
class Interruption:
    """Un evento de interrupción detectado entre dos hablantes."""

    time: float
    interrupter_id: str
    interrupted_id: str
    overlap_duration: float
    interruption_type: str = InterruptionType.DISRUPTIVE

    def to_dict(self) -> dict:
        return {
            "time": self.time,
            "interrupter_id": self.interrupter_id,
            "interrupted_id": self.interrupted_id,
            "overlap_duration": self.overlap_duration,
            "interruption_type": self.interruption_type,
        }


@dataclass
class SpeakerMetrics:
    """Métricas de participación agregadas para un hablante."""

    speaker_id: str
    speaking_time_seconds: float
    turn_count: int
    interruption_count: int
    interrupted_count: int
    avg_turn_duration: float
    participation_ratio: float
    back_channel_count: int = 0

    def to_dict(self) -> dict:
        return {
            "speaker_id": self.speaker_id,
            "speaking_time_seconds": self.speaking_time_seconds,
            "turn_count": self.turn_count,
            "interruption_count": self.interruption_count,
            "interrupted_count": self.interrupted_count,
            "avg_turn_duration": self.avg_turn_duration,
            "participation_ratio": self.participation_ratio,
            "back_channel_count": self.back_channel_count,
        }


@dataclass
class AudioSessionMetrics:
    """Métricas agregadas para toda la sesión de análisis de audio."""

    total_speakers: int
    duration: float
    total_speaking_time: float
    silence_ratio: float
    participation_cv: float
    turn_alternation_rate: float
    per_speaker_metrics: List[SpeakerMetrics] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_speakers": self.total_speakers,
            "duration": self.duration,
            "total_speaking_time": self.total_speaking_time,
            "silence_ratio": self.silence_ratio,
            "participation_cv": self.participation_cv,
            "turn_alternation_rate": self.turn_alternation_rate,
            "per_speaker_metrics": [m.to_dict() for m in self.per_speaker_metrics],
        }


@dataclass
class AudioResult:
    """Resultado completo del pipeline de audio para un archivo de video."""

    video_path: str
    duration_seconds: float
    sample_rate: int
    segments: List[SpeakerSegment] = field(default_factory=list)
    turns: List[SpeakerTurn] = field(default_factory=list)
    transcripts: List[TranscriptSegment] = field(default_factory=list)
    interruptions: List[Interruption] = field(default_factory=list)
    processing_time_seconds: float = 0.0
    session_metrics: AudioSessionMetrics | None = None

    def to_dict(self) -> dict:
        result: dict = {
            "video_path": self.video_path,
            "duration_seconds": self.duration_seconds,
            "sample_rate": self.sample_rate,
            "processing_time_seconds": self.processing_time_seconds,
            "segments": [s.to_dict() for s in self.segments],
            "turns": [t.to_dict() for t in self.turns],
            "transcripts": [t.to_dict() for t in self.transcripts],
            "interruptions": [i.to_dict() for i in self.interruptions],
        }
        if self.session_metrics is not None:
            result["session_metrics"] = self.session_metrics.to_dict()
        return result
