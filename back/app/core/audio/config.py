"""Configuration dataclass for the audio processing pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AudioConfig:
    """Configuration for the audio processing pipeline (Module 2.2).

    All fields can be overridden when constructing the pipeline.  Sensible
    defaults are provided for development use; see inline comments for
    production recommendations.
    """

    # ------------------------------------------------------------------
    # Audio extraction
    # ------------------------------------------------------------------
    audio_sample_rate: int = 16000
    audio_device: str = "auto"

    # ------------------------------------------------------------------
    # Diarization (Pyannote)
    # ------------------------------------------------------------------
    diarize_min_speakers: int = 2
    diarize_max_speakers: int = 8
    diarize_min_duration: float = 0.5
    pyannote_auth_token: str = ""

    # ------------------------------------------------------------------
    # Transcription (Whisper)
    # NOTE: Use "medium" as default for dev (2 GB).
    #       Use "large-v3" for production (~10 GB VRAM).
    # ------------------------------------------------------------------
    whisper_model_size: str = "medium"
    whisper_language: str = "es"
    whisper_beam_size: int = 5
    whisper_no_speech_threshold: float = 0.6
    # Domain vocab priming for Spanish technical terms
    whisper_initial_prompt: str | None = None

    # ------------------------------------------------------------------
    # Turn analysis
    # ------------------------------------------------------------------
    interruption_overlap_threshold: float = 0.3
    interruption_gap_threshold: float = 0.2
    turn_min_gap: float = 0.5

    # ------------------------------------------------------------------
    # Feature flags
    # ------------------------------------------------------------------
    enable_diarization: bool = True
    enable_transcription: bool = True
    enable_interruption: bool = True
    enable_participation: bool = True

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "audio_sample_rate": self.audio_sample_rate,
            "audio_device": self.audio_device,
            "diarize_min_speakers": self.diarize_min_speakers,
            "diarize_max_speakers": self.diarize_max_speakers,
            "diarize_min_duration": self.diarize_min_duration,
            "pyannote_auth_token": self.pyannote_auth_token,
            "whisper_model_size": self.whisper_model_size,
            "whisper_language": self.whisper_language,
            "whisper_beam_size": self.whisper_beam_size,
            "whisper_no_speech_threshold": self.whisper_no_speech_threshold,
            "whisper_initial_prompt": self.whisper_initial_prompt,
            "interruption_overlap_threshold": self.interruption_overlap_threshold,
            "interruption_gap_threshold": self.interruption_gap_threshold,
            "turn_min_gap": self.turn_min_gap,
            "enable_diarization": self.enable_diarization,
            "enable_transcription": self.enable_transcription,
            "enable_interruption": self.enable_interruption,
            "enable_participation": self.enable_participation,
        }
