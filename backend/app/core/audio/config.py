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
    # Diarization (Deepgram)
    # ------------------------------------------------------------------
    diarize_min_speakers: int = 2
    diarize_max_speakers: int = 8
    diarize_min_duration: float = 0.5
    deepgram_api_key: str = ""

    # ------------------------------------------------------------------
    # Transcription (Deepgram)
    # Replaces OpenAI Whisper API and local Whisper model.
    # ------------------------------------------------------------------
    deepgram_model: str = "nova-2"
    whisper_language: str = "es"
    whisper_no_speech_threshold: float = 0.6
    whisper_initial_prompt: str | None = None
    whisper_model_size: str = "small"
    whisper_beam_size: int = 5

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

    # ------------------------------------------------------------------
    # Smart frame selection
    # ------------------------------------------------------------------
    sfs_speech_onset_offset_ms: float = 200.0  # ms after segment start
    sfs_question_lookahead_s: float = 2.0  # window after "?"
    sfs_hesitation_fillers: list | None = None  # populated in __post_init__
    sfs_hesitation_pause_threshold_ms: float = 800.0
    sfs_long_turn_threshold_s: float = 15.0
    sfs_post_silence_min_gap_ms: float = 1500.0
    sfs_max_frames_per_participant: int = 10
    sfs_max_frames_total: int = 60

    def __post_init__(self) -> None:
        if self.sfs_hesitation_fillers is None:
            self.sfs_hesitation_fillers = [
                "eh",
                "um",
                "uh",
                "este",
                "mmm",
                "bueno",
                "o sea",
            ]

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary.

        Note: ``deepgram_api_key`` is intentionally excluded to avoid leaking
        secrets into logs, task results, or API responses.
        """
        return {
            "audio_sample_rate": self.audio_sample_rate,
            "audio_device": self.audio_device,
            "diarize_min_speakers": self.diarize_min_speakers,
            "diarize_max_speakers": self.diarize_max_speakers,
            "diarize_min_duration": self.diarize_min_duration,
            "deepgram_model": self.deepgram_model,
            "whisper_language": self.whisper_language,
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
