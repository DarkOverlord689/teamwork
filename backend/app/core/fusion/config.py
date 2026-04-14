"""Configuration dataclass for the multimodal fusion pipeline (Module 2.3)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FusionConfig:
    """Configuration for the multimodal fusion pipeline.

    All fields can be overridden when constructing the pipeline.  Sensible
    defaults are provided for development use; see inline comments for
    production recommendations.
    """

    # ------------------------------------------------------------------
    # Temporal alignment
    # ------------------------------------------------------------------
    temporal_window_ms: int = 500  # sliding window size in milliseconds
    min_overlap_ratio: float = 0.3  # min audio/visual overlap to consider aligned
    face_similarity_threshold: float = 0.6  # cosine similarity to match speaker -> face

    # ------------------------------------------------------------------
    # Participation equity
    # ------------------------------------------------------------------
    cv_threshold: float = 0.3  # target coefficient of variation (equity)
    min_turn_duration: float = 0.5  # seconds; shorter turns are back-channel only

    # ------------------------------------------------------------------
    # Rubric weights per dimension
    # Each sub-dict maps metric names to their contribution weight (summing to 1.0).
    # ------------------------------------------------------------------
    rubric_weights: dict = field(
        default_factory=lambda: {
            "collaboration": {
                "turn_equity": 0.5,  # low CV = high equity
                "gaze_contact": 0.5,
            },
            "communication": {
                "low_interruption_rate": 0.5,
                "turn_synchronization": 0.5,
            },
            "responsibility": {
                "participation_consistency": 0.6,
                "cv_penalty": 0.4,
            },
            "leadership": {
                "speaking_share": 0.4,
                "initiation_rate": 0.6,
            },
            "technical_contribution": {
                "turn_count_share": 0.5,
                "keyword_density": 0.5,
            },
        }
    )

    # ------------------------------------------------------------------
    # LLM / OpenAI
    # ------------------------------------------------------------------
    llm_model: str = "gpt-4o-mini"
    openai_api_key: str = ""
    max_explanation_tokens: int = 1024
    llm_timeout_seconds: float = 60.0
    llm_temperature: float = 0.3  # lower = more deterministic

    # ------------------------------------------------------------------
    # Feature flags
    # ------------------------------------------------------------------
    enable_explanation: bool = True  # set False to skip LLM call entirely

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "temporal_window_ms": self.temporal_window_ms,
            "min_overlap_ratio": self.min_overlap_ratio,
            "face_similarity_threshold": self.face_similarity_threshold,
            "cv_threshold": self.cv_threshold,
            "min_turn_duration": self.min_turn_duration,
            "rubric_weights": dict(self.rubric_weights),
            "llm_model": self.llm_model,
            "max_explanation_tokens": self.max_explanation_tokens,
            "llm_timeout_seconds": self.llm_timeout_seconds,
            "llm_temperature": self.llm_temperature,
            "enable_explanation": self.enable_explanation,
        }
