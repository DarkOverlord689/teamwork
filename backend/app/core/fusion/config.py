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
    # VALUE rubric weights per dimension (AAC&U Teamwork Rubric)
    # Each sub-dict maps sub-metric names to their contribution weight.
    # ------------------------------------------------------------------
    rubric_weights: dict = field(
        default_factory=lambda: {
            "contributes_to_team_meetings": {
                "participation_engagement": 0.40,  # participation_ratio vs equal_share
                "idea_contribution": 0.35,  # initiation_count relative to turns
                "turn_frequency": 0.25,  # turn_count relative to group
            },
            "facilitates_contributions": {
                "active_listening": 0.35,  # back_channel_count + gaze during others
                "cooperative_engagement": 0.40,  # cooperative interruptions + gaze
                "engagement_signals": 0.25,  # gaze when others speak
            },
            "fosters_constructive_climate": {
                "emotional_tone": 0.35,  # dominant_emotion positivity
                "respect_signals": 0.35,  # low disruptive interruptions + high gaze
                "body_language": 0.30,  # body_orientation + gestures
            },
            "responds_to_conflict": {
                "conflict_management": 0.50,  # disruptive_rate + turn_sync
                "constructive_engagement": 0.50,  # cooperative vs disruptive balance
            },
            "individual_contributions_outside": {
                "placeholder": 1.00,  # cannot be measured from session recording
            },
        }
    )

    # ------------------------------------------------------------------
    # VALUE level thresholds (mapped from 1-4 to 0-20)
    # ------------------------------------------------------------------
    value_level_benchmark: float = 5.0  # VALUE Level 1 → 5.0
    value_level_milestone_2: float = 10.0  # VALUE Level 2 → 10.0
    value_level_milestone_3: float = 15.0  # VALUE Level 3 → 15.0
    value_level_capstone: float = 20.0  # VALUE Level 4 → 20.0

    # ------------------------------------------------------------------
    # Emotional tone mapping (positive vs negative emotions)
    # ------------------------------------------------------------------
    positive_emotions: set = field(
        default_factory=lambda: {"attentive", "happy", "surprised", "neutral"}
    )
    negative_emotions: set = field(
        default_factory=lambda: {"angry", "sad", "fearful", "disgusted", "contemptuous"}
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
