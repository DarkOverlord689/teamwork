"""Participation metrics aggregator for the audio processing pipeline.

``ParticipationAggregator`` combines turn analysis results, detected
interruptions, and transcripts into a single ``AudioSessionMetrics`` object
that summarises the full session.
"""

from __future__ import annotations

from app.core.audio.config import AudioConfig
from app.core.audio.data_types import (
    AudioSessionMetrics,
    Interruption,
    InterruptionType,
    SpeakerMetrics,
    TranscriptSegment,
)
from app.core.audio.turn_analyzer import TurnAnalysisResult


class ParticipationAggregator:
    """Aggregate per-speaker and session-level participation metrics.

    Parameters
    ----------
    config : AudioConfig
        Pipeline-wide configuration (reserved for future use / thresholds).
    """

    def __init__(self, config: AudioConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        turn_result: TurnAnalysisResult,
        interruptions: list[Interruption],
        transcripts: list[TranscriptSegment],
        total_duration: float,
    ) -> AudioSessionMetrics:
        """Aggregate all metrics into a single ``AudioSessionMetrics`` object.

        Parameters
        ----------
        turn_result : TurnAnalysisResult
            Structured output of ``TurnAnalyzer.process()``.
        interruptions : list[Interruption]
            Classified interruption events.
        transcripts : list[TranscriptSegment]
            Speaker-aligned Whisper transcripts (may be empty).
        total_duration : float
            Total session duration in seconds.

        Returns
        -------
        AudioSessionMetrics
        """
        speaking_time = turn_result.speaking_time
        turn_count = turn_result.turn_count
        avg_turn_duration = turn_result.avg_turn_duration

        # Per-speaker interruption counters.
        interruption_count: dict[str, int] = {}
        interrupted_count: dict[str, int] = {}
        back_channel_count: dict[str, int] = {}

        for ev in interruptions:
            interruption_count[ev.interrupter_id] = (
                interruption_count.get(ev.interrupter_id, 0) + 1
            )
            interrupted_count[ev.interrupted_id] = (
                interrupted_count.get(ev.interrupted_id, 0) + 1
            )
            if ev.interruption_type == InterruptionType.BACK_CHANNEL:
                back_channel_count[ev.interrupter_id] = (
                    back_channel_count.get(ev.interrupter_id, 0) + 1
                )

        # Participation ratio per speaker.
        total_speaking = sum(speaking_time.values())
        participation_ratio: dict[str, float] = {}
        if total_speaking > 0:
            for spk, t in speaking_time.items():
                participation_ratio[spk] = t / total_speaking
        else:
            for spk in speaking_time:
                participation_ratio[spk] = 0.0

        # Build per-speaker metrics list.
        all_speakers = set(speaking_time.keys())
        per_speaker: list[SpeakerMetrics] = []
        for spk in sorted(all_speakers):
            per_speaker.append(
                SpeakerMetrics(
                    speaker_id=spk,
                    speaking_time_seconds=speaking_time.get(spk, 0.0),
                    turn_count=turn_count.get(spk, 0),
                    interruption_count=interruption_count.get(spk, 0),
                    interrupted_count=interrupted_count.get(spk, 0),
                    avg_turn_duration=avg_turn_duration.get(spk, 0.0),
                    participation_ratio=participation_ratio.get(spk, 0.0),
                    back_channel_count=back_channel_count.get(spk, 0),
                )
            )

        # Session-level metrics.
        silence_ratio = 0.0
        if total_duration > 0:
            silence_ratio = (total_duration - total_speaking) / total_duration
            silence_ratio = max(0.0, min(1.0, silence_ratio))

        return AudioSessionMetrics(
            total_speakers=len(all_speakers),
            duration=total_duration,
            total_speaking_time=total_speaking,
            silence_ratio=silence_ratio,
            participation_cv=turn_result.participation_cv,
            turn_alternation_rate=turn_result.turn_alternation_rate,
            per_speaker_metrics=per_speaker,
        )
