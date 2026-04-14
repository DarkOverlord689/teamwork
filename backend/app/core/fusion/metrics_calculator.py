"""Metrics calculation for the multimodal fusion pipeline (Module 2.3).

``MetricsCalculator`` computes participation equity (CV), visual contact
percentages, interruption rates, and turn synchronization scores from
the aligned feature set produced by ``TemporalAligner``.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List

from app.core.audio.data_types import AudioResult
from app.core.fusion.config import FusionConfig
from app.core.fusion.data_types import (
    AlignedFeatures,
    GroupMetrics,
    StudentMetrics,
)

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Compute collaboration metrics from aligned audio + visual features.

    Parameters
    ----------
    config : FusionConfig
        Fusion pipeline configuration.
    """

    def __init__(self, config: FusionConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_student_metrics(
        self,
        aligned: AlignedFeatures,
        audio_result: AudioResult,
    ) -> List[StudentMetrics]:
        """Compute per-student fused metrics.

        Parameters
        ----------
        aligned : AlignedFeatures
            Temporal alignment result.
        audio_result : AudioResult
            Original audio pipeline output (for interruption data).

        Returns
        -------
        list[StudentMetrics]
            One entry per unique speaker/person identified.
        """
        if audio_result.session_metrics is None:
            logger.warning("AudioResult has no session_metrics; metrics will be approximate")

        total_speaking_time = sum(
            t.duration for t in audio_result.turns
        ) or 1.0  # avoid div-by-zero

        speaker_metrics: Dict[str, dict] = {}

        # Index interruption counts from audio result
        interrupter_counts: Dict[str, int] = {}
        interrupted_counts: Dict[str, int] = {}
        for intr in audio_result.interruptions:
            interrupter_counts[intr.interrupter_id] = (
                interrupter_counts.get(intr.interrupter_id, 0) + 1
            )
            interrupted_counts[intr.interrupted_id] = (
                interrupted_counts.get(intr.interrupted_id, 0) + 1
            )

        # Aggregate per-speaker metrics from turns
        for turn in audio_result.turns:
            sid = turn.speaker_id
            if sid not in speaker_metrics:
                speaker_metrics[sid] = {
                    "speaking_time_seconds": 0.0,
                    "turn_count": 0,
                }
            speaker_metrics[sid]["speaking_time_seconds"] += turn.duration
            speaker_metrics[sid]["turn_count"] += 1

        # Aggregate gaze + visual metrics from aligned windows
        gaze_counts: Dict[str, int] = {}
        gaze_total: Dict[str, int] = {}
        body_sums: Dict[str, float] = {}
        body_counts: Dict[str, int] = {}
        emotion_votes: Dict[str, Dict[str, int]] = {}

        for w in aligned.windows:
            # Determine who is active in this window
            sid = w.speaker_id
            pid = w.person_id

            # Prefer speaker_id as canonical student key; fall back to person_id
            student_key = sid or pid
            if student_key is None:
                continue

            if student_key not in gaze_total:
                gaze_total[student_key] = 0
                gaze_counts[student_key] = 0
                body_sums[student_key] = 0.0
                body_counts[student_key] = 0
                emotion_votes[student_key] = {}

            gaze_total[student_key] += 1
            if w.gaze_at_camera:
                gaze_counts[student_key] += 1
            body_sums[student_key] += w.body_orientation
            body_counts[student_key] += 1
            emotion_votes[student_key][w.emotion] = (
                emotion_votes[student_key].get(w.emotion, 0) + 1
            )

        # Identify initiation turns: turns that start after silence or after a
        # different speaker (topic-starter heuristic)
        initiation_counts: Dict[str, int] = {}
        sorted_turns = sorted(audio_result.turns, key=lambda t: t.start)
        for i, turn in enumerate(sorted_turns):
            if i == 0:
                initiation_counts[turn.speaker_id] = (
                    initiation_counts.get(turn.speaker_id, 0) + 1
                )
            else:
                prev = sorted_turns[i - 1]
                gap = turn.start - prev.end
                if gap > 2.0 or prev.speaker_id != turn.speaker_id:
                    initiation_counts[turn.speaker_id] = (
                        initiation_counts.get(turn.speaker_id, 0) + 1
                    )

        # Identify back-channel turns (< min_turn_duration)
        back_channel_counts: Dict[str, int] = {}
        for turn in audio_result.turns:
            if turn.duration < self.config.min_turn_duration:
                back_channel_counts[turn.speaker_id] = (
                    back_channel_counts.get(turn.speaker_id, 0) + 1
                )

        # Build StudentMetrics list
        all_student_ids = set(speaker_metrics.keys()) | set(gaze_total.keys())
        metrics_list: List[StudentMetrics] = []

        for sid in sorted(all_student_ids):
            sm = speaker_metrics.get(sid, {})
            speaking_time = sm.get("speaking_time_seconds", 0.0)
            turn_count = sm.get("turn_count", 0)
            participation_ratio = speaking_time / total_speaking_time

            gt = gaze_total.get(sid, 0)
            gaze_pct = (gaze_counts.get(sid, 0) / gt * 100.0) if gt > 0 else 0.0

            bc = body_counts.get(sid, 0)
            avg_body = (body_sums.get(sid, 0.0) / bc) if bc > 0 else 0.0

            ev = emotion_votes.get(sid, {})
            dominant_emotion = max(ev, key=lambda k: ev[k]) if ev else "neutral"

            metrics_list.append(
                StudentMetrics(
                    student_id=sid,
                    speaking_time_seconds=speaking_time,
                    turn_count=turn_count,
                    interruption_count=interrupter_counts.get(sid, 0),
                    interrupted_count=interrupted_counts.get(sid, 0),
                    participation_ratio=participation_ratio,
                    gaze_contact_percentage=gaze_pct,
                    avg_body_orientation=avg_body,
                    dominant_emotion=dominant_emotion,
                    initiation_count=initiation_counts.get(sid, 0),
                    back_channel_count=back_channel_counts.get(sid, 0),
                )
            )

        return metrics_list

    def calculate_participation_cv(self, student_metrics: List[StudentMetrics]) -> float:
        """Compute the coefficient of variation of speaking times.

        CV = std / mean.  A value below ``config.cv_threshold`` (0.3) indicates
        equitable participation.

        Parameters
        ----------
        student_metrics : list[StudentMetrics]

        Returns
        -------
        float
            CV value; 0.0 when there is only one speaker.
        """
        times = [m.speaking_time_seconds for m in student_metrics]
        if len(times) < 2:
            return 0.0
        mean = sum(times) / len(times)
        if mean == 0.0:
            return 0.0
        variance = sum((t - mean) ** 2 for t in times) / len(times)
        return math.sqrt(variance) / mean

    def calculate_disruptive_interruption_rate(
        self,
        audio_result: AudioResult,
    ) -> float:
        """Compute disruptive interruptions / total turns.

        Returns
        -------
        float
            Rate in [0, 1]; 0.0 when there are no turns.
        """
        from app.core.audio.data_types import InterruptionType

        total_turns = len(audio_result.turns)
        if total_turns == 0:
            return 0.0

        disruptive = sum(
            1
            for i in audio_result.interruptions
            if i.interruption_type == InterruptionType.DISRUPTIVE
        )
        return disruptive / total_turns

    def calculate_turn_synchronization_score(
        self,
        audio_result: AudioResult,
    ) -> float:
        """Compute the ratio of smooth turn transitions to total transitions.

        A transition is "smooth" when the gap between consecutive turns is
        between 0 and ``interruption_gap_threshold`` seconds (a brief natural
        pause, not an overlap and not a long silence).

        Returns
        -------
        float
            Score in [0, 1]; 1.0 = perfectly synchronised.
        """
        turns = sorted(audio_result.turns, key=lambda t: t.start)
        if len(turns) < 2:
            return 1.0

        smooth_count = 0
        total_transitions = len(turns) - 1

        for i in range(total_transitions):
            gap = turns[i + 1].start - turns[i].end
            # Smooth: small positive gap (natural pause, no overlap, no long silence)
            if 0.0 <= gap <= 1.0:
                smooth_count += 1

        return smooth_count / total_transitions if total_transitions > 0 else 1.0

    def calculate_group_metrics(
        self,
        aligned: AlignedFeatures,
        audio_result: AudioResult,
        student_metrics: List[StudentMetrics],
    ) -> GroupMetrics:
        """Compute group-level collaboration metrics.

        Parameters
        ----------
        aligned : AlignedFeatures
        audio_result : AudioResult
        student_metrics : list[StudentMetrics]
            Pre-computed per-student metrics.

        Returns
        -------
        GroupMetrics
        """
        participation_cv = self.calculate_participation_cv(student_metrics)
        disruptive_rate = self.calculate_disruptive_interruption_rate(audio_result)
        turn_sync = self.calculate_turn_synchronization_score(audio_result)

        avg_gaze = (
            sum(m.gaze_contact_percentage for m in student_metrics) / len(student_metrics)
            if student_metrics
            else 0.0
        )

        # Silence ratio: windows with no speaker
        total_windows = len(aligned.windows) or 1
        silent_windows = sum(1 for w in aligned.windows if w.speaker_id is None)
        silence_ratio = silent_windows / total_windows

        return GroupMetrics(
            total_students=len(student_metrics),
            duration_seconds=aligned.duration_seconds,
            participation_cv=participation_cv,
            disruptive_interruption_rate=disruptive_rate,
            turn_synchronization_score=turn_sync,
            avg_gaze_contact_percentage=avg_gaze,
            silence_ratio=silence_ratio,
            per_student_metrics=student_metrics,
        )
