"""Turn-level analysis of diariser output.

``TurnAnalyzer`` converts raw ``SpeakerSegment`` lists into structured
``SpeakerTurn`` objects, detects speaker overlaps, and computes participation
statistics — all without any ML models.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from app.core.audio.config import AudioConfig
from app.core.audio.data_types import SpeakerSegment, SpeakerTurn


@dataclass
class TurnAnalysisResult:
    """Structured output of ``TurnAnalyzer.process()``.

    Attributes
    ----------
    turns : list[SpeakerTurn]
        Merged speaker turns (consecutive same-speaker segments within
        ``turn_min_gap`` are merged into one turn).
    overlaps : list[tuple[float, float, str, str]]
        Detected overlap intervals as ``(start, end, speaker_a, speaker_b)``.
    speaking_time : dict[str, float]
        Total active speaking time per speaker (seconds).
    turn_count : dict[str, int]
        Number of turns per speaker.
    avg_turn_duration : dict[str, float]
        Average turn duration per speaker (seconds).
    participation_cv : float
        Coefficient of variation of speaking times — a measure of equity.
        0.0 means perfectly equal participation; higher values indicate
        dominance by one or more speakers.
    turn_alternation_rate : float
        Number of turns per minute across the whole session.
    """

    turns: list[SpeakerTurn] = field(default_factory=list)
    overlaps: list[tuple[float, float, str, str]] = field(default_factory=list)
    speaking_time: dict[str, float] = field(default_factory=dict)
    turn_count: dict[str, int] = field(default_factory=dict)
    avg_turn_duration: dict[str, float] = field(default_factory=dict)
    participation_cv: float = 0.0
    turn_alternation_rate: float = 0.0


class TurnAnalyzer:
    """Pure-Python algorithmic turn analysis — no ML model required.

    Parameters
    ----------
    config : AudioConfig
        Pipeline-wide configuration; ``turn_min_gap`` controls the maximum
        silence gap that is still considered part of the same turn.
    """

    def __init__(self, config: AudioConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        segments: list[SpeakerSegment],
        total_duration: float,
    ) -> TurnAnalysisResult:
        """Analyse a list of speaker segments and return structured results.

        Parameters
        ----------
        segments : list[SpeakerSegment]
            Raw diariser output for the session.
        total_duration : float
            Total session duration in seconds (used for rate calculations).

        Returns
        -------
        TurnAnalysisResult
        """
        if not segments:
            return TurnAnalysisResult()

        turns = self._merge_segments(segments)
        overlaps = self._detect_overlaps(turns)

        # Per-speaker aggregates.
        speaking_time: dict[str, float] = {}
        turn_count: dict[str, int] = {}
        turn_durations: dict[str, list[float]] = {}

        for turn in turns:
            spk = turn.speaker_id
            speaking_time[spk] = speaking_time.get(spk, 0.0) + turn.duration
            turn_count[spk] = turn_count.get(spk, 0) + 1
            turn_durations.setdefault(spk, []).append(turn.duration)

        avg_turn_duration: dict[str, float] = {
            spk: sum(durs) / len(durs) for spk, durs in turn_durations.items()
        }

        participation_cv = self._compute_participation_cv(speaking_time)

        total_turns = len(turns)
        duration_minutes = total_duration / 60.0
        turn_alternation_rate = (total_turns / duration_minutes) if duration_minutes > 0 else 0.0

        return TurnAnalysisResult(
            turns=turns,
            overlaps=overlaps,
            speaking_time=speaking_time,
            turn_count=turn_count,
            avg_turn_duration=avg_turn_duration,
            participation_cv=participation_cv,
            turn_alternation_rate=turn_alternation_rate,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _merge_segments(self, segments: list[SpeakerSegment]) -> list[SpeakerTurn]:
        """Merge consecutive same-speaker segments with gap < ``turn_min_gap``.

        Algorithm
        ---------
        1. Sort all segments by start time.
        2. Group by speaker_id.
        3. For each speaker, iterate through their segments in order and merge
           adjacent segments whose gap is less than ``turn_min_gap``.
        4. Re-sort the resulting turns by start time.
        """
        min_gap = self.config.turn_min_gap

        # Group segments by speaker.
        by_speaker: dict[str, list[SpeakerSegment]] = {}
        for seg in segments:
            by_speaker.setdefault(seg.speaker_id, []).append(seg)

        turns: list[SpeakerTurn] = []

        for speaker_id, spk_segs in by_speaker.items():
            # Sort this speaker's segments by start time.
            sorted_segs = sorted(spk_segs, key=lambda s: s.start)

            current_start = sorted_segs[0].start
            current_end = sorted_segs[0].end
            current_count = 1

            for seg in sorted_segs[1:]:
                gap = seg.start - current_end
                if gap < min_gap:
                    # Merge: extend the current turn.
                    current_end = max(current_end, seg.end)
                    current_count += 1
                else:
                    # Emit the completed turn and start a new one.
                    duration = current_end - current_start
                    turns.append(
                        SpeakerTurn(
                            start=current_start,
                            end=current_end,
                            speaker_id=speaker_id,
                            duration=duration,
                            segment_count=current_count,
                        )
                    )
                    current_start = seg.start
                    current_end = seg.end
                    current_count = 1

            # Emit the last accumulated turn.
            duration = current_end - current_start
            turns.append(
                SpeakerTurn(
                    start=current_start,
                    end=current_end,
                    speaker_id=speaker_id,
                    duration=duration,
                    segment_count=current_count,
                )
            )

        # Re-sort by start time across all speakers.
        turns.sort(key=lambda t: t.start)
        return turns

    def _detect_overlaps(
        self,
        turns: list[SpeakerTurn],
    ) -> list[tuple[float, float, str, str]]:
        """Find intervals where two different speakers overlap.

        Uses an O(n log n) sweep: after sorting by start, maintain an active
        set of turns that have not yet ended and check for overlaps with the
        new turn being processed.

        Returns
        -------
        list of (overlap_start, overlap_end, speaker_a, speaker_b)
        """
        if not turns:
            return []

        # Turns are already sorted by start from _merge_segments.
        overlaps: list[tuple[float, float, str, str]] = []

        # active: turns that have started but not yet ended.
        active: list[SpeakerTurn] = []

        for turn in turns:
            # Evict turns that have already ended before this one starts.
            active = [a for a in active if a.end > turn.start]

            for active_turn in active:
                if active_turn.speaker_id == turn.speaker_id:
                    continue
                # Overlap interval.
                overlap_start = max(active_turn.start, turn.start)
                overlap_end = min(active_turn.end, turn.end)
                if overlap_start < overlap_end:
                    overlaps.append(
                        (overlap_start, overlap_end, active_turn.speaker_id, turn.speaker_id)
                    )

            active.append(turn)

        return overlaps

    def _compute_participation_cv(self, speaking_time: dict[str, float]) -> float:
        """Compute coefficient of variation of speaking times.

        CV = std / mean.  Returns 0.0 when there is only one speaker or when
        all speakers have identical speaking times (std == 0).
        """
        times = list(speaking_time.values())
        if len(times) < 2:
            return 0.0
        mean = statistics.mean(times)
        if mean == 0.0:
            return 0.0
        std = statistics.stdev(times)
        return std / mean
