"""Interruption detection for the audio processing pipeline.

``InterruptionDetector`` classifies overlap events detected by ``TurnAnalyzer``
as DISRUPTIVE interruptions, cooperative overlaps, or back-channel responses —
all with pure Python logic, no ML model required.
"""

from __future__ import annotations

from app.core.audio.config import AudioConfig
from app.core.audio.data_types import (
    Interruption,
    InterruptionType,
    SpeakerTurn,
    TranscriptSegment,
)

# ---------------------------------------------------------------------------
# Back-channel phrase vocabulary (Spanish)
# ---------------------------------------------------------------------------

SPANISH_BACK_CHANNELS: frozenset[str] = frozenset(
    {
        "sí",
        "si",
        "claro",
        "ok",
        "okay",
        "ajá",
        "aja",
        "mhm",
        "mm",
        "hmm",
        "entiendo",
        "ya",
        "bueno",
        "bien",
        "exacto",
        "correcto",
        "verdad",
        "claro que sí",
        "por supuesto",
        "efectivamente",
    }
)


class InterruptionDetector:
    """Classify overlapping speech intervals into interruption types.

    Parameters
    ----------
    config : AudioConfig
        Pipeline-wide configuration.  Uses ``interruption_overlap_threshold``
        (minimum overlap duration to consider) and
        ``interruption_gap_threshold`` (maximum remaining duration of the
        interrupted turn for the interruption to be DISRUPTIVE).
    """

    def __init__(self, config: AudioConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        turns: list[SpeakerTurn],
        transcripts: list[TranscriptSegment],
        overlaps: list[tuple[float, float, str, str]],
    ) -> list[Interruption]:
        """Classify each overlap as DISRUPTIVE, BACK_CHANNEL, or COOPERATIVE.

        Parameters
        ----------
        turns : list[SpeakerTurn]
            Merged speaker turns produced by ``TurnAnalyzer``.
        transcripts : list[TranscriptSegment]
            Whisper transcripts aligned to speaker segments (may be empty).
        overlaps : list[tuple[float, float, str, str]]
            Overlap intervals from ``TurnAnalysisResult.overlaps``:
            ``(start, end, speaker_a, speaker_b)``.

        Returns
        -------
        list[Interruption]
            One entry per classified overlap (short overlaps are discarded).
        """
        results: list[Interruption] = []
        for overlap_start, overlap_end, speaker_a, speaker_b in overlaps:
            interruption = self._classify_overlap(
                overlap_start,
                overlap_end,
                speaker_a,
                speaker_b,
                turns,
                transcripts,
            )
            if interruption is not None:
                results.append(interruption)
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _classify_overlap(
        self,
        overlap_start: float,
        overlap_end: float,
        speaker_a: str,
        speaker_b: str,
        turns: list[SpeakerTurn],
        transcripts: list[TranscriptSegment],
    ) -> Interruption | None:
        """Classify a single overlap interval.

        Returns
        -------
        Interruption or None
            ``None`` when the overlap is shorter than the configured threshold
            and should be ignored as noise.
        """
        overlap_duration = overlap_end - overlap_start

        # Step 1 — discard noise-level overlaps.
        if overlap_duration < self.config.interruption_overlap_threshold:
            return None

        # Step 2 — identify interrupter vs interrupted speaker.
        turn_a = self._find_turn_at(turns, overlap_start, speaker_a)
        turn_b = self._find_turn_at(turns, overlap_start, speaker_b)

        if turn_a is None or turn_b is None:
            # Cannot determine roles without turn context — skip.
            return None

        # The speaker whose turn started earlier is "interrupted".
        if turn_a.start <= turn_b.start:
            interrupted_turn = turn_a
            interrupted_id = speaker_a
            interrupter_id = speaker_b
        else:
            interrupted_turn = turn_b
            interrupted_id = speaker_b
            interrupter_id = speaker_a

        # Step 3 — decide DISRUPTIVE vs COOPERATIVE.
        # "remaining" is how much of the interrupted speaker's turn was left
        # when the interrupter started.  If they had significant time remaining
        # (>= gap_threshold) the interrupter genuinely cut them off (DISRUPTIVE).
        # If the interrupted speaker was almost done (remaining < gap_threshold)
        # both speakers simply started speaking at the same natural boundary
        # (COOPERATIVE).
        remaining = interrupted_turn.end - overlap_start
        if remaining >= self.config.interruption_gap_threshold:
            interruption_type: InterruptionType = InterruptionType.DISRUPTIVE
        else:
            interruption_type = InterruptionType.COOPERATIVE

        # Step 4 — back-channel text filter (overrides DISRUPTIVE/COOPERATIVE).
        if transcripts:
            text = self._find_transcript_in_interval(
                transcripts, overlap_start, overlap_end, interrupter_id
            )
            if text and self._is_back_channel(text):
                interruption_type = InterruptionType.BACK_CHANNEL

        return Interruption(
            time=overlap_start,
            interrupter_id=interrupter_id,
            interrupted_id=interrupted_id,
            overlap_duration=overlap_duration,
            interruption_type=interruption_type,
        )

    def _is_back_channel(self, text: str) -> bool:
        """Return True if *text* is a cooperative back-channel response.

        A phrase is a back-channel when:
        - It is 2 words or fewer AND appears in ``SPANISH_BACK_CHANNELS``, or
        - It exactly matches a known multi-word back-channel phrase.
        """
        cleaned = text.strip().lower()
        # Multi-word phrase exact match first.
        if cleaned in SPANISH_BACK_CHANNELS:
            return True
        # Short utterance: at most 2 tokens.
        words = cleaned.split()
        if len(words) <= 2 and cleaned in SPANISH_BACK_CHANNELS:
            return True
        return False

    def _find_turn_at(
        self,
        turns: list[SpeakerTurn],
        timestamp: float,
        speaker_id: str,
    ) -> SpeakerTurn | None:
        """Return the turn for *speaker_id* that contains *timestamp*.

        If no turn contains the timestamp exactly, return the turn whose
        interval is nearest (by start time) to *timestamp*.
        """
        candidates = [t for t in turns if t.speaker_id == speaker_id]
        if not candidates:
            return None

        # Prefer a turn that strictly contains the timestamp.
        for turn in candidates:
            if turn.start <= timestamp <= turn.end:
                return turn

        # Fall back to the closest turn by start time.
        return min(candidates, key=lambda t: abs(t.start - timestamp))

    def _find_transcript_in_interval(
        self,
        transcripts: list[TranscriptSegment],
        start: float,
        end: float,
        speaker_id: str,
    ) -> str:
        """Collect transcript text for *speaker_id* within [*start*, *end*].

        Returns
        -------
        str
            Concatenated text of all matching segments, stripped.
        """
        parts: list[str] = []
        for seg in transcripts:
            if seg.speaker_id != speaker_id:
                continue
            # Include segments that overlap with [start, end].
            if seg.end <= start or seg.start >= end:
                continue
            parts.append(seg.text.strip())
        return " ".join(parts).strip()
