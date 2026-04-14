"""Key frame selector for smart frame selection.

Detects key conversational moments from audio pipeline outputs and returns
a budget-constrained list of KeyMoment objects for downstream frame capture.
"""

from __future__ import annotations

from collections import defaultdict

from app.core.audio.config import AudioConfig
from app.core.audio.data_types import (
    KeyMoment,
    MomentType,
    SpeakerSegment,
    SpeakerTurn,
    TranscriptSegment,
)


class KeyFrameSelector:
    """Select key moments from audio pipeline outputs for frame capture.

    Parameters
    ----------
    config : AudioConfig
        Pipeline configuration.  ``sfs_*`` fields control detection thresholds
        and budget limits.
    """

    def __init__(self, config: AudioConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select(
        self,
        segments: list[SpeakerSegment],
        transcript: list[TranscriptSegment],
        interruptions: list,
        turns: list,
    ) -> list[KeyMoment]:
        """Detect all moment types, merge, apply budget, and return sorted list.

        Parameters
        ----------
        segments : list[SpeakerSegment]
            Diarised speaker segments.
        transcript : list[TranscriptSegment]
            Whisper-aligned transcript segments.
        interruptions : list
            Interruption objects from the interruption detector.
        turns : list
            Turn objects from the turn analyser.

        Returns
        -------
        list[KeyMoment]
            Budget-constrained moments sorted by timestamp.
        """
        all_moments: list[KeyMoment] = []
        all_moments.extend(self._detect_speech_onset(segments, transcript))
        all_moments.extend(self._detect_receiving_question(segments, transcript))
        all_moments.extend(self._detect_hesitation(segments, transcript))
        all_moments.extend(self._detect_interruption_received(interruptions))
        all_moments.extend(self._detect_post_question_silence(segments, transcript))
        all_moments.extend(self._detect_end_of_long_turn(turns))
        all_moments.extend(self._detect_back_channel(interruptions))

        budgeted = self._apply_budget(all_moments)
        budgeted.sort(key=lambda m: m.timestamp_ms)
        return budgeted

    # ------------------------------------------------------------------
    # Detectors
    # ------------------------------------------------------------------

    def _detect_speech_onset(
        self,
        segments: list[SpeakerSegment],
        transcript: list[TranscriptSegment],
    ) -> list[KeyMoment]:
        moments: list[KeyMoment] = []
        for i, seg in enumerate(segments):
            is_first = i == 0
            speaker_changed = not is_first and segments[i - 1].speaker_id != seg.speaker_id
            if is_first or speaker_changed:
                timestamp_ms = seg.start * 1000.0 + self.config.sfs_speech_onset_offset_ms
                context_text = ""
                for t in transcript:
                    if t.speaker_id == seg.speaker_id and abs(t.start - seg.start) < 1.0:
                        context_text = t.text[:50]
                        break
                moments.append(
                    KeyMoment(
                        timestamp_ms=timestamp_ms,
                        speaker_id=seg.speaker_id,
                        moment_type=MomentType.SPEECH_ONSET,
                        priority=1,
                        context_text=context_text,
                    )
                )
        return moments

    def _detect_receiving_question(
        self,
        segments: list[SpeakerSegment],
        transcript: list[TranscriptSegment],
    ) -> list[KeyMoment]:
        moments: list[KeyMoment] = []
        for i, seg in enumerate(transcript):
            if "?" not in seg.text:
                continue
            # Find next transcript segment with a different speaker
            for j in range(i + 1, len(transcript)):
                next_seg = transcript[j]
                if next_seg.speaker_id != seg.speaker_id:
                    moments.append(
                        KeyMoment(
                            timestamp_ms=next_seg.start * 1000.0,
                            speaker_id=next_seg.speaker_id,
                            moment_type=MomentType.RECEIVING_QUESTION,
                            priority=1,
                            context_text=seg.text,
                        )
                    )
                    break
        return moments

    def _detect_hesitation(
        self,
        segments: list[SpeakerSegment],
        transcript: list[TranscriptSegment],
    ) -> list[KeyMoment]:
        moments: list[KeyMoment] = []
        fillers = set(self.config.sfs_hesitation_fillers)
        pause_threshold_ms = self.config.sfs_hesitation_pause_threshold_ms

        for seg in transcript:
            text_lower = seg.text.lower()
            words = text_lower.split()
            for word in words:
                # Strip punctuation for matching
                clean = word.strip(".,;:!?")
                if clean in fillers:
                    moments.append(
                        KeyMoment(
                            timestamp_ms=seg.start * 1000.0,
                            speaker_id=seg.speaker_id,
                            moment_type=MomentType.HESITATION,
                            priority=2,
                            context_text=seg.text[:50],
                        )
                    )
                    break  # one moment per segment is enough

        # Intra-segment pauses between consecutive same-speaker segments
        for i in range(len(transcript) - 1):
            curr = transcript[i]
            nxt = transcript[i + 1]
            if curr.speaker_id != nxt.speaker_id:
                continue
            gap_ms = (nxt.start - curr.end) * 1000.0
            if gap_ms > pause_threshold_ms:
                midpoint_ms = (curr.end + nxt.start) / 2.0 * 1000.0
                moments.append(
                    KeyMoment(
                        timestamp_ms=midpoint_ms,
                        speaker_id=curr.speaker_id,
                        moment_type=MomentType.HESITATION,
                        priority=2,
                        context_text=curr.text[:50],
                    )
                )

        return moments

    def _detect_interruption_received(self, interruptions: list) -> list[KeyMoment]:
        moments: list[KeyMoment] = []
        for interruption in interruptions:
            itype = getattr(interruption, "interruption_type", None)
            if itype not in ("disruptive", "overlapping"):
                # Also support enum values
                itype_str = str(itype).lower() if itype else ""
                if "disruptive" not in itype_str and "overlapping" not in itype_str:
                    continue
            interrupted_id = getattr(interruption, "interrupted_id", None)
            time = getattr(interruption, "time", 0.0)
            if interrupted_id is None:
                continue
            moments.append(
                KeyMoment(
                    timestamp_ms=time * 1000.0,
                    speaker_id=interrupted_id,
                    moment_type=MomentType.INTERRUPTION_RECEIVED,
                    priority=1,
                    context_text="",
                )
            )
        return moments

    def _detect_post_question_silence(
        self,
        segments: list[SpeakerSegment],
        transcript: list[TranscriptSegment],
    ) -> list[KeyMoment]:
        moments: list[KeyMoment] = []
        min_gap_ms = self.config.sfs_post_silence_min_gap_ms

        for i, seg in enumerate(transcript):
            if "?" not in seg.text:
                continue
            # Find the next transcript segment (any speaker)
            if i + 1 >= len(transcript):
                continue
            next_seg = transcript[i + 1]
            gap_ms = (next_seg.start - seg.end) * 1000.0
            if gap_ms > min_gap_ms:
                moments.append(
                    KeyMoment(
                        timestamp_ms=seg.end * 1000.0,
                        speaker_id=seg.speaker_id,
                        moment_type=MomentType.POST_QUESTION_SILENCE,
                        priority=2,
                        context_text=seg.text,
                    )
                )
        return moments

    def _detect_end_of_long_turn(self, turns: list) -> list[KeyMoment]:
        moments: list[KeyMoment] = []
        threshold_s = self.config.sfs_long_turn_threshold_s

        for turn in turns:
            duration = getattr(turn, "duration", None)
            if duration is None:
                end = getattr(turn, "end", 0.0)
                start = getattr(turn, "start", 0.0)
                duration = end - start
            if duration > threshold_s:
                end_s = getattr(turn, "end", 0.0)
                speaker_id = getattr(turn, "speaker_id", "")
                moments.append(
                    KeyMoment(
                        timestamp_ms=end_s * 1000.0,
                        speaker_id=speaker_id,
                        moment_type=MomentType.END_OF_LONG_TURN,
                        priority=3,
                        context_text="",
                    )
                )
        return moments

    def _detect_back_channel(self, interruptions: list) -> list[KeyMoment]:
        moments: list[KeyMoment] = []
        for interruption in interruptions:
            itype = getattr(interruption, "interruption_type", None)
            itype_str = str(itype).lower() if itype else ""
            if "back_channel" not in itype_str:
                continue
            interrupter_id = getattr(interruption, "interrupter_id", None)
            time = getattr(interruption, "time", 0.0)
            if interrupter_id is None:
                continue
            moments.append(
                KeyMoment(
                    timestamp_ms=time * 1000.0,
                    speaker_id=interrupter_id,
                    moment_type=MomentType.BACK_CHANNEL,
                    priority=3,
                    context_text="",
                )
            )
        return moments

    # ------------------------------------------------------------------
    # Budget enforcement
    # ------------------------------------------------------------------

    def _apply_budget(self, moments: list[KeyMoment]) -> list[KeyMoment]:
        """Enforce per-participant and total frame budgets.

        Sorts by priority (ascending = highest priority first), then greedily
        selects moments while respecting ``sfs_max_frames_per_participant`` and
        ``sfs_max_frames_total``.
        """
        max_per = self.config.sfs_max_frames_per_participant
        max_total = self.config.sfs_max_frames_total

        sorted_moments = sorted(moments, key=lambda m: m.priority)
        per_participant: dict[str, int] = defaultdict(int)
        selected: list[KeyMoment] = []

        for moment in sorted_moments:
            if len(selected) >= max_total:
                break
            if per_participant[moment.speaker_id] >= max_per:
                continue
            selected.append(moment)
            per_participant[moment.speaker_id] += 1

        return selected
