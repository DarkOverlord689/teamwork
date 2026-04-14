"""Speaker-to-face mapping for smart frame selection.

Associates diarised speaker IDs with tracked person/face IDs by voting over
frames where both audio and video data are available.
"""

from __future__ import annotations

from collections import defaultdict

from app.core.audio.data_types import SpeakerSegment
from app.core.vision.data_types import SpeakerFaceMapping


class SpeakerFaceMapper:
    """Map diarised speaker IDs to tracked face person IDs.

    Parameters
    ----------
    min_confidence : float
        Minimum vote-fraction to consider a mapping confident.
    min_frames : int
        Minimum number of co-occurring frames required for a confident mapping.
    """

    def __init__(self, min_confidence: float = 0.5, min_frames: int = 3) -> None:
        self.min_confidence = min_confidence
        self.min_frames = min_frames

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_mapping(
        self,
        speaker_segments: list[SpeakerSegment],
        frame_person_data: list[dict],
    ) -> list[SpeakerFaceMapping]:
        """Build speaker-to-face mappings using a voting algorithm.

        Parameters
        ----------
        speaker_segments : list[SpeakerSegment]
            Diarised speaker segments from the audio pipeline.
        frame_person_data : list[dict]
            Per-frame person detections.  Each element must contain:
            ``{"timestamp_s": float, "person_ids": list[str]}``.

        Returns
        -------
        list[SpeakerFaceMapping]
            One mapping entry per speaker.  Entries are sorted by speaker_id.

        Algorithm
        ---------
        1. For each frame, find the active speaker at that timestamp.
        2. Build a vote matrix: ``vote_matrix[speaker_id][person_id] += 1``.
        3. Greedy assignment: for each speaker (sorted by max votes descending),
           assign to the person_id with the most votes not yet assigned.
        4. Compute ``confidence = assigned_votes / total_votes_for_speaker``.
        5. Mark ``uncertain=True`` if confidence < min_confidence or
           frame_count < min_frames.
        """
        # Step 1 & 2: build vote matrix
        vote_matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for frame in frame_person_data:
            timestamp_s: float = frame["timestamp_s"]
            person_ids: list[str] = frame.get("person_ids", [])
            speaker_id = self._find_active_speaker(timestamp_s, speaker_segments)
            if speaker_id is None:
                continue
            for person_id in person_ids:
                vote_matrix[speaker_id][person_id] += 1

        # Step 3: greedy assignment
        # Sort speakers by their maximum vote count (descending) so the most
        # confident pairings are resolved first.
        def max_votes(speaker: str) -> int:
            votes = vote_matrix.get(speaker, {})
            return max(votes.values()) if votes else 0

        speakers_sorted = sorted(vote_matrix.keys(), key=max_votes, reverse=True)
        assigned_persons: set[str] = set()
        mappings: list[SpeakerFaceMapping] = []

        for speaker_id in speakers_sorted:
            person_votes = vote_matrix[speaker_id]
            total_votes = sum(person_votes.values())

            # Find best unassigned person
            best_person_id: str | None = None
            best_votes: int = 0
            for person_id, votes in sorted(
                person_votes.items(), key=lambda kv: kv[1], reverse=True
            ):
                if person_id not in assigned_persons:
                    best_person_id = person_id
                    best_votes = votes
                    break

            if best_person_id is None:
                # No unassigned person available; mark as uncertain with empty person
                mappings.append(
                    SpeakerFaceMapping(
                        speaker_id=speaker_id,
                        person_id="",
                        confidence=0.0,
                        frame_count=0,
                        uncertain=True,
                    )
                )
                continue

            assigned_persons.add(best_person_id)
            confidence = best_votes / total_votes if total_votes > 0 else 0.0
            uncertain = confidence < self.min_confidence or best_votes < self.min_frames

            mappings.append(
                SpeakerFaceMapping(
                    speaker_id=speaker_id,
                    person_id=best_person_id,
                    confidence=confidence,
                    frame_count=best_votes,
                    uncertain=uncertain,
                )
            )

        # Include speakers that appear in segments but had no overlapping frames
        seen_speakers = {m.speaker_id for m in mappings}
        for seg in speaker_segments:
            if seg.speaker_id not in seen_speakers:
                seen_speakers.add(seg.speaker_id)
                mappings.append(
                    SpeakerFaceMapping(
                        speaker_id=seg.speaker_id,
                        person_id="",
                        confidence=0.0,
                        frame_count=0,
                        uncertain=True,
                    )
                )

        mappings.sort(key=lambda m: m.speaker_id)
        return mappings

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_active_speaker(
        self, timestamp_s: float, segments: list[SpeakerSegment]
    ) -> str | None:
        """Return the speaker_id of the segment containing *timestamp_s*, or None."""
        for seg in segments:
            if seg.start <= timestamp_s <= seg.end:
                return seg.speaker_id
        return None
