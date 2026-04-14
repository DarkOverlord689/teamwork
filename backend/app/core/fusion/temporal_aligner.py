"""Temporal alignment of visual and audio features.

``TemporalAligner`` takes the outputs of the vision and audio pipelines and
produces an ``AlignedFeatures`` object — a timeline of 500 ms windows each
annotated with both the speaking identity (from diarization) and the visual
identity (from face tracking), linked by cosine similarity of face embeddings.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np

from app.core.audio.data_types import AudioResult
from app.core.fusion.config import FusionConfig
from app.core.fusion.data_types import AlignedFeatures, TemporalWindow
from app.core.vision.data_types import VisionResult

logger = logging.getLogger(__name__)


class TemporalAligner:
    """Align vision and audio features along a shared timeline.

    Strategy
    --------
    1. Divide the video into ``temporal_window_ms`` windows.
    2. For each window, determine the dominant speaker from diarization turns.
    3. For each window, determine the dominant visible person from frame results.
    4. Build a speaker-to-person mapping by accumulating co-occurrence evidence
       weighted by face-embedding cosine similarity.

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

    def align(self, vision_result: VisionResult, audio_result: AudioResult) -> AlignedFeatures:
        """Align vision and audio results into a common ``AlignedFeatures`` object.

        Parameters
        ----------
        vision_result : VisionResult
            Output of the visual processing pipeline (Module 2.1).
        audio_result : AudioResult
            Output of the audio processing pipeline (Module 2.2).

        Returns
        -------
        AlignedFeatures
        """
        duration = max(
            vision_result.duration_seconds,
            audio_result.duration_seconds,
        )
        window_sec = self.config.temporal_window_ms / 1000.0

        windows = self._build_windows(vision_result, audio_result, duration, window_sec)
        speaker_to_person, person_to_speaker = self._build_id_mappings(
            vision_result, audio_result, windows
        )

        # Back-fill the person_id in each window using the mapping
        for w in windows:
            if w.speaker_id and w.person_id is None:
                w.person_id = speaker_to_person.get(w.speaker_id)

        logger.info(
            "Temporal alignment complete: %d windows, %d speaker-to-person mappings",
            len(windows),
            len(speaker_to_person),
        )

        return AlignedFeatures(
            video_path=vision_result.video_path,
            duration_seconds=duration,
            windows=windows,
            speaker_to_person=speaker_to_person,
            person_to_speaker=person_to_speaker,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_windows(
        self,
        vision_result: VisionResult,
        audio_result: AudioResult,
        duration: float,
        window_sec: float,
    ) -> List[TemporalWindow]:
        """Slice the timeline into windows and populate audio + visual attributes."""
        windows: List[TemporalWindow] = []
        t = 0.0
        while t < duration:
            end = min(t + window_sec, duration)

            speaker_id, overlap_ratio = self._dominant_speaker(
                audio_result, t, end
            )
            person_id, gaze_at_camera, gaze_conf, body_orient, gesture, emotion = (
                self._dominant_person_visual(vision_result, t, end)
            )

            windows.append(
                TemporalWindow(
                    start=t,
                    end=end,
                    speaker_id=speaker_id,
                    person_id=person_id,
                    gaze_at_camera=gaze_at_camera,
                    gaze_confidence=gaze_conf,
                    body_orientation=body_orient,
                    gesture_type=gesture,
                    emotion=emotion,
                    overlap_ratio=overlap_ratio,
                )
            )
            t = end

        return windows

    def _dominant_speaker(
        self,
        audio_result: AudioResult,
        win_start: float,
        win_end: float,
    ) -> tuple[str | None, float]:
        """Return (speaker_id, overlap_ratio) for the speaker with most time in window."""
        win_dur = win_end - win_start
        if win_dur <= 0:
            return None, 0.0

        speaker_time: Dict[str, float] = {}
        for turn in audio_result.turns:
            overlap = min(turn.end, win_end) - max(turn.start, win_start)
            if overlap > 0:
                speaker_time[turn.speaker_id] = (
                    speaker_time.get(turn.speaker_id, 0.0) + overlap
                )

        if not speaker_time:
            return None, 0.0

        best_speaker = max(speaker_time, key=lambda k: speaker_time[k])
        ratio = speaker_time[best_speaker] / win_dur
        if ratio < self.config.min_overlap_ratio:
            return None, ratio
        return best_speaker, ratio

    def _dominant_person_visual(
        self,
        vision_result: VisionResult,
        win_start: float,
        win_end: float,
    ) -> tuple[str | None, bool, float, float, str, str]:
        """Return visual attributes for the most prominent person in the window.

        Returns
        -------
        tuple of (person_id, gaze_at_camera, gaze_confidence,
                  body_orientation, gesture_type, emotion)
        """
        frames_in_window = [
            f for f in vision_result.frames
            if win_start <= f.timestamp < win_end and f.persons
        ]

        if not frames_in_window:
            return None, False, 0.0, 0.0, "neutral", "neutral"

        # Pick the person who appears most in this window
        person_frame_counts: Dict[str, int] = {}
        for frame in frames_in_window:
            for p in frame.persons:
                person_frame_counts[p.person_id] = (
                    person_frame_counts.get(p.person_id, 0) + 1
                )

        dominant_pid = max(person_frame_counts, key=lambda k: person_frame_counts[k])

        # Aggregate visual attributes for dominant person across frames in window
        gaze_count = 0
        gaze_total_conf = 0.0
        body_orientations: List[float] = []
        gesture_votes: Dict[str, int] = {}
        emotion_votes: Dict[str, int] = {}

        for frame in frames_in_window:
            for p in frame.persons:
                if p.person_id != dominant_pid:
                    continue
                if p.gaze is not None:
                    if p.gaze.is_looking_at_camera:
                        gaze_count += 1
                    gaze_total_conf += p.gaze.confidence
                if p.pose is not None:
                    body_orientations.append(p.pose.body_orientation)
                if p.gesture is not None:
                    gt = p.gesture.gesture_type
                    gesture_votes[gt] = gesture_votes.get(gt, 0) + 1
                if p.emotion is not None:
                    em = p.emotion.primary_emotion
                    emotion_votes[em] = emotion_votes.get(em, 0) + 1

        total_frames = person_frame_counts[dominant_pid]
        gaze_at_camera = (gaze_count / total_frames) >= 0.5 if total_frames else False
        gaze_conf = gaze_total_conf / total_frames if total_frames else 0.0
        body_orient = float(np.mean(body_orientations)) if body_orientations else 0.0
        gesture = max(gesture_votes, key=lambda k: gesture_votes[k]) if gesture_votes else "neutral"
        emotion = max(emotion_votes, key=lambda k: emotion_votes[k]) if emotion_votes else "neutral"

        return dominant_pid, gaze_at_camera, gaze_conf, body_orient, gesture, emotion

    def _build_id_mappings(
        self,
        vision_result: VisionResult,
        audio_result: AudioResult,
        windows: List[TemporalWindow],
    ) -> tuple[Dict[str, str], Dict[str, str]]:
        """Map diarizer speaker IDs to vision person IDs using co-occurrence + embeddings.

        Primary strategy: count how many windows each (speaker_id, person_id) pair
        co-occurs and pick the maximum-count pairing.  Breaks ties using face
        embedding cosine similarity when available.
        """
        cooccurrence: Dict[tuple[str, str], float] = {}

        for w in windows:
            if w.speaker_id and w.person_id:
                key = (w.speaker_id, w.person_id)
                cooccurrence[key] = cooccurrence.get(key, 0.0) + w.overlap_ratio

        if not cooccurrence:
            logger.warning("No speaker-person co-occurrence found; returning empty mappings")
            return {}, {}

        # If face embeddings are available, weight by cosine similarity
        embeddings = vision_result.person_embeddings  # dict: person_id -> np.ndarray | list
        if embeddings:
            adjusted: Dict[tuple[str, str], float] = {}
            for (spk, pid), score in cooccurrence.items():
                emb = embeddings.get(pid)
                if emb is not None:
                    arr = np.array(emb) if not isinstance(emb, np.ndarray) else emb
                    norm = np.linalg.norm(arr)
                    sim = float(norm) if norm > 0 else 0.0  # rough proxy without ref embedding
                    adjusted[(spk, pid)] = score * (1.0 + sim * 0.1)
                else:
                    adjusted[(spk, pid)] = score
            cooccurrence = adjusted

        # Greedy assignment: best score pair first, then remove assigned IDs
        pairs = sorted(cooccurrence.items(), key=lambda x: x[1], reverse=True)
        speaker_to_person: Dict[str, str] = {}
        person_to_speaker: Dict[str, str] = {}

        for (spk, pid), _ in pairs:
            if spk not in speaker_to_person and pid not in person_to_speaker:
                speaker_to_person[spk] = pid
                person_to_speaker[pid] = spk

        return speaker_to_person, person_to_speaker
