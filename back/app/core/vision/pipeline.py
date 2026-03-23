"""Vision pipeline orchestrator.

Manages the full lifecycle of video analysis: frame extraction, face
detection, per-person sub-processing (emotion, gaze, gesture, pose),
cross-frame person tracking, and result aggregation.
"""

from __future__ import annotations

import collections
import logging
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .config import VisionConfig
from .data_types import (
    FrameResult,
    PersonFrame,
    PersonMetrics,
    SessionMetrics,
    VisionResult,
)
from .face_detector import FaceDetector
from .frame_extractor import FrameExtractor
from .emotion_classifier import EmotionClassifier
from .gaze_estimator import GazeEstimator
from .gesture_analyzer import GestureAnalyzer
from .pose_estimator import PoseEstimator
from .person_tracker import PersonTracker

logger = logging.getLogger(__name__)

# Log progress every N frames
_LOG_INTERVAL = 25


class VisionPipeline:
    """Orchestrate all vision sub-processors on a single video.

    Parameters
    ----------
    config : VisionConfig, optional
        Pipeline-wide configuration.  Sub-processor enable flags and
        tuneable parameters are read from this object.
    """

    def __init__(self, config: Optional[VisionConfig] = None) -> None:
        self.config = config or VisionConfig()

        # Always-present processors
        self._frame_extractor = FrameExtractor(config=self.config, device=self.config.device)
        self._face_detector = FaceDetector(config=self.config, device=self.config.device)

        # Optional processors (created based on config flags)
        self._emotion_classifier: Optional[EmotionClassifier] = None
        self._gaze_estimator: Optional[GazeEstimator] = None
        self._gesture_analyzer: Optional[GestureAnalyzer] = None
        self._pose_estimator: Optional[PoseEstimator] = None
        self._person_tracker: Optional[PersonTracker] = None

        if self.config.enable_emotion:
            self._emotion_classifier = EmotionClassifier(config=self.config)
        if self.config.enable_gaze:
            self._gaze_estimator = GazeEstimator(config=self.config)
        if self.config.enable_gesture:
            self._gesture_analyzer = GestureAnalyzer(config=self.config)
        if self.config.enable_pose:
            self._pose_estimator = PoseEstimator(config=self.config)
        if self.config.enable_tracking:
            self._person_tracker = PersonTracker(config=self.config, device=self.config.device)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load_all(self) -> None:
        """Load all enabled processors."""
        logger.info("Loading vision pipeline processors...")
        self._frame_extractor.load()
        self._face_detector.load()

        if self._emotion_classifier is not None:
            self._emotion_classifier.load()
        if self._gaze_estimator is not None:
            self._gaze_estimator.load()
        if self._gesture_analyzer is not None:
            self._gesture_analyzer.load()
        if self._pose_estimator is not None:
            self._pose_estimator.load()
        if self._person_tracker is not None:
            self._person_tracker.load()

        logger.info("All vision pipeline processors loaded")

    def unload_all(self) -> None:
        """Unload all processors and free resources."""
        logger.info("Unloading vision pipeline processors...")

        for processor in [
            self._person_tracker,
            self._pose_estimator,
            self._gesture_analyzer,
            self._gaze_estimator,
            self._emotion_classifier,
            self._face_detector,
            self._frame_extractor,
        ]:
            if processor is not None and processor.is_loaded:
                try:
                    processor.unload()
                except Exception:
                    logger.exception("Error unloading %s", type(processor).__name__)

        logger.info("Vision pipeline processors unloaded")

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "VisionPipeline":
        self.load_all()
        return self

    def __exit__(self, *args: Any) -> None:
        self.unload_all()

    # ------------------------------------------------------------------
    # Main processing
    # ------------------------------------------------------------------

    def process_video(
        self,
        video_path: str,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> VisionResult:
        """Run the full vision pipeline on *video_path*.

        Parameters
        ----------
        video_path : str
            Path to the video file on disk.

        Returns
        -------
        VisionResult
            Aggregated results across all sampled frames.

        Raises
        ------
        FileNotFoundError
            If *video_path* does not exist.
        RuntimeError
            If the video cannot be opened or processors are not loaded.
        """
        t_start = time.time()

        # 1. Get video metadata
        total_frames, duration, native_fps = self._frame_extractor.get_video_properties(video_path)
        logger.info(
            "Processing video: %s (%.1fs, %d frames, %.1f fps)",
            video_path, duration, total_frames, native_fps,
        )

        # 2. Extract frames
        raw_frames = self._frame_extractor.process(video_path)
        logger.info("Extracted %d frames for processing", len(raw_frames))

        if progress_callback:
            progress_callback(0.10, "processing_frames")

        # 3. Process each frame
        frame_results: List[FrameResult] = []

        num_raw = len(raw_frames) if raw_frames else 1  # avoid division by zero

        for idx, (frame_number, timestamp, frame) in enumerate(raw_frames):
            # Log progress
            if idx % _LOG_INTERVAL == 0:
                logger.info("Processing frame %d / %d (%.0f%%)", idx + 1, len(raw_frames),
                            (idx + 1) / len(raw_frames) * 100)

            # Report progress (10% - 90% range for frame processing)
            if progress_callback and idx % _LOG_INTERVAL == 0:
                pct = 0.10 + 0.80 * (idx / num_raw)
                progress_callback(pct, "processing_frames")

            # 3a. Detect faces
            persons: List[PersonFrame] = self._face_detector.process(frame)

            if not persons:
                frame_results.append(FrameResult(
                    frame_number=frame_number,
                    timestamp=timestamp,
                    persons=[],
                ))
                continue

            # 3b. Track persons across frames (assigns person_id)
            if self._person_tracker is not None and self._person_tracker.is_loaded:
                persons = self._person_tracker.process(frame, persons)

            # 3c. Per-person sub-processing

            # Batch emotion classification when multiple faces detected
            if (self._emotion_classifier is not None
                    and self._emotion_classifier.is_loaded
                    and len(persons) > 1):
                face_crops = [
                    self._extract_face_crop(frame, p.bbox) for p in persons
                ]
                # Replace None crops with empty arrays for batch processing
                safe_crops = [
                    c if c is not None else np.empty((0, 0, 3), dtype=np.uint8)
                    for c in face_crops
                ]
                emotions = self._emotion_classifier.process_batch(safe_crops)
                for person, emotion in zip(persons, emotions):
                    person.emotion = emotion
            elif self._emotion_classifier is not None and self._emotion_classifier.is_loaded:
                # Single face — use original process method
                for person in persons:
                    face_crop = self._extract_face_crop(frame, person.bbox)
                    if face_crop is not None:
                        person.emotion = self._emotion_classifier.process(face_crop)

            for person in persons:
                # Gaze estimation (needs landmarks)
                if (self._gaze_estimator is not None
                        and self._gaze_estimator.is_loaded
                        and person.landmarks is not None):
                    h, w = frame.shape[:2]
                    person.gaze = self._gaze_estimator.process(person.landmarks, (h, w))

                # Gesture analysis (needs landmarks + person_id)
                if (self._gesture_analyzer is not None
                        and self._gesture_analyzer.is_loaded
                        and person.landmarks is not None):
                    person.gesture = self._gesture_analyzer.process(
                        person.person_id, person.landmarks
                    )

                # Pose estimation (needs frame + bbox)
                if self._pose_estimator is not None and self._pose_estimator.is_loaded:
                    person.pose = self._pose_estimator.process(frame, person.bbox)

            frame_results.append(FrameResult(
                frame_number=frame_number,
                timestamp=timestamp,
                persons=persons,
            ))

        t_elapsed = time.time() - t_start
        logger.info(
            "Video processing complete: %d frames in %.1fs (%.1f frames/s)",
            len(frame_results), t_elapsed,
            len(frame_results) / t_elapsed if t_elapsed > 0 else 0,
        )

        if progress_callback:
            progress_callback(0.90, "computing_metrics")

        # 4. Collect person embeddings from tracker
        person_embeddings: Dict[str, Any] = {}
        if self._person_tracker is not None and self._person_tracker.is_loaded:
            person_embeddings = dict(self._person_tracker.known_embeddings)

        # 5. Compute per-person aggregation metrics
        session_metrics = self._compute_session_metrics(
            frame_results, duration,
        )

        return VisionResult(
            video_path=video_path,
            total_frames=total_frames,
            fps_processed=self.config.fps,
            duration_seconds=duration,
            frames=frame_results,
            processing_time_seconds=round(t_elapsed, 3),
            person_embeddings=person_embeddings,
            session_metrics=session_metrics,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_session_metrics(
        frame_results: List[FrameResult],
        duration: float,
    ) -> SessionMetrics:
        """Compute per-person and session-level aggregation metrics.

        Parameters
        ----------
        frame_results : list of FrameResult
            All processed frames.
        duration : float
            Total video duration in seconds.

        Returns
        -------
        SessionMetrics
        """
        # Collect per-person data across all frames
        person_frames_seen: Dict[str, int] = collections.Counter()
        person_gaze_contact: Dict[str, int] = collections.Counter()
        person_emotions: Dict[str, List[str]] = collections.defaultdict(list)
        person_emotion_scores: Dict[str, Dict[str, float]] = collections.defaultdict(
            lambda: collections.defaultdict(float)
        )
        person_orientations: Dict[str, List[float]] = collections.defaultdict(list)
        person_gestures: Dict[str, Dict[str, int]] = collections.defaultdict(
            lambda: collections.defaultdict(int)
        )

        for fr in frame_results:
            for person in fr.persons:
                pid = person.person_id
                person_frames_seen[pid] += 1

                if person.gaze is not None and person.gaze.is_looking_at_camera:
                    person_gaze_contact[pid] += 1

                if person.emotion is not None:
                    person_emotions[pid].append(person.emotion.primary_emotion)
                    for emo, score in person.emotion.all_emotions.items():
                        person_emotion_scores[pid][emo] += score

                if person.pose is not None:
                    person_orientations[pid].append(person.pose.body_orientation)

                if person.gesture is not None and person.gesture.gesture_type != "neutral":
                    person_gestures[pid][person.gesture.gesture_type] += 1

        # Build per-person metrics
        per_person: List[PersonMetrics] = []
        for pid in sorted(person_frames_seen.keys()):
            total_seen = person_frames_seen[pid]

            # Gaze contact percentage
            gaze_pct = (person_gaze_contact[pid] / total_seen * 100.0) if total_seen > 0 else 0.0

            # Dominant emotion and distribution
            emotion_list = person_emotions.get(pid, [])
            if emotion_list:
                emotion_counter = collections.Counter(emotion_list)
                dominant_emotion = emotion_counter.most_common(1)[0][0]
                total_emotion_frames = len(emotion_list)
                emotion_distribution = {
                    emo: count / total_emotion_frames
                    for emo, count in emotion_counter.items()
                }
            else:
                dominant_emotion = "neutral"
                emotion_distribution = {}

            # Average body orientation
            orientations = person_orientations.get(pid, [])
            avg_orientation = float(np.mean(orientations)) if orientations else 0.0

            # Gesture counts
            gestures = dict(person_gestures.get(pid, {}))

            # Attention score: weighted combination
            # 40% gaze contact + 30% facing camera (low orientation) + 20% nod gestures + 10% emotion engagement
            gaze_score = gaze_pct / 100.0
            orientation_score = max(0.0, 1.0 - avg_orientation / 90.0) if orientations else 0.0
            nod_score = min(1.0, gestures.get("nod", 0) / max(1, total_seen) * 10.0)
            emotion_score = 1.0 - emotion_distribution.get("neutral", 1.0)
            attention = (
                0.40 * gaze_score
                + 0.30 * orientation_score
                + 0.20 * nod_score
                + 0.10 * emotion_score
            )

            per_person.append(PersonMetrics(
                person_id=pid,
                total_frames_seen=total_seen,
                gaze_contact_percentage=round(gaze_pct, 2),
                dominant_emotion=dominant_emotion,
                emotion_distribution=emotion_distribution,
                average_body_orientation=round(avg_orientation, 2),
                gesture_counts=gestures,
                attention_score=round(min(1.0, max(0.0, attention)), 4),
            ))

        return SessionMetrics(
            total_persons=len(per_person),
            total_frames=len(frame_results),
            duration=duration,
            per_person_metrics=per_person,
        )

    @staticmethod
    def _extract_face_crop(
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> Optional[np.ndarray]:
        """Extract and return the face crop from *frame* using *bbox*.

        Returns ``None`` if the crop would be empty.
        """
        x, y, w, h = bbox
        fh, fw = frame.shape[:2]

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(fw, x + w)
        y2 = min(fh, y + h)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        return crop
