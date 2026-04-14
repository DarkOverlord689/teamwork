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
    MultimodalFrameAnalysis,
    PersonFrame,
    PersonMetrics,
    SessionMetrics,
    SpeakerFaceMapping,
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

        # Smart frame selection: multimodal analyzer (optional)
        self._multimodal_analyzer: Optional[Any] = None
        if getattr(self.config, "use_multimodal", False):
            from .multimodal_analyzer import MultimodalAnalyzer
            self._multimodal_analyzer = MultimodalAnalyzer(config=self.config)

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
        session_id: Optional[str] = None,
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

        # Set up thumbnail saver when a session_id is provided
        thumbnail_saver = None
        if session_id is not None:
            from app.utils.config import settings as _settings
            from .frame_saver import FrameThumbnailSaver
            thumbnail_saver = FrameThumbnailSaver(
                frames_dir=_settings.frames_dir,
                session_id=session_id,
            )

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
                # Still save a thumbnail even when no persons were detected
                if thumbnail_saver is not None:
                    thumbnail_saver.maybe_save(timestamp, frame)
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

            # Save thumbnail (best-effort; errors are logged, not raised)
            if thumbnail_saver is not None:
                thumbnail_saver.maybe_save(timestamp, frame)

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
            frame_thumbnails=thumbnail_saver.thumbnails if thumbnail_saver is not None else {},
        )

    # ------------------------------------------------------------------
    # Smart frame selection pipeline
    # ------------------------------------------------------------------

    def process_video_smart(
        self,
        video_path: str,
        key_moments: list,
        speaker_segments: list,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        session_id: Optional[str] = None,
    ) -> VisionResult:
        """Run the smart vision pipeline using audio-guided key moments.

        Parameters
        ----------
        video_path : str
            Path to the video file on disk.
        key_moments : list[KeyMoment]
            Key moments selected by KeyFrameSelector; each has ``timestamp_ms``
            and ``moment_type`` attributes.
        speaker_segments : list[SpeakerSegment]
            Diarisation segments from the audio pipeline, used to build
            speaker-to-face mappings.
        progress_callback : callable, optional
            ``(fraction, stage_name)`` progress reporter.

        Returns
        -------
        VisionResult
            Standard VisionResult with ``smart_mode=True``, populated
            ``key_moment_analyses``, and ``speaker_face_mappings``.
        """
        import time as _time

        t_start = _time.time()

        if not key_moments:
            logger.warning("process_video_smart called with empty key_moments list.")
            return VisionResult(
                video_path=video_path,
                total_frames=0,
                fps_processed=0.0,
                duration_seconds=0.0,
                smart_mode=True,
            )

        # 1. Extract frames at key moment timestamps
        # key_moments carry timestamp_ms (milliseconds); convert to seconds
        # because extract_at_timestamps expects seconds.
        timestamps_s = [km.timestamp_ms / 1000.0 for km in key_moments]
        logger.info(
            "Smart pipeline: extracting %d targeted frames from %s",
            len(timestamps_s), video_path,
        )

        if progress_callback:
            progress_callback(0.05, "extracting_targeted_frames")

        extracted = self._frame_extractor.extract_at_timestamps(video_path, timestamps_s)
        logger.info("Smart pipeline: extracted %d frames", len(extracted))

        if progress_callback:
            progress_callback(0.15, "processing_frames")

        # Set up thumbnail saver for smart pipeline
        smart_thumbnail_saver = None
        if session_id is not None:
            from app.utils.config import settings as _settings
            from .frame_saver import FrameThumbnailSaver
            smart_thumbnail_saver = FrameThumbnailSaver(
                frames_dir=_settings.frames_dir,
                session_id=session_id,
            )

        # 2. Process each frame: detect faces, track persons
        frame_results: List[FrameResult] = []
        # Map timestamp_ms → key_moment for context lookup
        ts_to_km: Dict[int, Any] = {int(km.timestamp_ms): km for km in key_moments}

        # extract_at_timestamps returns (frame_number, timestamp_s, rgb_frame) 3-tuples
        for idx, (frame_num, timestamp_s, frame) in enumerate(extracted):
            persons: List[PersonFrame] = self._face_detector.process(frame)

            if persons and self._person_tracker is not None and self._person_tracker.is_loaded:
                persons = self._person_tracker.process(frame, persons)

            if persons and self._emotion_classifier is not None and self._emotion_classifier.is_loaded:
                face_crops = [self._extract_face_crop(frame, p.bbox) for p in persons]
                if len(persons) > 1:
                    safe_crops = [
                        c if c is not None else np.empty((0, 0, 3), dtype=np.uint8)
                        for c in face_crops
                    ]
                    emotions = self._emotion_classifier.process_batch(safe_crops)
                    for person, emotion in zip(persons, emotions):
                        person.emotion = emotion
                else:
                    for person, crop in zip(persons, face_crops):
                        if crop is not None:
                            person.emotion = self._emotion_classifier.process(crop)

            # Save thumbnail for every key moment frame unconditionally so that
            # each MultimodalFrameAnalysis can carry a valid image_url regardless
            # of how close in time adjacent key moments are.
            if smart_thumbnail_saver is not None:
                smart_thumbnail_saver.force_save(timestamp_s, frame)

            frame_results.append(FrameResult(
                frame_number=frame_num,
                timestamp=timestamp_s,
                persons=persons,
            ))

            if progress_callback and idx % _LOG_INTERVAL == 0:
                pct = 0.15 + 0.50 * ((idx + 1) / len(extracted))
                progress_callback(pct, "processing_frames")

        if progress_callback:
            progress_callback(0.65, "mapping_speakers_to_faces")

        # 3. Build speaker-face mappings using SpeakerFaceMapper
        from .speaker_face_mapper import SpeakerFaceMapper

        mapper = SpeakerFaceMapper(
            min_confidence=getattr(self.config, "mapping_min_confidence", 0.40),
        )
        frame_person_data = [
            {
                "timestamp_s": fr.timestamp,
                "person_ids": [p.person_id for p in fr.persons],
            }
            for fr in frame_results
        ]
        speaker_mappings: List[SpeakerFaceMapping] = mapper.build_mapping(
            speaker_segments, frame_person_data
        )

        # Build speaker_id → person_id lookup (only confident mappings)
        speaker_to_person: Dict[str, str | None] = {
            m.speaker_id: m.person_id if not m.uncertain else None
            for m in speaker_mappings
        }

        if progress_callback:
            progress_callback(0.70, "analyzing_frames_multimodal")

        # 4. Run MultimodalAnalyzer (or CLIP fallback) on each extracted frame
        frame_analyses: List[MultimodalFrameAnalysis] = []

        # Load multimodal analyzer if configured
        if self._multimodal_analyzer is not None:
            try:
                self._multimodal_analyzer.load()
            except Exception:
                logger.warning(
                    "Failed to load multimodal model; proceeding with CLIP fallback.",
                    exc_info=True,
                )

        # extract_at_timestamps returns (frame_number, timestamp_s, rgb_frame) 3-tuples
        for idx, (frame_num, timestamp_s, frame) in enumerate(extracted):
            fr = frame_results[idx]
            face_detected = len(fr.persons) > 0

            # Convert back to ms for key moment lookup and MultimodalFrameAnalysis
            actual_ts_ms = timestamp_s * 1000.0

            # Find the closest key moment for context
            closest_km = None
            best_diff = float("inf")
            for km in key_moments:
                diff = abs(km.timestamp_ms - actual_ts_ms)
                if diff < best_diff:
                    best_diff = diff
                    closest_km = km

            moment_type = getattr(closest_km, "moment_type", None)
            context_text = getattr(closest_km, "context_text", "") or ""
            km_speaker_id = getattr(closest_km, "speaker_id", None)
            person_id = speaker_to_person.get(km_speaker_id) if km_speaker_id else None

            # Resolve the thumbnail URL for this frame (if one was saved by the saver)
            frame_image_url: str | None = None
            if smart_thumbnail_saver is not None:
                frame_image_url = smart_thumbnail_saver.thumbnails.get(timestamp_s)

            from app.core.audio.data_types import MomentType as _MT

            if self._multimodal_analyzer is not None:
                # Always run Qwen regardless of face_detected — face_detected is
                # metadata only and must NOT gate multimodal inference.
                best_person_id = fr.persons[0].person_id if fr.persons else (person_id or "")
                analysis = self._multimodal_analyzer.analyze_frame(
                    frame_rgb=frame,
                    moment_type=moment_type,
                    context_text=context_text,
                    speaker_id=km_speaker_id or "",
                    person_id=best_person_id,
                    frame_number=frame_num,
                    timestamp_ms=actual_ts_ms,
                )
                # Back-fill speaker_id from mapping if not already set
                if analysis.speaker_id is None:
                    analysis.speaker_id = km_speaker_id
                # Propagate face detection state, thumbnail URL, and timestamp (seconds) as metadata
                analysis.face_detected = face_detected
                analysis.image_url = frame_image_url
                analysis.timestamp = timestamp_s
                frame_analyses.append(analysis)
            else:
                # CLIP-only fallback when multimodal analyzer not configured
                clip_result: Dict[str, Any] = {}
                if (self._emotion_classifier is not None
                        and self._emotion_classifier.is_loaded
                        and fr.persons):
                    crop = self._extract_face_crop(frame, fr.persons[0].bbox)
                    if crop is not None:
                        emotion_data = self._emotion_classifier.process(crop)
                        if emotion_data is not None:
                            clip_result = {
                                "emotion_primary": emotion_data.primary_emotion,
                                "emotion_confidence": emotion_data.confidence,
                            }

                frame_analyses.append(MultimodalFrameAnalysis(
                    frame_number=frame_num,
                    timestamp_ms=actual_ts_ms,
                    timestamp=timestamp_s,
                    speaker_id=km_speaker_id,
                    person_id=person_id,
                    moment_type=moment_type or _MT.SPEECH_ONSET,
                    multimodal_description=None,
                    emotion_primary=clip_result.get("emotion_primary"),
                    emotion_confidence=clip_result.get("emotion_confidence"),
                    attention_direction=None,
                    engagement_level=None,
                    face_detected=face_detected,
                    fallback_model="clip" if fr.persons else None,
                    fallback_reason="multimodal_not_configured",
                    image_url=frame_image_url,
                ))

        if progress_callback:
            progress_callback(0.90, "computing_metrics")

        # 5. Aggregate standard per-person metrics from frame_results
        total_frames, duration, native_fps = self._frame_extractor.get_video_properties(video_path)
        session_metrics = self._compute_session_metrics(frame_results, duration)

        t_elapsed = _time.time() - t_start
        logger.info(
            "Smart pipeline complete: %d frames in %.1fs",
            len(frame_results), t_elapsed,
        )

        if progress_callback:
            progress_callback(1.0, "done")

        return VisionResult(
            video_path=video_path,
            total_frames=total_frames,
            fps_processed=0.0,  # N/A for smart mode (not uniform)
            duration_seconds=duration,
            frames=frame_results,
            processing_time_seconds=round(t_elapsed, 3),
            session_metrics=session_metrics,
            smart_mode=True,
            key_moment_analyses=frame_analyses,
            speaker_face_mappings=speaker_mappings,
            frame_thumbnails=smart_thumbnail_saver.thumbnails if smart_thumbnail_saver is not None else {},
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
