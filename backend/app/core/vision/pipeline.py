"""Pipeline de visión.

Gestiona el ciclo completo del análisis de video: extracción de frames,
detección de rostros, procesamiento por persona (emoción, mirada, gesto, postura),
seguimiento de personas entre frames y agregación de resultados.
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

_LOG_INTERVAL = 25


class VisionPipeline:
    def __init__(self, config: Optional[VisionConfig] = None) -> None:
        self.config = config or VisionConfig()

        self._frame_extractor = FrameExtractor(
            config=self.config, device=self.config.device
        )
        self._face_detector = FaceDetector(
            config=self.config, device=self.config.device
        )

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
            self._person_tracker = PersonTracker(
                config=self.config, device=self.config.device
            )

        self._multimodal_analyzer: Optional[Any] = None
        if getattr(self.config, "use_multimodal", False):
            from .multimodal_analyzer import MultimodalAnalyzer

            self._multimodal_analyzer = MultimodalAnalyzer(config=self.config)

    def load_all(self) -> None:
        """Carga todos los procesadores habilitados."""
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
        """Descarga todos los procesadores y libera recursos."""
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

    def __enter__(self) -> "VisionPipeline":
        self.load_all()
        return self

    def __exit__(self, *args: Any) -> None:
        self.unload_all()

    def process_video(
        self,
        video_path: str,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        session_id: Optional[str] = None,
    ) -> VisionResult:
        """Ejecuta el pipeline completo de visión en el video."""
        t_start = time.time()

        total_frames, duration, native_fps = self._frame_extractor.get_video_properties(
            video_path
        )
        logger.info(
            "Processing video: %s (%.1fs, %d frames, %.1f fps)",
            video_path,
            duration,
            total_frames,
            native_fps,
        )

        raw_frames = self._frame_extractor.process(video_path)
        logger.info("Extracted %d frames for processing", len(raw_frames))

        if progress_callback:
            progress_callback(0.10, "processing_frames")

        thumbnail_saver = None
        if session_id is not None:
            from app.utils.config import settings as _settings
            from .frame_saver import FrameThumbnailSaver

            thumbnail_saver = FrameThumbnailSaver(
                frames_dir=_settings.frames_dir,
                session_id=session_id,
            )

        frame_results: List[FrameResult] = []

        num_raw = len(raw_frames) if raw_frames else 1

        for idx, (frame_number, timestamp, frame) in enumerate(raw_frames):
            if idx % _LOG_INTERVAL == 0:
                logger.info(
                    "Processing frame %d / %d (%.0f%%)",
                    idx + 1,
                    len(raw_frames),
                    (idx + 1) / len(raw_frames) * 100,
                )

            if progress_callback and idx % _LOG_INTERVAL == 0:
                pct = 0.10 + 0.80 * (idx / num_raw)
                progress_callback(pct, "processing_frames")

            persons: List[PersonFrame] = self._face_detector.process(frame)

            if not persons:
                if thumbnail_saver is not None:
                    thumbnail_saver.maybe_save(timestamp, frame)
                frame_results.append(
                    FrameResult(
                        frame_number=frame_number,
                        timestamp=timestamp,
                        persons=[],
                    )
                )
                continue

            if self._person_tracker is not None and self._person_tracker.is_loaded:
                persons = self._person_tracker.process(frame, persons)

            if (
                self._emotion_classifier is not None
                and self._emotion_classifier.is_loaded
                and len(persons) > 1
            ):
                face_crops = [self._extract_face_crop(frame, p.bbox) for p in persons]
                safe_crops = [
                    c if c is not None else np.empty((0, 0, 3), dtype=np.uint8)
                    for c in face_crops
                ]
                emotions = self._emotion_classifier.process_batch(safe_crops)
                for person, emotion in zip(persons, emotions):
                    person.emotion = emotion
            elif (
                self._emotion_classifier is not None
                and self._emotion_classifier.is_loaded
            ):
                for person in persons:
                    face_crop = self._extract_face_crop(frame, person.bbox)
                    if face_crop is not None:
                        person.emotion = self._emotion_classifier.process(face_crop)

            for person in persons:
                if (
                    self._gaze_estimator is not None
                    and self._gaze_estimator.is_loaded
                    and person.landmarks is not None
                ):
                    h, w = frame.shape[:2]
                    person.gaze = self._gaze_estimator.process(person.landmarks, (h, w))

                if (
                    self._gesture_analyzer is not None
                    and self._gesture_analyzer.is_loaded
                    and person.landmarks is not None
                ):
                    person.gesture = self._gesture_analyzer.process(
                        person.person_id, person.landmarks
                    )

                if self._pose_estimator is not None and self._pose_estimator.is_loaded:
                    person.pose = self._pose_estimator.process(frame, person.bbox)

            if thumbnail_saver is not None:
                thumbnail_saver.maybe_save(timestamp, frame)

            frame_results.append(
                FrameResult(
                    frame_number=frame_number,
                    timestamp=timestamp,
                    persons=persons,
                )
            )

        t_elapsed = time.time() - t_start
        logger.info(
            "Video processing complete: %d frames in %.1fs (%.1f frames/s)",
            len(frame_results),
            t_elapsed,
            len(frame_results) / t_elapsed if t_elapsed > 0 else 0,
        )

        if progress_callback:
            progress_callback(0.90, "computing_metrics")

        person_embeddings: Dict[str, Any] = {}
        if self._person_tracker is not None and self._person_tracker.is_loaded:
            person_embeddings = dict(self._person_tracker.known_embeddings)

        session_metrics = self._compute_session_metrics(
            frame_results,
            duration,
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
            frame_thumbnails=thumbnail_saver.thumbnails
            if thumbnail_saver is not None
            else {},
        )

    def process_video_smart(
        self,
        video_path: str,
        key_moments: list,
        speaker_segments: list,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        session_id: Optional[str] = None,
    ) -> VisionResult:
        """Ejecuta el pipeline de visión inteligente usando momentos clave guiados por audio."""
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

        timestamps_s = [km.timestamp_ms / 1000.0 for km in key_moments]
        logger.info(
            "Smart pipeline: extracting %d targeted frames from %s",
            len(timestamps_s),
            video_path,
        )

        if progress_callback:
            progress_callback(0.05, "extracting_targeted_frames")

        extracted = self._frame_extractor.extract_at_timestamps(
            video_path, timestamps_s
        )
        logger.info("Smart pipeline: extracted %d frames", len(extracted))

        if progress_callback:
            progress_callback(0.15, "processing_frames")

        smart_thumbnail_saver = None
        if session_id is not None:
            from app.utils.config import settings as _settings
            from .frame_saver import FrameThumbnailSaver

            smart_thumbnail_saver = FrameThumbnailSaver(
                frames_dir=_settings.frames_dir,
                session_id=session_id,
            )

        frame_results: List[FrameResult] = []
        ts_to_km: Dict[int, Any] = {int(km.timestamp_ms): km for km in key_moments}

        for idx, (frame_num, timestamp_s, frame) in enumerate(extracted):
            persons: List[PersonFrame] = self._face_detector.process(frame)

            if (
                persons
                and self._person_tracker is not None
                and self._person_tracker.is_loaded
            ):
                persons = self._person_tracker.process(frame, persons)

            if (
                persons
                and self._emotion_classifier is not None
                and self._emotion_classifier.is_loaded
            ):
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

            if smart_thumbnail_saver is not None:
                smart_thumbnail_saver.force_save(timestamp_s, frame)

            frame_results.append(
                FrameResult(
                    frame_number=frame_num,
                    timestamp=timestamp_s,
                    persons=persons,
                )
            )

            if progress_callback and idx % _LOG_INTERVAL == 0:
                pct = 0.15 + 0.50 * ((idx + 1) / len(extracted))
                progress_callback(pct, "processing_frames")

        if progress_callback:
            progress_callback(0.65, "mapping_speakers_to_faces")

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

        speaker_to_person: Dict[str, str | None] = {
            m.speaker_id: m.person_id if not m.uncertain else None
            for m in speaker_mappings
        }

        if progress_callback:
            progress_callback(0.70, "analyzing_frames_multimodal")

        frame_analyses: List[MultimodalFrameAnalysis] = []

        if self._multimodal_analyzer is not None:
            try:
                self._multimodal_analyzer.load()
            except Exception:
                logger.warning(
                    "Failed to load multimodal model; proceeding with CLIP fallback.",
                    exc_info=True,
                )

        for idx, (frame_num, timestamp_s, frame) in enumerate(extracted):
            fr = frame_results[idx]
            face_detected = len(fr.persons) > 0

            actual_ts_ms = timestamp_s * 1000.0

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

            frame_image_url: str | None = None
            if smart_thumbnail_saver is not None:
                frame_image_url = smart_thumbnail_saver.thumbnails.get(timestamp_s)

            from app.core.audio.data_types import MomentType as _MT

            if self._multimodal_analyzer is not None:
                best_person_id = (
                    fr.persons[0].person_id if fr.persons else (person_id or "")
                )
                analysis = self._multimodal_analyzer.analyze_frame(
                    frame_rgb=frame,
                    moment_type=moment_type,
                    context_text=context_text,
                    speaker_id=km_speaker_id or "",
                    person_id=best_person_id,
                    frame_number=frame_num,
                    timestamp_ms=actual_ts_ms,
                )
                if analysis.speaker_id is None:
                    analysis.speaker_id = km_speaker_id
                analysis.face_detected = face_detected
                analysis.image_url = frame_image_url
                analysis.timestamp = timestamp_s
                frame_analyses.append(analysis)
            else:
                clip_result: Dict[str, Any] = {}
                if (
                    self._emotion_classifier is not None
                    and self._emotion_classifier.is_loaded
                    and fr.persons
                ):
                    crop = self._extract_face_crop(frame, fr.persons[0].bbox)
                    if crop is not None:
                        emotion_data = self._emotion_classifier.process(crop)
                        if emotion_data is not None:
                            clip_result = {
                                "emotion_primary": emotion_data.primary_emotion,
                                "emotion_confidence": emotion_data.confidence,
                            }

                frame_analyses.append(
                    MultimodalFrameAnalysis(
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
                    )
                )

        if progress_callback:
            progress_callback(0.90, "computing_metrics")

        total_frames, duration, native_fps = self._frame_extractor.get_video_properties(
            video_path
        )
        session_metrics = self._compute_session_metrics(frame_results, duration)

        t_elapsed = _time.time() - t_start
        logger.info(
            "Smart pipeline complete: %d frames in %.1fs",
            len(frame_results),
            t_elapsed,
        )

        if progress_callback:
            progress_callback(1.0, "done")

        return VisionResult(
            video_path=video_path,
            total_frames=total_frames,
            fps_processed=0.0,
            duration_seconds=duration,
            frames=frame_results,
            processing_time_seconds=round(t_elapsed, 3),
            session_metrics=session_metrics,
            smart_mode=True,
            key_moment_analyses=frame_analyses,
            speaker_face_mappings=speaker_mappings,
            frame_thumbnails=smart_thumbnail_saver.thumbnails
            if smart_thumbnail_saver is not None
            else {},
        )

    @staticmethod
    def _compute_session_metrics(
        frame_results: List[FrameResult],
        duration: float,
    ) -> SessionMetrics:
        """Calcula métricas por persona y a nivel de sesión."""
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

                if (
                    person.gesture is not None
                    and person.gesture.gesture_type != "neutral"
                ):
                    person_gestures[pid][person.gesture.gesture_type] += 1

        per_person: List[PersonMetrics] = []
        for pid in sorted(person_frames_seen.keys()):
            total_seen = person_frames_seen[pid]

            gaze_pct = (
                (person_gaze_contact[pid] / total_seen * 100.0)
                if total_seen > 0
                else 0.0
            )

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

            orientations = person_orientations.get(pid, [])
            avg_orientation = float(np.mean(orientations)) if orientations else 0.0

            gestures = dict(person_gestures.get(pid, {}))

            gaze_score = gaze_pct / 100.0
            orientation_score = (
                max(0.0, 1.0 - avg_orientation / 90.0) if orientations else 0.0
            )
            nod_score = min(1.0, gestures.get("nod", 0) / max(1, total_seen) * 10.0)
            emotion_score = 1.0 - emotion_distribution.get("neutral", 1.0)
            attention = (
                0.40 * gaze_score
                + 0.30 * orientation_score
                + 0.20 * nod_score
                + 0.10 * emotion_score
            )

            per_person.append(
                PersonMetrics(
                    person_id=pid,
                    total_frames_seen=total_seen,
                    gaze_contact_percentage=round(gaze_pct, 2),
                    dominant_emotion=dominant_emotion,
                    emotion_distribution=emotion_distribution,
                    average_body_orientation=round(avg_orientation, 2),
                    gesture_counts=gestures,
                    attention_score=round(min(1.0, max(0.0, attention)), 4),
                )
            )

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
        """Extrae y retorna el recorte del rostro del frame usando el bbox."""
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
