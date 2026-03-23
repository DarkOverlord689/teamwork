"""Frame extraction from video files using OpenCV.

Extracts frames at a configurable sample rate, returning frame number,
timestamp, and the raw numpy frame for downstream processors.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

from .base_processor import BaseProcessor
from .config import VisionConfig

logger = logging.getLogger(__name__)


class FrameExtractor(BaseProcessor):
    """Extract frames from a video file at a target sample FPS.

    Parameters
    ----------
    config : VisionConfig, optional
        Pipeline configuration.  ``fps`` and ``max_frames`` are used as
        defaults when the caller does not override them in ``process()``.
    device : str
        Passed through to ``BaseProcessor`` but unused – OpenCV does not
        require GPU state management.
    """

    def __init__(
        self,
        config: Optional[VisionConfig] = None,
        device: str = "auto",
    ) -> None:
        super().__init__(device=device)
        self.config = config or VisionConfig()

    # ------------------------------------------------------------------
    # Lifecycle (no-ops – OpenCV needs no persistent model)
    # ------------------------------------------------------------------

    def load(self) -> None:
        """No-op.  OpenCV does not require persistent resource allocation."""
        super().load()

    def unload(self) -> None:
        """No-op."""
        super().unload()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_video_properties(
        self, video_path: str
    ) -> Tuple[int, float, float]:
        """Return ``(total_frames, duration_seconds, native_fps)`` for *video_path*.

        Raises
        ------
        FileNotFoundError
            If *video_path* does not exist on disk.
        RuntimeError
            If the file cannot be opened or has zero duration.
        """
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video file: {video_path}")

        try:
            native_fps: float = cap.get(cv2.CAP_PROP_FPS)
            total_frames: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if native_fps <= 0 or total_frames <= 0:
                raise RuntimeError(
                    f"Video has invalid properties (fps={native_fps}, "
                    f"frames={total_frames}): {video_path}"
                )

            duration: float = total_frames / native_fps
            if duration <= 0:
                raise RuntimeError(
                    f"Video has zero or negative duration: {video_path}"
                )

            return total_frames, duration, native_fps
        finally:
            cap.release()

    def process(
        self,
        video_path: str,
        fps: Optional[float] = None,
        max_frames: Optional[int] = None,
    ) -> List[Tuple[int, float, np.ndarray]]:
        """Extract frames from *video_path*.

        Parameters
        ----------
        video_path : str
            Path to the video file.
        fps : float, optional
            Target sample rate.  Defaults to ``config.fps``.
        max_frames : int, optional
            Maximum number of frames to extract.  Defaults to
            ``config.max_frames``.

        Returns
        -------
        list of (frame_number, timestamp_seconds, numpy_frame)
            Each element is a 3-tuple containing the original frame index,
            the corresponding timestamp in seconds, and the decoded BGR
            numpy array.

        Raises
        ------
        FileNotFoundError
            If *video_path* does not exist.
        RuntimeError
            If the video cannot be opened or has zero duration.
        """
        target_fps = fps if fps is not None else self.config.fps
        frame_limit = max_frames if max_frames is not None else self.config.max_frames

        # Validate file
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video file: {video_path}")

        try:
            native_fps: float = cap.get(cv2.CAP_PROP_FPS)
            total_frames: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if native_fps <= 0 or total_frames <= 0:
                logger.warning(
                    "Video has invalid properties (fps=%s, frames=%s): %s",
                    native_fps, total_frames, video_path,
                )
                return []

            duration: float = total_frames / native_fps
            if duration <= 0:
                logger.warning(
                    "Video has zero or negative duration: %s", video_path,
                )
                return []

            # Compute sampling interval (in native frame indices)
            frame_interval = max(1, int(round(native_fps / target_fps)))

            logger.info(
                "Extracting frames from %s – native_fps=%.2f, "
                "target_fps=%.2f, interval=%d, max=%d",
                video_path,
                native_fps,
                target_fps,
                frame_interval,
                frame_limit,
            )

            frames: List[Tuple[int, float, np.ndarray]] = []
            frame_idx = 0

            while len(frames) < frame_limit:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_interval == 0:
                    timestamp = frame_idx / native_fps
                    # Convert BGR (OpenCV default) to RGB for downstream processors
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append((frame_idx, timestamp, rgb_frame))

                frame_idx += 1

            logger.info("Extracted %d frames from %s", len(frames), video_path)
            return frames
        finally:
            cap.release()
