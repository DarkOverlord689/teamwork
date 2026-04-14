"""Utility for saving key frame thumbnails to disk during vision processing.

Saves one JPEG per configurable interval (default 3 seconds) so that the
audit tab can display representative frames alongside the analysis timeline.

Files are saved to ``{frames_dir}/{session_id}/frame_{timestamp:.1f}.jpg``.
The relative URL path returned is ``/api/v1/vision/frames/{session_id}/frame_{timestamp:.1f}.jpg``
so it can be served directly by the FastAPI endpoint without any extra lookup.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Save one thumbnail every this many seconds of video time
DEFAULT_THUMBNAIL_INTERVAL_S: float = 3.0


class FrameThumbnailSaver:
    """Save representative JPEG thumbnails for key frames.

    Parameters
    ----------
    frames_dir : str
        Root directory where session sub-folders will be created.
    session_id : str
        UUID string of the analysis session; used as the sub-folder name.
    interval_s : float
        Minimum elapsed video time (seconds) between saved thumbnails.
    max_dimension : int
        Largest side of the saved JPEG.  Frames are proportionally scaled
        so neither width nor height exceeds this value.
    jpeg_quality : int
        JPEG compression quality (0-100).
    """

    def __init__(
        self,
        frames_dir: str,
        session_id: str,
        interval_s: float = DEFAULT_THUMBNAIL_INTERVAL_S,
        max_dimension: int = 320,
        jpeg_quality: int = 75,
    ) -> None:
        self.frames_dir = Path(frames_dir)
        self.session_id = session_id
        self.interval_s = interval_s
        self.max_dimension = max_dimension
        self.jpeg_quality = jpeg_quality

        self._session_dir = self.frames_dir / session_id
        self._last_saved_ts: float = -interval_s  # ensure first frame is always saved
        self._thumbnails: Dict[float, str] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def maybe_save(
        self,
        timestamp: float,
        frame_rgb: np.ndarray,
    ) -> Optional[str]:
        """Save *frame_rgb* as a thumbnail if enough time has elapsed.

        Parameters
        ----------
        timestamp : float
            Frame timestamp in seconds.
        frame_rgb : np.ndarray
            RGB frame (H x W x 3 uint8).

        Returns
        -------
        str or None
            Relative URL path if the frame was saved, ``None`` otherwise.
        """
        if timestamp - self._last_saved_ts < self.interval_s:
            return None

        return self._write(timestamp, frame_rgb)

    def force_save(
        self,
        timestamp: float,
        frame_rgb: np.ndarray,
    ) -> Optional[str]:
        """Save *frame_rgb* unconditionally, bypassing the time interval check.

        Use this for key moment frames that must always have a thumbnail
        regardless of how recently another frame was saved.

        Parameters
        ----------
        timestamp : float
            Frame timestamp in seconds.
        frame_rgb : np.ndarray
            RGB frame (H x W x 3 uint8).

        Returns
        -------
        str or None
            Relative URL path if the frame was saved, ``None`` otherwise.
        """
        return self._write(timestamp, frame_rgb)

    def _write(
        self,
        timestamp: float,
        frame_rgb: np.ndarray,
    ) -> Optional[str]:
        """Internal: write a JPEG thumbnail to disk and record its URL."""
        try:
            self._ensure_dir()
            filename = f"frame_{timestamp:.1f}.jpg"
            filepath = self._session_dir / filename

            # Scale down if needed
            thumbnail = self._resize(frame_rgb)

            # Convert RGB → BGR for OpenCV imwrite
            bgr = cv2.cvtColor(thumbnail, cv2.COLOR_RGB2BGR)
            ok = cv2.imwrite(
                str(filepath),
                bgr,
                [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
            )
            if not ok:
                logger.warning("cv2.imwrite failed for %s", filepath)
                return None

            url = f"/api/v1/vision/frames/{self.session_id}/{filename}"
            self._thumbnails[timestamp] = url
            self._last_saved_ts = timestamp
            logger.debug("Saved thumbnail: %s", filepath)
            return url

        except Exception:
            logger.exception("Failed to save thumbnail at t=%.3f for session %s", timestamp, self.session_id)
            return None

    @property
    def thumbnails(self) -> Dict[float, str]:
        """Return the collected ``{timestamp: url}`` mapping."""
        return dict(self._thumbnails)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ensure_dir(self) -> None:
        self._session_dir.mkdir(parents=True, exist_ok=True)

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if max(h, w) <= self.max_dimension:
            return frame
        if w >= h:
            new_w = self.max_dimension
            new_h = int(h * self.max_dimension / w)
        else:
            new_h = self.max_dimension
            new_w = int(w * self.max_dimension / h)
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
