"""Audio extraction from video files.

Extracts a mono 16 kHz waveform from a video file using librosa.  The
extracted waveform can be used directly by downstream processors or
written to a temporary WAV file (required by Pyannote which expects a
file path rather than an in-memory buffer).
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np

from app.core.audio.base_processor import AudioBaseProcessor
from app.core.audio.config import AudioConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class AudioExtractionError(Exception):
    """Raised when audio cannot be extracted from the given file."""


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


class AudioExtractor(AudioBaseProcessor):
    """Extract a mono waveform from a video or audio file.

    Parameters
    ----------
    config : AudioConfig
        Pipeline configuration.  ``audio_sample_rate`` and ``audio_device``
        are used during extraction.
    """

    processor_name = "audio_extractor"

    def __init__(self, config: AudioConfig) -> None:
        super().__init__(device=config.audio_device)
        self.config = config

    # ------------------------------------------------------------------
    # Lifecycle (no-op — librosa does not require persistent state)
    # ------------------------------------------------------------------

    def load(self) -> None:
        """No-op — no model to load."""
        self._loaded = True

    def unload(self) -> None:
        """No-op."""
        self._loaded = False

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def process(self, video_path: str) -> tuple[np.ndarray, int, float]:
        """Extract audio from *video_path*.

        Parameters
        ----------
        video_path : str
            Path to the video (or audio) file.

        Returns
        -------
        tuple[np.ndarray, int, float]
            ``(waveform, sample_rate, duration)`` where *waveform* is a
            float32 mono numpy array, *sample_rate* is the target sample rate
            from config, and *duration* is the file duration in seconds.

        Raises
        ------
        AudioExtractionError
            If the file does not exist or librosa cannot decode it.
        """
        if not Path(video_path).exists():
            raise AudioExtractionError(f"File not found: {video_path}")

        try:
            import librosa  # type: ignore

            waveform, sr = librosa.load(
                video_path,
                sr=self.config.audio_sample_rate,
                mono=True,
            )
            duration: float = float(len(waveform)) / sr
            logger.info(
                "Extracted audio from %s — sr=%d, duration=%.2fs, samples=%d",
                video_path,
                sr,
                duration,
                len(waveform),
            )
            return waveform.astype(np.float32), sr, duration

        except AudioExtractionError:
            raise
        except Exception as exc:
            raise AudioExtractionError(
                f"Failed to extract audio from '{video_path}': {exc}"
            ) from exc

    def get_audio_properties(self, video_path: str) -> dict:
        """Return basic audio metadata for *video_path*.

        Returns
        -------
        dict
            Keys: ``duration``, ``sample_rate``, ``channels``, ``has_audio``.
        """
        if not Path(video_path).exists():
            return {"duration": 0.0, "sample_rate": 0, "channels": 0, "has_audio": False}

        try:
            import soundfile as sf  # type: ignore

            info = sf.info(video_path)
            return {
                "duration": float(info.duration),
                "sample_rate": int(info.samplerate),
                "channels": int(info.channels),
                "has_audio": True,
            }
        except Exception:
            # Fall back to librosa for formats soundfile cannot inspect (e.g. mp4/mkv)
            try:
                import librosa  # type: ignore

                duration = librosa.get_duration(path=video_path)
                return {
                    "duration": float(duration),
                    "sample_rate": self.config.audio_sample_rate,
                    "channels": 1,
                    "has_audio": duration > 0,
                }
            except Exception:
                return {"duration": 0.0, "sample_rate": 0, "channels": 0, "has_audio": False}

    def write_temp_wav(self, waveform: np.ndarray, sample_rate: int) -> str:
        """Write *waveform* to a temporary WAV file and return its path.

        The caller is responsible for deleting the file when finished.

        Parameters
        ----------
        waveform : np.ndarray
            Float32 mono waveform array.
        sample_rate : int
            Sample rate for the WAV header.

        Returns
        -------
        str
            Absolute path to the temporary WAV file.
        """
        import soundfile as sf  # type: ignore

        temp_path = tempfile.mktemp(suffix=".wav")
        sf.write(temp_path, waveform, sample_rate)
        logger.debug("Wrote temporary WAV to %s (%d samples)", temp_path, len(waveform))
        return temp_path
