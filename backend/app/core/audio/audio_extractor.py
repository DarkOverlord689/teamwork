"""Extracción de audio de archivos de video.

Extrae un waveform mono de 16 kHz de un archivo de video usando librosa.
El waveform extraído puede usarse directamente por los procesadores posteriores
o escribirse en un archivo WAV temporal.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np

from app.core.audio.base_processor import AudioBaseProcessor
from app.core.audio.config import AudioConfig

logger = logging.getLogger(__name__)


class AudioExtractionError(Exception):
    """Se lanza cuando no se puede extraer audio del archivo dado."""


class AudioExtractor(AudioBaseProcessor):
    def __init__(self, config: AudioConfig) -> None:
        super().__init__(device=config.audio_device)
        self.config = config

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def process(self, video_path: str) -> tuple[np.ndarray, int, float]:
        """Extrae audio del video.

        Retorna una tupla con (waveform, sample_rate, duration).
        """
        if not Path(video_path).exists():
            raise AudioExtractionError(f"File not found: {video_path}")

        try:
            import librosa

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
        """Retorna metadatos básicos del audio."""
        if not Path(video_path).exists():
            return {
                "duration": 0.0,
                "sample_rate": 0,
                "channels": 0,
                "has_audio": False,
            }

        try:
            import soundfile as sf

            info = sf.info(video_path)
            return {
                "duration": float(info.duration),
                "sample_rate": int(info.samplerate),
                "channels": int(info.channels),
                "has_audio": True,
            }
        except Exception:
            try:
                import librosa

                duration = librosa.get_duration(path=video_path)
                return {
                    "duration": float(duration),
                    "sample_rate": self.config.audio_sample_rate,
                    "channels": 1,
                    "has_audio": duration > 0,
                }
            except Exception:
                return {
                    "duration": 0.0,
                    "sample_rate": 0,
                    "channels": 0,
                    "has_audio": False,
                }

    def write_temp_wav(self, waveform: np.ndarray, sample_rate: int) -> str:
        """Escribe el waveform en un archivo WAV temporal y retorna su ruta."""
        import soundfile as sf

        temp_path = tempfile.mktemp(suffix=".wav")
        sf.write(temp_path, waveform, sample_rate)
        logger.debug("Wrote temporary WAV to %s (%d samples)", temp_path, len(waveform))
        return temp_path
