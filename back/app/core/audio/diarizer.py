"""Speaker diarization using Pyannote Audio 3.1.

Wraps the ``pyannote/speaker-diarization-3.1`` pretrained pipeline and
produces a list of :class:`~app.core.audio.data_types.SpeakerSegment`
objects sorted by start time with normalized speaker IDs
(``speaker_0``, ``speaker_1``, … assigned in order of first appearance).
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import List

from app.core.audio.base_processor import AudioBaseProcessor
from app.core.audio.config import AudioConfig
from app.core.audio.data_types import SpeakerSegment

# Import Pipeline at module level so it is patchable in tests.
# Wrapped in try/except so the module can be imported in environments
# where pyannote.audio is not installed (e.g. CI without GPU).
try:
    from pyannote.audio import Pipeline  # type: ignore  # noqa: F401
except ImportError:  # pragma: no cover
    Pipeline = None  # type: ignore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class DiarizationError(Exception):
    """Raised when speaker diarization fails."""


# ---------------------------------------------------------------------------
# Diarizer
# ---------------------------------------------------------------------------


class Diarizer(AudioBaseProcessor):
    """Run Pyannote speaker diarization on an audio file.

    Parameters
    ----------
    config : AudioConfig
        Pipeline configuration including auth token, speaker bounds, and
        minimum segment duration.
    """

    processor_name = "diarizer"

    def __init__(self, config: AudioConfig) -> None:
        super().__init__(device=config.audio_device)
        self.config = config
        self._pipeline = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load the Pyannote speaker-diarization-3.1 pipeline.

        Raises
        ------
        DiarizationError
            If ``pyannote_auth_token`` is empty or the pipeline cannot be loaded.
        """
        if not self.config.pyannote_auth_token:
            raise DiarizationError(
                "pyannote_auth_token is required to load the Pyannote pipeline. "
                "Obtain a token from https://huggingface.co/settings/tokens and "
                "set AudioConfig.pyannote_auth_token."
            )

        try:
            import torch  # type: ignore

            if Pipeline is None:
                raise DiarizationError(
                    "pyannote.audio is not installed. "
                    "Install it with: pip install pyannote.audio"
                )

            logger.info("Loading Pyannote speaker-diarization-3.1 pipeline…")
            self._pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.config.pyannote_auth_token,
            )
            self._pipeline.to(torch.device(self.device))
            self._loaded = True
            logger.info("Pyannote pipeline loaded on device=%s", self.device)

        except DiarizationError:
            raise
        except Exception as exc:
            raise DiarizationError(
                f"Failed to load Pyannote pipeline: {exc}"
            ) from exc

    def unload(self) -> None:
        """Release the Pyannote pipeline and free memory."""
        self._pipeline = None
        self._loaded = False
        logger.info("Pyannote pipeline unloaded")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def process(self, audio_path: str) -> list[SpeakerSegment]:
        """Run diarization on *audio_path*.

        Parameters
        ----------
        audio_path : str
            Path to a WAV (or compatible) audio file.

        Returns
        -------
        list[SpeakerSegment]
            Segments sorted by ``start`` time with normalized speaker IDs.

        Raises
        ------
        DiarizationError
            If the pipeline has not been loaded or inference fails.
        """
        if not self._loaded or self._pipeline is None:
            raise DiarizationError(
                "Diarizer is not loaded. Call load() or use as a context manager."
            )

        try:
            logger.info("Running diarization on %s", audio_path)
            diarization = self._pipeline(
                audio_path,
                min_speakers=self.config.diarize_min_speakers,
                max_speakers=self.config.diarize_max_speakers,
            )
        except Exception as exc:
            raise DiarizationError(
                f"Pyannote inference failed on '{audio_path}': {exc}"
            ) from exc

        # Collect raw (start, end, raw_label) triples
        raw_tracks: list[tuple[float, float, str]] = []
        for turn, _, speaker_label in diarization.itertracks(yield_label=True):
            raw_tracks.append((float(turn.start), float(turn.end), str(speaker_label)))

        # Normalize labels by order of first appearance
        label_map: dict[str, str] = OrderedDict()
        all_labels = [label for _, _, label in raw_tracks]
        normalized_labels = self._normalize_labels(all_labels, label_map=label_map)

        # Build SpeakerSegment list, filtering short segments
        segments: list[SpeakerSegment] = []
        for (start, end, _), norm_label in zip(raw_tracks, normalized_labels):
            duration = end - start
            if duration < self.config.diarize_min_duration:
                logger.debug(
                    "Skipping short segment %.2f–%.2f (%s, %.3fs < min %.3fs)",
                    start,
                    end,
                    norm_label,
                    duration,
                    self.config.diarize_min_duration,
                )
                continue
            segments.append(SpeakerSegment(start=start, end=end, speaker_id=norm_label))

        segments.sort(key=lambda s: s.start)
        logger.info(
            "Diarization complete: %d segments, %d speakers",
            len(segments),
            len(label_map),
        )
        return segments

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _normalize_labels(
        self,
        raw_labels: list[str],
        label_map: dict[str, str] | None = None,
    ) -> list[str]:
        """Normalize raw Pyannote speaker labels to ``speaker_0``, ``speaker_1``, …

        Labels are assigned in order of **first appearance** in *raw_labels*,
        not alphabetically.

        Parameters
        ----------
        raw_labels : list[str]
            Raw labels as returned by ``itertracks`` (e.g. ``"SPEAKER_02"``).
        label_map : dict, optional
            If provided, the mapping is written into this dict (useful for
            inspecting the final mapping after the call).

        Returns
        -------
        list[str]
            Normalized labels in the same order as *raw_labels*.
        """
        mapping: dict[str, str] = {}
        normalized: list[str] = []
        for raw in raw_labels:
            if raw not in mapping:
                mapping[raw] = f"speaker_{len(mapping)}"
            normalized.append(mapping[raw])
        if label_map is not None:
            label_map.update(mapping)
        return normalized
