"""OpenAI Whisper API transcription processor for the audio pipeline.

``Transcriber`` calls the OpenAI ``audio.transcriptions`` API to produce
per-segment transcripts with timestamps, then aligns those transcription
segments to the diariser's ``SpeakerSegment`` output to assign speaker labels.

Replacing the local Whisper model eliminates the SIGSEGV / OOM crash that
occurred when loading model weights after diarization on memory-constrained
CPU workers.
"""

from __future__ import annotations

import io
import logging
import wave

import numpy as np

from app.core.audio.base_processor import AudioBaseProcessor
from app.core.audio.config import AudioConfig
from app.core.audio.data_types import SpeakerSegment, TranscriptSegment, WordTimestamp

logger = logging.getLogger(__name__)


class TranscriptionError(Exception):
    """Raised when transcription fails for a segment."""


class Transcriber(AudioBaseProcessor):
    """ASR transcription using the OpenAI Whisper API (whisper-1).

    The entire audio file is sent to OpenAI in a single request using
    ``response_format="verbose_json"``, which returns segment-level timestamps.
    Those segments are then aligned to the diarisation output to assign
    speaker IDs.  No local model weights are loaded, so memory usage is
    negligible.

    Parameters
    ----------
    config : AudioConfig
        Pipeline-wide configuration (language, API key, sample rate, etc.).
    """

    processor_name = "transcriber"

    def __init__(self, config: AudioConfig) -> None:
        super().__init__(device=config.audio_device)
        self.config = config
        self._client = None  # openai.OpenAI instance, created lazily

    # ------------------------------------------------------------------
    # Lifecycle (no-ops — no local model to manage)
    # ------------------------------------------------------------------

    def load(self) -> None:
        """No-op: no local model weights to load for the API-based transcriber."""
        self._loaded = True

    def unload(self) -> None:
        """No-op: no local model weights to release."""
        self._loaded = False

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------

    def process(
        self,
        waveform: np.ndarray,
        segments: list[SpeakerSegment],
    ) -> list[TranscriptSegment]:
        """Transcribe the full waveform and align results to speaker segments.

        Parameters
        ----------
        waveform : np.ndarray
            Full-session mono audio as a 1-D float32 array at
            ``config.audio_sample_rate`` Hz.
        segments : list[SpeakerSegment]
            Diariser output — each segment specifies a speaker and time range.

        Returns
        -------
        list[TranscriptSegment]
            One entry per OpenAI transcription segment, with the speaker ID
            assigned via maximum-overlap alignment against diarisation segments.
        """
        if not segments:
            return []

        api_key = self.config.openai_api_key
        if not api_key:
            logger.error(
                "openai_api_key is not set in AudioConfig — transcription skipped."
            )
            return []

        # Build the OpenAI client lazily so the import only happens here.
        import openai

        client = openai.OpenAI(api_key=api_key)

        # Encode the full waveform as a WAV file in memory so we can pass it
        # directly to the API without touching disk.
        wav_bytes = _waveform_to_wav_bytes(waveform, self.config.audio_sample_rate)

        try:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=("audio.wav", wav_bytes, "audio/wav"),
                language=self.config.whisper_language,
                response_format="verbose_json",
                timestamp_granularities=["segment"],
            )
        except openai.OpenAIError as exc:
            raise TranscriptionError(f"OpenAI transcription API failed: {exc}") from exc

        # ``response.segments`` is a list of objects with .start, .end, .text.
        api_segments = getattr(response, "segments", None) or []

        results: list[TranscriptSegment] = []
        language: str = getattr(response, "language", self.config.whisper_language) or self.config.whisper_language

        for api_seg in api_segments:
            seg_start: float = float(api_seg.start)
            seg_end: float = float(api_seg.end)
            text: str = (api_seg.text or "").strip()

            if not text:
                continue

            speaker_id = _assign_speaker(seg_start, seg_end, segments)

            # The verbose_json segment-level response does not include per-word
            # timestamps unless timestamp_granularities includes "word".  We
            # request segment-only granularity to keep payload small, so
            # words is empty here.
            results.append(
                TranscriptSegment(
                    start=seg_start,
                    end=seg_end,
                    speaker_id=speaker_id,
                    text=text,
                    words=[],
                    language=language,
                    no_speech_prob=0.0,
                )
            )

        logger.info(
            "OpenAI transcription: %d API segments aligned to %d speaker segments.",
            len(results),
            len(segments),
        )
        return results


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _waveform_to_wav_bytes(waveform: np.ndarray, sample_rate: int) -> bytes:
    """Encode a float32 mono waveform as a 16-bit PCM WAV byte string."""
    # Clamp to [-1, 1] and convert to int16.
    pcm = np.clip(waveform, -1.0, 1.0)
    pcm_int16 = (pcm * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_int16.tobytes())

    return buf.getvalue()


def _assign_speaker(
    seg_start: float,
    seg_end: float,
    diarization_segments: list[SpeakerSegment],
) -> str:
    """Return the speaker ID with the greatest overlap with [seg_start, seg_end].

    If no diarization segment overlaps, returns the speaker ID of the
    diarization segment whose midpoint is nearest to the transcription
    segment's midpoint (last-resort fallback).
    """
    best_speaker = "UNKNOWN"
    best_overlap = 0.0

    for ds in diarization_segments:
        overlap = min(seg_end, ds.end) - max(seg_start, ds.start)
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = ds.speaker_id

    if best_overlap == 0.0 and diarization_segments:
        # Nearest-midpoint fallback.
        mid = (seg_start + seg_end) / 2.0
        best_speaker = min(
            diarization_segments,
            key=lambda d: abs((d.start + d.end) / 2.0 - mid),
        ).speaker_id

    return best_speaker
