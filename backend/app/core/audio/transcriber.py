"""Deepgram transcription processor for the audio pipeline.

``Transcriber`` calls the Deepgram ``listen.v1.media.transcribe_file`` API to
produce per-word transcripts with timestamps, then aligns those transcription
segments to the diariser's ``SpeakerSegment`` output to assign speaker labels.
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


try:
    from deepgram import DeepgramClient  # type: ignore
except ImportError:  # pragma: no cover
    DeepgramClient = None  # type: ignore


class TranscriptionError(Exception):
    """Raised when transcription fails for a segment."""


class Transcriber(AudioBaseProcessor):
    """ASR transcription using the Deepgram API (nova-2).

    The entire audio waveform is encoded as a WAV file in memory and sent to
    Deepgram in a single request.  The response includes per-word timestamps
    and speaker labels when diarization was enabled upstream.  No local model
    weights are loaded, so memory usage is negligible.

    Parameters
    ----------
    config : AudioConfig
        Pipeline-wide configuration (language, API key, sample rate, etc.).
    """

    processor_name = "transcriber"

    def __init__(self, config: AudioConfig) -> None:
        super().__init__(device=config.audio_device)
        self.config = config
        self._client = None

    # ------------------------------------------------------------------
    # Lifecycle (lightweight — client is stateless)
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Validate that the Deepgram API key is available."""
        if not self.config.deepgram_api_key:
            logger.error(
                "deepgram_api_key is not set in AudioConfig — transcription skipped."
            )
            raise TranscriptionError("deepgram_api_key is required for transcription.")

        if DeepgramClient is None:
            raise TranscriptionError(
                "deepgram-sdk is not installed. "
                "Install it with: pip install deepgram-sdk"
            )

        self._client = DeepgramClient(api_key=self.config.deepgram_api_key)
        self._loaded = True
        logger.info("Deepgram client initialised for transcription")

    def unload(self) -> None:
        """Release the Deepgram client reference."""
        self._client = None
        self._loaded = False
        logger.info("Deepgram client unloaded")

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
            One entry per Deepgram utterance, with the speaker ID assigned via
            maximum-overlap alignment against diarisation segments.
        """
        if not segments:
            return []

        if not self._loaded or self._client is None:
            raise TranscriptionError(
                "Transcriber is not loaded. Call load() or use as a context manager."
            )

        wav_bytes = _waveform_to_wav_bytes(waveform, self.config.audio_sample_rate)

        try:
            response = self._client.listen.v1.media.transcribe_file(
                request=wav_bytes,
                model=self.config.deepgram_model,
                punctuate=True,
                language=self.config.whisper_language,
                smart_format=True,
            )
        except Exception as exc:
            raise TranscriptionError(
                f"Deepgram transcription API failed: {exc}"
            ) from exc

        alternatives = (
            response.results.channels[0].alternatives
            if response.results and response.results.channels
            else []
        )
        if not alternatives:
            logger.warning("Deepgram returned no alternatives")
            return []

        words = alternatives[0].words or []
        if not words:
            logger.warning("Deepgram returned no words")
            return []

        # Group consecutive words from the same speaker into transcript segments
        results: list[TranscriptSegment] = []
        current_words: list[WordTimestamp] = []
        current_speaker = None
        seg_start = 0.0

        for word in words:
            speaker = str(getattr(word, "speaker", "0"))
            wt = WordTimestamp(
                word=word.word or "",
                start=word.start,
                end=word.end,
                confidence=getattr(word, "confidence", 0.0) or 0.0,
            )

            if current_speaker is None:
                current_speaker = speaker
                seg_start = word.start
                current_words = [wt]
            elif speaker == current_speaker:
                current_words.append(wt)
            else:
                text = " ".join(w.word for w in current_words).strip()
                if text:
                    speaker_id = _assign_speaker(
                        seg_start, current_words[-1].end, segments
                    )
                    results.append(
                        TranscriptSegment(
                            start=seg_start,
                            end=current_words[-1].end,
                            speaker_id=speaker_id,
                            text=text,
                            words=current_words,
                            language=self.config.whisper_language,
                            no_speech_prob=0.0,
                        )
                    )
                current_speaker = speaker
                seg_start = word.start
                current_words = [wt]

        # Flush final segment
        if current_words:
            text = " ".join(w.word for w in current_words).strip()
            if text:
                speaker_id = _assign_speaker(seg_start, current_words[-1].end, segments)
                results.append(
                    TranscriptSegment(
                        start=seg_start,
                        end=current_words[-1].end,
                        speaker_id=speaker_id,
                        text=text,
                        words=current_words,
                        language=self.config.whisper_language,
                        no_speech_prob=0.0,
                    )
                )

        logger.info(
            "Deepgram transcription: %d utterance segments aligned to %d speaker segments.",
            len(results),
            len(segments),
        )
        return results


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _waveform_to_wav_bytes(waveform: np.ndarray, sample_rate: int) -> bytes:
    """Encode a float32 mono waveform as a 16-bit PCM WAV byte string."""
    pcm = np.clip(waveform, -1.0, 1.0)
    pcm_int16 = (pcm * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
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
        mid = (seg_start + seg_end) / 2.0
        best_speaker = min(
            diarization_segments,
            key=lambda d: abs((d.start + d.end) / 2.0 - mid),
        ).speaker_id

    return best_speaker
