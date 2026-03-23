"""Whisper-based ASR transcription processor for the audio pipeline.

``Transcriber`` wraps OpenAI Whisper to produce per-segment transcripts with
word-level timestamps aligned to the diariser's ``SpeakerSegment`` output.
"""

from __future__ import annotations

import numpy as np

from app.core.audio.base_processor import AudioBaseProcessor
from app.core.audio.config import AudioConfig
from app.core.audio.data_types import SpeakerSegment, TranscriptSegment, WordTimestamp


class TranscriptionError(Exception):
    """Raised when transcription fails for a segment."""


class Transcriber(AudioBaseProcessor):
    """ASR transcription using OpenAI Whisper.

    Parameters
    ----------
    config : AudioConfig
        Pipeline-wide configuration (model size, language, thresholds, etc.).
    """

    processor_name = "transcriber"

    # Minimum chunk duration (seconds) worth transcribing.
    _MIN_CHUNK_SECONDS: float = 0.5

    # Whisper's hard limit for a single transcription call (seconds).
    _MAX_CHUNK_SECONDS: float = 30.0

    # Overlap when splitting long chunks to avoid cutting words at boundaries.
    _CHUNK_OVERLAP_SECONDS: float = 1.0

    def __init__(self, config: AudioConfig) -> None:
        super().__init__(device=config.audio_device)
        self.config = config
        self._model = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load the Whisper model into memory."""
        import whisper  # type: ignore

        device = self.device if self.device != "auto" else None
        self._model = whisper.load_model(self.config.whisper_model_size, device=device)
        self._loaded = True

    def unload(self) -> None:
        """Release Whisper model weights."""
        self._model = None
        self._loaded = False

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------

    def process(
        self,
        waveform: np.ndarray,
        segments: list[SpeakerSegment],
    ) -> list[TranscriptSegment]:
        """Transcribe audio for each speaker segment.

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
            One entry per segment that passes the no-speech filter, with
            word-level timestamps offset to the session timeline.
        """
        results: list[TranscriptSegment] = []

        for segment in segments:
            transcript = self._transcribe_segment(waveform, segment)
            if transcript is not None:
                results.append(transcript)

        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _transcribe_segment(
        self,
        waveform: np.ndarray,
        segment: SpeakerSegment,
    ) -> TranscriptSegment | None:
        """Transcribe a single ``SpeakerSegment``.

        Returns ``None`` when the segment is too short or classified as
        non-speech by Whisper.
        """
        sample_rate = self.config.audio_sample_rate
        start_sample = int(segment.start * sample_rate)
        end_sample = int(segment.end * sample_rate)
        audio_chunk = waveform[start_sample:end_sample]

        duration = segment.end - segment.start
        if duration < self._MIN_CHUNK_SECONDS:
            return None

        # Ensure float32 as required by Whisper.
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)

        if duration <= self._MAX_CHUNK_SECONDS:
            raw = self._call_whisper(audio_chunk)
        else:
            raw = self._transcribe_long_chunk(audio_chunk, sample_rate)

        # Filter out non-speech segments.
        no_speech_prob: float = raw.get("no_speech_prob", 0.0)
        if no_speech_prob > self.config.whisper_no_speech_threshold:
            return None

        # Extract word timestamps, offsetting by segment.start.
        words: list[WordTimestamp] = []
        for whisper_seg in raw.get("segments", []):
            for w in whisper_seg.get("words", []):
                words.append(
                    WordTimestamp(
                        word=w["word"],
                        start=w["start"] + segment.start,
                        end=w["end"] + segment.start,
                        confidence=w.get("probability", 0.0),
                    )
                )

        language: str = raw.get("language", self.config.whisper_language) or self.config.whisper_language

        return TranscriptSegment(
            start=segment.start,
            end=segment.end,
            speaker_id=segment.speaker_id,
            text=raw["text"].strip(),
            words=words,
            language=language,
            no_speech_prob=no_speech_prob,
        )

    def _call_whisper(self, audio_chunk: np.ndarray) -> dict:
        """Invoke the Whisper model on a single audio array."""
        return self._model.transcribe(
            audio_chunk,
            language=self.config.whisper_language,
            beam_size=self.config.whisper_beam_size,
            word_timestamps=True,
            initial_prompt=self.config.whisper_initial_prompt,
            no_speech_threshold=self.config.whisper_no_speech_threshold,
        )

    def _transcribe_long_chunk(
        self,
        audio_chunk: np.ndarray,
        sample_rate: int,
    ) -> dict:
        """Split a >30 s chunk into overlapping sub-chunks and merge results.

        Each sub-chunk is transcribed independently; text and segment/word
        data are concatenated and the ``no_speech_prob`` is averaged.
        """
        max_samples = int(self._MAX_CHUNK_SECONDS * sample_rate)
        overlap_samples = int(self._CHUNK_OVERLAP_SECONDS * sample_rate)
        step_samples = max_samples - overlap_samples

        combined_text_parts: list[str] = []
        combined_segments: list[dict] = []
        no_speech_probs: list[float] = []

        offset_samples = 0
        while offset_samples < len(audio_chunk):
            end_sample = min(offset_samples + max_samples, len(audio_chunk))
            sub_chunk = audio_chunk[offset_samples:end_sample]

            raw = self._call_whisper(sub_chunk)

            no_speech_probs.append(raw.get("no_speech_prob", 0.0))

            text_part = raw.get("text", "").strip()
            if text_part:
                combined_text_parts.append(text_part)

            # Shift word/segment timestamps by the sub-chunk's time offset.
            time_offset = offset_samples / sample_rate
            for seg in raw.get("segments", []):
                shifted_seg = dict(seg)
                shifted_words = []
                for w in seg.get("words", []):
                    shifted_words.append({
                        **w,
                        "start": w["start"] + time_offset,
                        "end": w["end"] + time_offset,
                    })
                shifted_seg["words"] = shifted_words
                combined_segments.append(shifted_seg)

            if end_sample >= len(audio_chunk):
                break
            offset_samples += step_samples

        avg_no_speech = sum(no_speech_probs) / len(no_speech_probs) if no_speech_probs else 0.0

        return {
            "text": " ".join(combined_text_parts),
            "no_speech_prob": avg_no_speech,
            "segments": combined_segments,
            "language": self.config.whisper_language,
        }
