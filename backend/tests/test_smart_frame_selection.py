"""Integration tests for smart frame selection pipeline.

Tests cover KeyFrameSelector, SpeakerFaceMapper, and budget enforcement.
All tests use in-memory objects only — no GPU, no database, no network.
"""

from __future__ import annotations

import pytest

from app.core.audio.data_types import (
    Interruption,
    InterruptionType,
    KeyMoment,
    MomentType,
    SpeakerSegment,
    TranscriptSegment,
    WordTimestamp,
)
from app.core.audio.config import AudioConfig
from app.core.audio.key_frame_selector import KeyFrameSelector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**kwargs) -> AudioConfig:
    """Return an AudioConfig with optional field overrides."""
    config = AudioConfig()
    for key, value in kwargs.items():
        setattr(config, key, value)
    return config


# ---------------------------------------------------------------------------
# Test 1: KeyFrameSelector detects speech onset
# ---------------------------------------------------------------------------


def test_speech_onset_detection():
    """A speaker change triggers a SPEECH_ONSET moment."""
    config = _make_config()
    selector = KeyFrameSelector(config)

    segments = [
        SpeakerSegment(start=0.0, end=5.0, speaker_id="speaker_0"),
        SpeakerSegment(start=5.5, end=10.0, speaker_id="speaker_1"),
    ]

    moments = selector.select(segments=segments, transcript=[], interruptions=[], turns=[])
    onset_moments = [m for m in moments if m.moment_type == MomentType.SPEECH_ONSET]

    assert len(onset_moments) >= 1, "Expected at least one SPEECH_ONSET moment"
    assert any(
        m.speaker_id == "speaker_1" for m in onset_moments
    ), "speaker_1 should have an onset moment (different speaker)"


def test_speech_onset_first_segment_always_triggers():
    """The very first segment always triggers a SPEECH_ONSET."""
    config = _make_config()
    selector = KeyFrameSelector(config)

    segments = [SpeakerSegment(start=0.0, end=3.0, speaker_id="speaker_0")]
    moments = selector.select(segments=segments, transcript=[], interruptions=[], turns=[])
    onset_moments = [m for m in moments if m.moment_type == MomentType.SPEECH_ONSET]

    assert len(onset_moments) == 1
    assert onset_moments[0].speaker_id == "speaker_0"


# ---------------------------------------------------------------------------
# Test 2: Question detection
# ---------------------------------------------------------------------------


def test_question_detection():
    """A transcript segment ending with '?' triggers RECEIVING_QUESTION for the next speaker."""
    config = _make_config()
    selector = KeyFrameSelector(config)

    segments = [
        SpeakerSegment(start=0.0, end=4.0, speaker_id="speaker_0"),
        SpeakerSegment(start=4.5, end=9.0, speaker_id="speaker_1"),
    ]
    transcript = [
        TranscriptSegment(
            speaker_id="speaker_0",
            text="¿Qué piensas tú?",
            start=0.5,
            end=3.5,
            words=[],
        ),
        TranscriptSegment(
            speaker_id="speaker_1",
            text="Bueno, creo que...",
            start=4.5,
            end=8.0,
            words=[],
        ),
    ]

    moments = selector.select(segments=segments, transcript=transcript, interruptions=[], turns=[])
    q_moments = [m for m in moments if m.moment_type == MomentType.RECEIVING_QUESTION]

    assert len(q_moments) >= 1, "Expected at least one RECEIVING_QUESTION moment"


def test_question_detection_same_speaker_no_trigger():
    """A '?' segment where the next segment is the same speaker should NOT trigger."""
    config = _make_config()
    selector = KeyFrameSelector(config)

    segments = [SpeakerSegment(start=0.0, end=8.0, speaker_id="speaker_0")]
    transcript = [
        TranscriptSegment(
            speaker_id="speaker_0",
            text="Is this right?",
            start=0.5,
            end=3.5,
            words=[],
        ),
        TranscriptSegment(
            speaker_id="speaker_0",
            text="I think so.",
            start=4.0,
            end=7.0,
            words=[],
        ),
    ]

    moments = selector.select(segments=segments, transcript=transcript, interruptions=[], turns=[])
    q_moments = [m for m in moments if m.moment_type == MomentType.RECEIVING_QUESTION]

    # Both segments belong to speaker_0, so no RECEIVING_QUESTION should fire
    assert len(q_moments) == 0, "Same-speaker question should not trigger RECEIVING_QUESTION"


# ---------------------------------------------------------------------------
# Test 3: Budget enforcement
# ---------------------------------------------------------------------------


def test_budget_enforcement():
    """Per-participant and total frame limits are respected."""
    config = _make_config(
        sfs_max_frames_per_participant=3,
        sfs_max_frames_total=5,
    )
    selector = KeyFrameSelector(config)

    # Create many segments for one speaker — will generate many SPEECH_ONSET moments
    segments = [
        SpeakerSegment(start=i * 2.0, end=i * 2.0 + 1.5, speaker_id="speaker_0")
        for i in range(20)
    ]

    moments = selector.select(segments=segments, transcript=[], interruptions=[], turns=[])

    speaker_moments = [m for m in moments if m.speaker_id == "speaker_0"]
    assert len(speaker_moments) <= config.sfs_max_frames_per_participant, (
        f"Per-participant limit exceeded: {len(speaker_moments)} > "
        f"{config.sfs_max_frames_per_participant}"
    )
    assert len(moments) <= config.sfs_max_frames_total, (
        f"Total limit exceeded: {len(moments)} > {config.sfs_max_frames_total}"
    )


def test_budget_enforcement_two_speakers():
    """Budget is shared between participants."""
    config = _make_config(
        sfs_max_frames_per_participant=5,
        sfs_max_frames_total=6,
    )
    selector = KeyFrameSelector(config)

    # 10 alternating segments → many onset moments
    segments = []
    for i in range(10):
        speaker = "speaker_0" if i % 2 == 0 else "speaker_1"
        segments.append(SpeakerSegment(start=i * 2.0, end=i * 2.0 + 1.5, speaker_id=speaker))

    moments = selector.select(segments=segments, transcript=[], interruptions=[], turns=[])
    assert len(moments) <= config.sfs_max_frames_total


# ---------------------------------------------------------------------------
# Test 4: Hesitation detection
# ---------------------------------------------------------------------------


def test_hesitation_filler_detection():
    """Filler words in transcript trigger HESITATION moments."""
    config = _make_config()
    selector = KeyFrameSelector(config)

    segments = [SpeakerSegment(start=0.0, end=5.0, speaker_id="speaker_0")]
    transcript = [
        TranscriptSegment(
            speaker_id="speaker_0",
            text="eh este... no sé",
            start=0.5,
            end=4.5,
            words=[],
        ),
    ]

    moments = selector.select(segments=segments, transcript=transcript, interruptions=[], turns=[])
    hes_moments = [m for m in moments if m.moment_type == MomentType.HESITATION]

    assert len(hes_moments) >= 1, "Expected at least one HESITATION moment for filler word"


def test_hesitation_one_per_segment():
    """Multiple fillers in one segment produce only one HESITATION moment."""
    config = _make_config()
    selector = KeyFrameSelector(config)

    segments = [SpeakerSegment(start=0.0, end=5.0, speaker_id="speaker_0")]
    transcript = [
        TranscriptSegment(
            speaker_id="speaker_0",
            text="eh um uh este... no sé",
            start=0.5,
            end=4.5,
            words=[],
        ),
    ]

    moments = selector.select(segments=segments, transcript=transcript, interruptions=[], turns=[])
    hes_moments = [
        m
        for m in moments
        if m.moment_type == MomentType.HESITATION and m.speaker_id == "speaker_0"
    ]
    # One per segment (filler detection breaks after first match per segment)
    # Plus possibly one from pause detection — but filler detection itself: max 1
    filler_hes = [m for m in hes_moments if m.timestamp_ms == 0.5 * 1000.0]
    assert len(filler_hes) <= 1, "Filler detection should emit at most one moment per segment"


def test_no_hesitation_without_fillers():
    """Clean transcript with no fillers should produce no HESITATION moments."""
    config = _make_config()
    selector = KeyFrameSelector(config)

    segments = [SpeakerSegment(start=0.0, end=5.0, speaker_id="speaker_0")]
    transcript = [
        TranscriptSegment(
            speaker_id="speaker_0",
            text="The answer is clearly correct.",
            start=0.5,
            end=4.5,
            words=[],
        ),
    ]

    moments = selector.select(segments=segments, transcript=transcript, interruptions=[], turns=[])
    hes_moments = [m for m in moments if m.moment_type == MomentType.HESITATION]
    assert len(hes_moments) == 0


# ---------------------------------------------------------------------------
# Test 5: SpeakerFaceMapper
# ---------------------------------------------------------------------------


from app.core.vision.speaker_face_mapper import SpeakerFaceMapper


def test_speaker_face_mapper_basic():
    """Two speakers map to two different person_ids."""
    mapper = SpeakerFaceMapper(min_confidence=0.5, min_frames=2)

    segments = [
        SpeakerSegment(start=0.0, end=5.0, speaker_id="speaker_0"),
        SpeakerSegment(start=5.5, end=10.0, speaker_id="speaker_1"),
    ]
    frame_data = [
        {"timestamp_s": 1.0, "person_ids": ["person_0", "person_1"]},
        {"timestamp_s": 2.0, "person_ids": ["person_0", "person_1"]},
        {"timestamp_s": 6.0, "person_ids": ["person_0", "person_1"]},
        {"timestamp_s": 7.0, "person_ids": ["person_0", "person_1"]},
    ]

    mappings = mapper.build_mapping(segments, frame_data)

    # Both speakers should have a mapping
    assert len(mappings) == 2, f"Expected 2 mappings, got {len(mappings)}"

    mapped_persons = {m.speaker_id: m.person_id for m in mappings}

    # speaker_0 saw frames at 1.0 and 2.0 → should map to person_0 (appears in both)
    # speaker_1 saw frames at 6.0 and 7.0 → should map to person_1 (after greedy assignment)
    # The greedy algorithm ensures different speakers get different person_ids
    assert mapped_persons.get("speaker_0") != mapped_persons.get("speaker_1"), (
        "Two speakers should map to different person_ids"
    )


def test_speaker_face_mapper_no_overlap():
    """Speaker with no overlapping frames gets an uncertain mapping."""
    mapper = SpeakerFaceMapper(min_confidence=0.5, min_frames=2)

    segments = [
        SpeakerSegment(start=0.0, end=5.0, speaker_id="speaker_0"),
        SpeakerSegment(start=100.0, end=105.0, speaker_id="speaker_1"),  # no frames here
    ]
    frame_data = [
        {"timestamp_s": 1.0, "person_ids": ["person_0"]},
        {"timestamp_s": 2.0, "person_ids": ["person_0"]},
        {"timestamp_s": 3.0, "person_ids": ["person_0"]},
    ]

    mappings = mapper.build_mapping(segments, frame_data)
    speaker_1_mapping = next((m for m in mappings if m.speaker_id == "speaker_1"), None)

    assert speaker_1_mapping is not None, "speaker_1 should still appear in mappings"
    assert speaker_1_mapping.uncertain is True, "No-overlap mapping should be uncertain"


def test_speaker_face_mapper_single_speaker():
    """Single speaker reliably maps to a single person_id."""
    mapper = SpeakerFaceMapper(min_confidence=0.3, min_frames=1)

    segments = [SpeakerSegment(start=0.0, end=10.0, speaker_id="speaker_0")]
    frame_data = [
        {"timestamp_s": i * 1.0, "person_ids": ["person_0"]}
        for i in range(8)
    ]

    mappings = mapper.build_mapping(segments, frame_data)
    assert len(mappings) == 1
    assert mappings[0].speaker_id == "speaker_0"
    assert mappings[0].person_id == "person_0"
    assert mappings[0].confidence > 0.0


# ---------------------------------------------------------------------------
# Test 6: Interruption detection
# ---------------------------------------------------------------------------


def test_interruption_received_detection():
    """Disruptive interruptions produce INTERRUPTION_RECEIVED moments."""
    config = _make_config()
    selector = KeyFrameSelector(config)

    class FakeInterruption:
        def __init__(self, time, interrupter_id, interrupted_id, interruption_type):
            self.time = time
            self.interrupter_id = interrupter_id
            self.interrupted_id = interrupted_id
            self.interruption_type = interruption_type

    interruptions = [
        FakeInterruption(
            time=5.0,
            interrupter_id="speaker_1",
            interrupted_id="speaker_0",
            interruption_type="disruptive",
        )
    ]

    moments = selector.select(
        segments=[], transcript=[], interruptions=interruptions, turns=[]
    )
    int_moments = [m for m in moments if m.moment_type == MomentType.INTERRUPTION_RECEIVED]

    assert len(int_moments) >= 1
    assert int_moments[0].speaker_id == "speaker_0"
    assert abs(int_moments[0].timestamp_ms - 5000.0) < 1.0


def test_back_channel_detection():
    """Back channel interruptions produce BACK_CHANNEL moments."""
    config = _make_config()
    selector = KeyFrameSelector(config)

    class FakeInterruption:
        def __init__(self, time, interrupter_id, interrupted_id, interruption_type):
            self.time = time
            self.interrupter_id = interrupter_id
            self.interrupted_id = interrupted_id
            self.interruption_type = interruption_type

    interruptions = [
        FakeInterruption(
            time=3.5,
            interrupter_id="speaker_1",
            interrupted_id="speaker_0",
            interruption_type="back_channel",
        )
    ]

    moments = selector.select(
        segments=[], transcript=[], interruptions=interruptions, turns=[]
    )
    bc_moments = [m for m in moments if m.moment_type == MomentType.BACK_CHANNEL]

    assert len(bc_moments) >= 1
    assert bc_moments[0].speaker_id == "speaker_1"


# ---------------------------------------------------------------------------
# Test 7: Long turn end detection
# ---------------------------------------------------------------------------


def test_long_turn_end_detection():
    """Turns exceeding the threshold trigger END_OF_LONG_TURN moments."""
    config = _make_config(sfs_long_turn_threshold_s=10.0)
    selector = KeyFrameSelector(config)

    class FakeTurn:
        def __init__(self, start, end, speaker_id):
            self.start = start
            self.end = end
            self.speaker_id = speaker_id
            self.duration = end - start

    turns = [
        FakeTurn(start=0.0, end=20.0, speaker_id="speaker_0"),  # 20s > 10s threshold
        FakeTurn(start=25.0, end=30.0, speaker_id="speaker_1"),  # 5s < threshold
    ]

    moments = selector.select(segments=[], transcript=[], interruptions=[], turns=turns)
    lt_moments = [m for m in moments if m.moment_type == MomentType.END_OF_LONG_TURN]

    assert len(lt_moments) == 1
    assert lt_moments[0].speaker_id == "speaker_0"


def test_short_turn_no_long_turn_end():
    """Turns below the threshold do NOT trigger END_OF_LONG_TURN."""
    config = _make_config(sfs_long_turn_threshold_s=15.0)
    selector = KeyFrameSelector(config)

    class FakeTurn:
        def __init__(self, start, end, speaker_id):
            self.start = start
            self.end = end
            self.speaker_id = speaker_id
            self.duration = end - start

    turns = [FakeTurn(start=0.0, end=5.0, speaker_id="speaker_0")]
    moments = selector.select(segments=[], transcript=[], interruptions=[], turns=turns)
    lt_moments = [m for m in moments if m.moment_type == MomentType.END_OF_LONG_TURN]

    assert len(lt_moments) == 0


# ---------------------------------------------------------------------------
# Test 8: KeyMoment serialisation round-trip
# ---------------------------------------------------------------------------


def test_key_moment_serialisation():
    """KeyMoment values survive a dict round-trip (for Celery transport)."""
    km = KeyMoment(
        timestamp_ms=12345.0,
        speaker_id="speaker_0",
        moment_type=MomentType.SPEECH_ONSET,
        priority=1,
        context_text="hello world",
    )

    km_dict = {
        "timestamp_ms": km.timestamp_ms,
        "speaker_id": km.speaker_id,
        "moment_type": km.moment_type.value,
        "priority": km.priority,
        "context_text": km.context_text,
    }

    # Deserialise
    restored = KeyMoment(
        timestamp_ms=float(km_dict["timestamp_ms"]),
        speaker_id=km_dict["speaker_id"],
        moment_type=MomentType(km_dict["moment_type"]),
        priority=int(km_dict["priority"]),
        context_text=km_dict["context_text"],
    )

    assert restored.timestamp_ms == km.timestamp_ms
    assert restored.speaker_id == km.speaker_id
    assert restored.moment_type == km.moment_type
    assert restored.priority == km.priority
    assert restored.context_text == km.context_text
