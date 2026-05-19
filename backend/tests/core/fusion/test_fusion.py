"""Unit tests for Module 2.3 — Multimodal Fusion.

Each component is tested in isolation with mock/stub vision and audio results.
No real ML models or Ollama are required; HTTP calls are patched via unittest.mock.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import openai
import pytest

from app.core.audio.data_types import (
    AudioResult,
    AudioSessionMetrics,
    Interruption,
    InterruptionType,
    SpeakerMetrics,
    SpeakerSegment,
    SpeakerTurn,
    TranscriptSegment,
)
from app.core.fusion.config import FusionConfig
from app.core.fusion.data_types import (
    AlignedFeatures,
    ExplanationResult,
    FusionResult,
    GroupMetrics,
    GroupRubricScores,
    RubricScores,
    StudentMetrics,
    TemporalWindow,
)
from app.core.fusion.explanation_generator import ExplanationGenerator
from app.core.fusion.fusion_pipeline import FusionPipeline
from app.core.fusion.metrics_calculator import MetricsCalculator
from app.core.fusion.rubric_mapper import UPAORubricMapper
from app.core.fusion.temporal_aligner import TemporalAligner
from app.core.vision.data_types import (
    FrameResult,
    GazeData,
    PersonFrame,
    SessionMetrics,
    VisionResult,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_vision_result(duration: float = 10.0) -> VisionResult:
    """Minimal VisionResult with two tracked persons."""
    frames = []
    for i in range(20):
        ts = i * 0.5
        persons = [
            PersonFrame(
                person_id="person_0",
                bbox=(0, 0, 100, 100),
                gaze=GazeData(
                    direction=(0.0, 0.0, 0.0),
                    is_looking_at_camera=True,
                    confidence=0.9,
                    category="camera",
                ),
            ),
            PersonFrame(
                person_id="person_1",
                bbox=(200, 0, 100, 100),
                gaze=GazeData(
                    direction=(30.0, 0.0, 0.0),
                    is_looking_at_camera=False,
                    confidence=0.8,
                    category="away",
                ),
            ),
        ]
        frames.append(FrameResult(frame_number=i, timestamp=ts, persons=persons))

    return VisionResult(
        video_path="/tmp/test.mp4",
        total_frames=20,
        fps_processed=2.0,
        duration_seconds=duration,
        frames=frames,
        processing_time_seconds=1.0,
    )


def _make_audio_result(duration: float = 10.0) -> AudioResult:
    """Minimal AudioResult with two speakers and one interruption."""
    segments = [
        SpeakerSegment(start=0.0, end=3.0, speaker_id="speaker_0"),
        SpeakerSegment(start=3.0, end=7.0, speaker_id="speaker_1"),
        SpeakerSegment(start=7.0, end=9.5, speaker_id="speaker_0"),
    ]
    turns = [
        SpeakerTurn(start=0.0, end=3.0, speaker_id="speaker_0", duration=3.0),
        SpeakerTurn(start=3.0, end=7.0, speaker_id="speaker_1", duration=4.0),
        SpeakerTurn(start=7.0, end=9.5, speaker_id="speaker_0", duration=2.5),
    ]
    interruptions = [
        Interruption(
            time=6.8,
            interrupter_id="speaker_0",
            interrupted_id="speaker_1",
            overlap_duration=0.2,
            interruption_type=InterruptionType.DISRUPTIVE,
        )
    ]
    per_speaker = [
        SpeakerMetrics(
            speaker_id="speaker_0",
            speaking_time_seconds=5.5,
            turn_count=2,
            interruption_count=1,
            interrupted_count=0,
            avg_turn_duration=2.75,
            participation_ratio=0.578,
        ),
        SpeakerMetrics(
            speaker_id="speaker_1",
            speaking_time_seconds=4.0,
            turn_count=1,
            interruption_count=0,
            interrupted_count=1,
            avg_turn_duration=4.0,
            participation_ratio=0.421,
        ),
    ]
    session_metrics = AudioSessionMetrics(
        total_speakers=2,
        duration=duration,
        total_speaking_time=9.5,
        silence_ratio=0.05,
        participation_cv=0.15,
        turn_alternation_rate=0.67,
        per_speaker_metrics=per_speaker,
    )
    return AudioResult(
        video_path="/tmp/test.mp4",
        duration_seconds=duration,
        sample_rate=16000,
        segments=segments,
        turns=turns,
        transcripts=[],
        interruptions=interruptions,
        processing_time_seconds=2.0,
        session_metrics=session_metrics,
    )


def _make_student_metrics(n: int = 2) -> list[StudentMetrics]:
    return [
        StudentMetrics(
            student_id=f"speaker_{i}",
            speaking_time_seconds=5.0 if i == 0 else 4.0,
            turn_count=2 if i == 0 else 1,
            interruption_count=1 if i == 0 else 0,
            interrupted_count=0 if i == 0 else 1,
            participation_ratio=0.55 if i == 0 else 0.45,
            gaze_contact_percentage=80.0 if i == 0 else 40.0,
            avg_body_orientation=5.0,
            dominant_emotion="neutral",
            initiation_count=1,
            back_channel_count=0,
        )
        for i in range(n)
    ]


def _make_group_metrics() -> GroupMetrics:
    student_metrics = _make_student_metrics(2)
    return GroupMetrics(
        total_students=2,
        duration_seconds=10.0,
        total_speaking_time=8.0,
        participation_cv=0.10,
        disruptive_interruption_rate=0.10,
        turn_synchronization_score=0.80,
        avg_gaze_contact_percentage=60.0,
        silence_ratio=0.05,
        per_student_metrics=student_metrics,
    )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestFusionConfig:
    def test_defaults_are_sensible(self) -> None:
        config = FusionConfig()
        assert config.temporal_window_ms == 500
        assert config.min_overlap_ratio == 0.3
        assert config.cv_threshold == 0.3
        assert config.llm_model == "gpt-4o-mini"
        assert config.enable_explanation is True

    def test_to_dict_round_trips(self) -> None:
        config = FusionConfig()
        d = config.to_dict()
        assert d["temporal_window_ms"] == 500
        assert "rubric_weights" in d


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class TestDataTypes:
    def test_temporal_window_to_dict(self) -> None:
        w = TemporalWindow(
            start=0.0,
            end=0.5,
            speaker_id="s0",
            person_id="p0",
            gaze_at_camera=True,
            overlap_ratio=0.8,
        )
        d = w.to_dict()
        assert d["speaker_id"] == "s0"
        assert d["gaze_at_camera"] is True

    def test_student_metrics_to_dict(self) -> None:
        m = _make_student_metrics(1)[0]
        d = m.to_dict()
        assert d["student_id"] == "speaker_0"
        assert "participation_ratio" in d

    def test_fusion_result_serialization(self) -> None:
        aligned = AlignedFeatures(
            video_path="/tmp/test.mp4",
            duration_seconds=10.0,
            windows=[],
            speaker_to_person={"speaker_0": "person_0"},
            person_to_speaker={"person_0": "speaker_0"},
        )
        gm = _make_group_metrics()
        rubric = GroupRubricScores(
            collaboration=4.0,
            communication=3.5,
            responsibility=4.0,
            leadership=3.0,
            technical_contribution=3.5,
            per_student_scores=[
                RubricScores(
                    student_id="speaker_0",
                    collaboration=4.0,
                    communication=3.5,
                    responsibility=4.0,
                    leadership=3.0,
                    technical_contribution=3.5,
                )
            ],
        )
        explanation = ExplanationResult(
            narrative_text="Good collaboration.",
            strengths=["Equitable participation"],
            improvements=["More eye contact"],
            recommendations=["Use a moderator"],
            generated_by="rule_based",
        )
        result = FusionResult(
            video_path="/tmp/test.mp4",
            aligned_features=aligned,
            group_metrics=gm,
            rubric_scores=rubric,
            explanation=explanation,
            processing_time_seconds=3.0,
        )
        d = result.to_dict()
        assert d["video_path"] == "/tmp/test.mp4"
        assert "group_metrics" in d
        assert "rubric_scores" in d
        assert "explanation" in d


# ---------------------------------------------------------------------------
# TemporalAligner
# ---------------------------------------------------------------------------


class TestTemporalAligner:
    def test_align_produces_aligned_features(self) -> None:
        config = FusionConfig(temporal_window_ms=500)
        aligner = TemporalAligner(config)
        vision = _make_vision_result()
        audio = _make_audio_result()

        aligned = aligner.align(vision, audio)

        assert isinstance(aligned, AlignedFeatures)
        assert aligned.duration_seconds > 0
        assert len(aligned.windows) > 0

    def test_windows_cover_full_duration(self) -> None:
        config = FusionConfig(temporal_window_ms=500)
        aligner = TemporalAligner(config)
        vision = _make_vision_result(duration=10.0)
        audio = _make_audio_result(duration=10.0)

        aligned = aligner.align(vision, audio)

        # First window starts at 0, last window ends at duration
        assert aligned.windows[0].start == pytest.approx(0.0)
        assert aligned.windows[-1].end == pytest.approx(10.0)

    def test_speaker_to_person_mapping_is_built(self) -> None:
        config = FusionConfig(temporal_window_ms=500, min_overlap_ratio=0.1)
        aligner = TemporalAligner(config)
        vision = _make_vision_result()
        audio = _make_audio_result()

        aligned = aligner.align(vision, audio)

        # At least some mapping should be inferred from co-occurrence
        assert isinstance(aligned.speaker_to_person, dict)

    def test_silent_windows_have_no_speaker(self) -> None:
        config = FusionConfig(temporal_window_ms=500, min_overlap_ratio=0.3)
        aligner = TemporalAligner(config)
        # Audio with a gap of silence at the end
        audio = _make_audio_result(duration=10.0)
        vision = _make_vision_result(duration=10.0)

        aligned = aligner.align(vision, audio)

        # Windows from 9.5–10.0 should have no speaker
        late_windows = [w for w in aligned.windows if w.start >= 9.5]
        assert any(w.speaker_id is None for w in late_windows)


# ---------------------------------------------------------------------------
# MetricsCalculator
# ---------------------------------------------------------------------------


class TestMetricsCalculator:
    def test_participation_cv_two_equal_speakers(self) -> None:
        calc = MetricsCalculator(FusionConfig())
        students = [
            StudentMetrics(
                student_id="s0",
                speaking_time_seconds=5.0,
                turn_count=2,
                interruption_count=0,
                interrupted_count=0,
                participation_ratio=0.5,
                gaze_contact_percentage=50.0,
                avg_body_orientation=0.0,
                dominant_emotion="neutral",
                initiation_count=1,
                back_channel_count=0,
            ),
            StudentMetrics(
                student_id="s1",
                speaking_time_seconds=5.0,
                turn_count=2,
                interruption_count=0,
                interrupted_count=0,
                participation_ratio=0.5,
                gaze_contact_percentage=50.0,
                avg_body_orientation=0.0,
                dominant_emotion="neutral",
                initiation_count=1,
                back_channel_count=0,
            ),
        ]
        cv = calc.calculate_participation_cv(students)
        assert cv == pytest.approx(0.0)

    def test_participation_cv_unequal_speakers(self) -> None:
        calc = MetricsCalculator(FusionConfig())
        students = [
            StudentMetrics(
                student_id="s0",
                speaking_time_seconds=9.0,
                turn_count=5,
                interruption_count=0,
                interrupted_count=0,
                participation_ratio=0.9,
                gaze_contact_percentage=50.0,
                avg_body_orientation=0.0,
                dominant_emotion="neutral",
                initiation_count=3,
                back_channel_count=0,
            ),
            StudentMetrics(
                student_id="s1",
                speaking_time_seconds=1.0,
                turn_count=1,
                interruption_count=0,
                interrupted_count=0,
                participation_ratio=0.1,
                gaze_contact_percentage=50.0,
                avg_body_orientation=0.0,
                dominant_emotion="neutral",
                initiation_count=0,
                back_channel_count=0,
            ),
        ]
        cv = calc.calculate_participation_cv(students)
        assert cv > 0.3  # above threshold = inequitable

    def test_disruptive_interruption_rate(self) -> None:
        calc = MetricsCalculator(FusionConfig())
        audio = _make_audio_result()
        rate = calc.calculate_disruptive_interruption_rate(audio)
        # 1 disruptive interruption / 3 turns
        assert rate == pytest.approx(1 / 3, abs=0.01)

    def test_turn_synchronization_perfect(self) -> None:
        calc = MetricsCalculator(FusionConfig())
        # All turns have smooth gaps (0–1s)
        audio = AudioResult(
            video_path="",
            duration_seconds=10.0,
            sample_rate=16000,
            turns=[
                SpeakerTurn(start=0.0, end=3.0, speaker_id="s0", duration=3.0),
                SpeakerTurn(start=3.2, end=6.0, speaker_id="s1", duration=2.8),
                SpeakerTurn(start=6.5, end=9.0, speaker_id="s0", duration=2.5),
            ],
        )
        score = calc.calculate_turn_synchronization_score(audio)
        assert score == pytest.approx(1.0)

    def test_student_metrics_are_computed(self) -> None:
        config = FusionConfig()
        calc = MetricsCalculator(config)
        audio = _make_audio_result()
        aligner = TemporalAligner(config)
        vision = _make_vision_result()
        aligned = aligner.align(vision, audio)

        metrics = calc.calculate_student_metrics(aligned, audio)

        assert len(metrics) >= 1
        for m in metrics:
            assert m.turn_count >= 0
            assert 0.0 <= m.participation_ratio <= 1.0

    def test_group_metrics_aggregate(self) -> None:
        config = FusionConfig()
        calc = MetricsCalculator(config)
        audio = _make_audio_result()
        aligner = TemporalAligner(config)
        vision = _make_vision_result()
        aligned = aligner.align(vision, audio)
        students = calc.calculate_student_metrics(aligned, audio)

        gm = calc.calculate_group_metrics(aligned, audio, students)

        assert isinstance(gm, GroupMetrics)
        assert gm.total_students == len(students)
        assert 0.0 <= gm.silence_ratio <= 1.0
        assert 0.0 <= gm.turn_synchronization_score <= 1.0


# ---------------------------------------------------------------------------
# UPAORubricMapper
# ---------------------------------------------------------------------------


class TestUPAORubricMapper:
    def test_map_student_scores_are_in_range(self) -> None:
        mapper = UPAORubricMapper(FusionConfig())
        gm = _make_group_metrics()
        student = gm.per_student_metrics[0]

        scores = mapper.map_student_to_rubric(student, gm)

        assert 0.0 <= scores.collaboration <= 5.0
        assert 0.0 <= scores.communication <= 5.0
        assert 0.0 <= scores.responsibility <= 5.0
        assert 0.0 <= scores.leadership <= 5.0
        assert 0.0 <= scores.technical_contribution <= 5.0

    def test_map_group_rubric_returns_group_scores(self) -> None:
        mapper = UPAORubricMapper(FusionConfig())
        gm = _make_group_metrics()

        group_scores = mapper.map_group_to_rubric(gm)

        assert isinstance(group_scores, GroupRubricScores)
        assert len(group_scores.per_student_scores) == 2
        assert 0.0 <= group_scores.collaboration <= 5.0

    def test_equitable_group_gets_high_collaboration(self) -> None:
        mapper = UPAORubricMapper(FusionConfig())
        students = [
            StudentMetrics(
                student_id=f"s{i}",
                speaking_time_seconds=5.0,
                turn_count=3,
                interruption_count=0,
                interrupted_count=0,
                participation_ratio=0.5,
                gaze_contact_percentage=90.0,
                avg_body_orientation=0.0,
                dominant_emotion="neutral",
                initiation_count=1,
                back_channel_count=0,
            )
            for i in range(2)
        ]
        gm = GroupMetrics(
            total_students=2,
            duration_seconds=10.0,
            total_speaking_time=8.0,
            participation_cv=0.0,  # perfect equity
            disruptive_interruption_rate=0.0,
            turn_synchronization_score=1.0,
            avg_gaze_contact_percentage=90.0,
            silence_ratio=0.0,
            per_student_metrics=students,
        )
        group_scores = mapper.map_group_to_rubric(gm)
        assert group_scores.collaboration >= 4.0


# ---------------------------------------------------------------------------
# ExplanationGenerator
# ---------------------------------------------------------------------------


class TestExplanationGenerator:
    def test_rule_based_fallback_returns_result(self) -> None:
        gen = ExplanationGenerator(FusionConfig(enable_explanation=False))
        gm = _make_group_metrics()
        mapper = UPAORubricMapper(FusionConfig())
        rubric = mapper.map_group_to_rubric(gm)

        result = asyncio.run(gen.generate(gm, rubric))

        assert isinstance(result, ExplanationResult)
        assert result.generated_by == "rule_based"
        assert len(result.narrative_text) > 10

    def test_rule_based_has_strengths_and_improvements(self) -> None:
        gen = ExplanationGenerator(FusionConfig(enable_explanation=False))
        gm = _make_group_metrics()
        mapper = UPAORubricMapper(FusionConfig())
        rubric = mapper.map_group_to_rubric(gm)

        result = asyncio.run(gen.generate(gm, rubric))

        assert len(result.strengths) > 0
        assert len(result.recommendations) > 0

    @patch("app.core.fusion.explanation_generator.AsyncOpenAI")
    def test_llm_path_parses_response(self, mock_openai_cls) -> None:
        """Verify the LLM path parses a well-formed response."""
        raw_text = (
            "NARRATIVA:\nEl grupo mostró buena coordinación.\n\n"
            "FORTALEZAS:\n- Participación equilibrada\n- Buen contacto visual\n\n"
            "ÁREAS DE MEJORA:\n- Reducir interrupciones\n\n"
            "RECOMENDACIONES:\n- Usar un moderador\n- Practicar escucha activa"
        )

        mock_message = MagicMock()
        mock_message.content = raw_text
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)
        mock_openai_cls.return_value = mock_client

        config = FusionConfig(enable_explanation=True)
        gen = ExplanationGenerator(config)
        gm = _make_group_metrics()
        mapper = UPAORubricMapper(FusionConfig())
        rubric = mapper.map_group_to_rubric(gm)

        result = asyncio.run(gen._llm_explanation(gm, rubric))

        assert result.generated_by == "openai"
        assert len(result.strengths) >= 1
        assert "coordinación" in result.narrative_text

    @patch("app.core.fusion.explanation_generator.AsyncOpenAI")
    def test_llm_failure_falls_back_to_rule_based(self, mock_openai_cls) -> None:
        """On API error, generate() should silently fall back."""
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=openai.OpenAIError("connection refused")
        )
        mock_openai_cls.return_value = mock_client

        config = FusionConfig(enable_explanation=True)
        gen = ExplanationGenerator(config)
        gm = _make_group_metrics()
        mapper = UPAORubricMapper(FusionConfig())
        rubric = mapper.map_group_to_rubric(gm)

        result = asyncio.run(gen.generate(gm, rubric))

        assert result.generated_by == "rule_based"


# ---------------------------------------------------------------------------
# FusionPipeline
# ---------------------------------------------------------------------------


class TestFusionPipeline:
    def test_context_manager_is_noop(self) -> None:
        """Pipeline context manager should not raise even without models."""
        pipeline = FusionPipeline()
        with pipeline as p:
            assert p is pipeline

    def test_process_returns_fusion_result(self) -> None:
        """process() should orchestrate all stages and return a FusionResult."""
        config = FusionConfig(enable_explanation=False)
        pipeline = FusionPipeline(config)

        vision = _make_vision_result()
        audio = _make_audio_result()

        result = asyncio.run(pipeline.process(vision, audio))

        assert isinstance(result, FusionResult)
        assert result.video_path == "/tmp/test.mp4"
        # processing_time_seconds may be 0.0 on fast machines; just check type
        assert isinstance(result.processing_time_seconds, float)

    def test_process_group_metrics_populated(self) -> None:
        config = FusionConfig(enable_explanation=False)
        pipeline = FusionPipeline(config)

        result = asyncio.run(
            pipeline.process(_make_vision_result(), _make_audio_result())
        )

        gm = result.group_metrics
        assert gm.total_students >= 1
        assert 0.0 <= gm.participation_cv

    def test_process_rubric_scores_populated(self) -> None:
        config = FusionConfig(enable_explanation=False)
        pipeline = FusionPipeline(config)

        result = asyncio.run(
            pipeline.process(_make_vision_result(), _make_audio_result())
        )

        rs = result.rubric_scores
        assert 0.0 <= rs.collaboration <= 5.0
        assert 0.0 <= rs.communication <= 5.0

    def test_process_explanation_is_rule_based_when_disabled(self) -> None:
        config = FusionConfig(enable_explanation=False)
        pipeline = FusionPipeline(config)

        result = asyncio.run(
            pipeline.process(_make_vision_result(), _make_audio_result())
        )

        assert result.explanation.generated_by == "rule_based"

    def test_process_progress_callback_called(self) -> None:
        config = FusionConfig(enable_explanation=False)
        pipeline = FusionPipeline(config)

        progress_values: list[float] = []

        def cb(p: float, msg: str) -> None:
            progress_values.append(p)

        asyncio.run(
            pipeline.process(
                _make_vision_result(), _make_audio_result(), progress_callback=cb
            )
        )

        assert len(progress_values) > 0
        assert progress_values[0] == pytest.approx(0.0)
        assert progress_values[-1] == pytest.approx(1.0)

    def test_to_dict_is_json_serializable(self) -> None:
        """FusionResult.to_dict() should produce a JSON-serializable dict."""
        import json

        config = FusionConfig(enable_explanation=False)
        pipeline = FusionPipeline(config)

        result = asyncio.run(
            pipeline.process(_make_vision_result(), _make_audio_result())
        )
        d = result.to_dict()

        # Should not raise
        json.dumps(d)
