"""Multimodal fusion pipeline orchestrator (Module 2.3).

``FusionPipeline`` orchestrates the four stages of fusion:
  1. Temporal alignment  (TemporalAligner)
  2. Metrics calculation (MetricsCalculator)
  3. Rubric mapping      (UPAORubricMapper)
  4. Explanation         (ExplanationGenerator)

The pipeline mirrors the async ``process()`` interface used by VisionPipeline
and AudioPipeline so that it can be driven uniformly by a Celery task.
"""

from __future__ import annotations

import logging
import time
from typing import Callable

from app.core.audio.data_types import AudioResult
from app.core.fusion.config import FusionConfig
from app.core.fusion.data_types import FusionResult
from app.core.fusion.explanation_generator import ExplanationGenerator
from app.core.fusion.metrics_calculator import MetricsCalculator
from app.core.fusion.rubric_mapper import UPAORubricMapper
from app.core.fusion.temporal_aligner import TemporalAligner
from app.core.vision.data_types import VisionResult

logger = logging.getLogger(__name__)


class FusionPipeline:
    """Full multimodal fusion pipeline.

    Designed to be instantiated once per analysis session and reused.
    Unlike the vision/audio pipelines there are no ML models to load/unload
    (the LLM call is made via HTTP to Ollama), so lifecycle methods are
    provided as no-ops for API compatibility.

    Parameters
    ----------
    config : FusionConfig, optional
        Pipeline-wide configuration.  Defaults to ``FusionConfig()`` when not
        provided.
    """

    def __init__(self, config: FusionConfig | None = None) -> None:
        self.config = config or FusionConfig()
        self._aligner = TemporalAligner(self.config)
        self._metrics = MetricsCalculator(self.config)
        self._mapper = UPAORubricMapper(self.config)
        self._explainer = ExplanationGenerator(self.config)

    # ------------------------------------------------------------------
    # Lifecycle (no-ops; kept for API parity with audio/vision pipelines)
    # ------------------------------------------------------------------

    def load_all(self) -> None:
        """No-op: fusion pipeline has no ML models to pre-load."""
        logger.debug("FusionPipeline.load_all() called (no-op)")

    def unload_all(self) -> None:
        """No-op: fusion pipeline has no ML models to unload."""
        logger.debug("FusionPipeline.unload_all() called (no-op)")

    def __enter__(self) -> "FusionPipeline":
        self.load_all()
        return self

    def __exit__(self, *args: object) -> None:
        self.unload_all()

    # ------------------------------------------------------------------
    # Main processing
    # ------------------------------------------------------------------

    async def process(
        self,
        vision_result: VisionResult,
        audio_result: AudioResult,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> FusionResult:
        """Run the full fusion pipeline and return a ``FusionResult``.

        Parameters
        ----------
        vision_result : VisionResult
            Output of Module 2.1 visual processing.
        audio_result : AudioResult
            Output of Module 2.2 audio processing.
        progress_callback : callable, optional
            Called with ``(fraction: float, message: str)`` as each stage
            completes.  ``fraction`` increases monotonically from 0.0 to 1.0.

        Returns
        -------
        FusionResult
        """
        start_time = time.time()

        def _progress(p: float, msg: str) -> None:
            if progress_callback is not None:
                progress_callback(p, msg)

        # ------------------------------------------------------------------
        # Stage 1 — Temporal alignment
        # ------------------------------------------------------------------
        _progress(0.0, "Starting temporal alignment")
        aligned = self._aligner.align(vision_result, audio_result)
        _progress(0.30, "Temporal alignment complete")
        logger.info(
            "Stage 1 complete: %d alignment windows, %d identity mappings",
            len(aligned.windows),
            len(aligned.speaker_to_person),
        )

        # ------------------------------------------------------------------
        # Stage 2 — Metrics calculation
        # ------------------------------------------------------------------
        _progress(0.35, "Calculating student metrics")
        student_metrics = self._metrics.calculate_student_metrics(aligned, audio_result)
        _progress(0.55, "Calculating group metrics")
        group_metrics = self._metrics.calculate_group_metrics(aligned, audio_result, student_metrics)
        _progress(0.60, "Metrics calculation complete")
        logger.info(
            "Stage 2 complete: %d students, CV=%.3f, sync=%.2f",
            group_metrics.total_students,
            group_metrics.participation_cv,
            group_metrics.turn_synchronization_score,
        )

        # ------------------------------------------------------------------
        # Stage 3 — Rubric mapping
        # ------------------------------------------------------------------
        _progress(0.65, "Mapping metrics to UPAO rubric")
        rubric_scores = self._mapper.map_group_to_rubric(group_metrics)
        _progress(0.75, "Rubric mapping complete")
        logger.info(
            "Stage 3 complete: collab=%.1f comm=%.1f resp=%.1f lead=%.1f tech=%.1f",
            rubric_scores.collaboration,
            rubric_scores.communication,
            rubric_scores.responsibility,
            rubric_scores.leadership,
            rubric_scores.technical_contribution,
        )

        # ------------------------------------------------------------------
        # Stage 4 — Explanation generation
        # ------------------------------------------------------------------
        _progress(0.80, "Generating narrative explanation")
        explanation = await self._explainer.generate(group_metrics, rubric_scores, audio_result)
        _progress(1.0, "Fusion analysis complete")
        logger.info(
            "Stage 4 complete: explanation generated by %s",
            explanation.generated_by,
        )

        processing_time = time.time() - start_time
        logger.info(
            "Fusion pipeline complete: %.1fs processing time (video duration %.1fs)",
            processing_time,
            aligned.duration_seconds,
        )

        return FusionResult(
            video_path=vision_result.video_path,
            aligned_features=aligned,
            group_metrics=group_metrics,
            rubric_scores=rubric_scores,
            explanation=explanation,
            processing_time_seconds=processing_time,
        )
