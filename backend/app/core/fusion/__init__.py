"""Multimodal fusion module for SMATC-UPAO (Module 2.3).

Exports the main pipeline class and all public data types so that service,
task, and API layers can import from this package without knowing the internal
sub-module layout.
"""

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

__all__ = [
    # Config
    "FusionConfig",
    # Data types
    "TemporalWindow",
    "AlignedFeatures",
    "StudentMetrics",
    "GroupMetrics",
    "RubricScores",
    "GroupRubricScores",
    "ExplanationResult",
    "FusionResult",
    # Components
    "TemporalAligner",
    "MetricsCalculator",
    "UPAORubricMapper",
    "ExplanationGenerator",
    # Pipeline
    "FusionPipeline",
]
