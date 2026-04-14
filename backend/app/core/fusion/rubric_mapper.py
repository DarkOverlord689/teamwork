"""UPAO rubric mapping for the multimodal fusion pipeline (Module 2.3).

``UPAORubricMapper`` converts raw collaboration metrics into the five UPAO
rubric dimensions (collaboration, communication, responsibility, leadership,
technical_contribution) on a 0–20 scale.
"""

from __future__ import annotations

import logging
from typing import List

from app.core.fusion.config import FusionConfig
from app.core.fusion.data_types import (
    GroupMetrics,
    GroupRubricScores,
    RubricScores,
    StudentMetrics,
)

logger = logging.getLogger(__name__)

# UPAO technical vocabulary keywords (Spanish academic/engineering terms)
_TECHNICAL_KEYWORDS = {
    "algoritmo", "función", "variable", "clase", "módulo", "interfaz",
    "base de datos", "arquitectura", "requisito", "diseño", "implementación",
    "prueba", "depuración", "repositorio", "versión", "sprint", "iteración",
    "diagrama", "diagrama de flujo", "caso de uso", "api", "endpoint",
    "autenticación", "autorización", "deployment", "infraestructura",
}


def _clamp(value: float, lo: float = 0.0, hi: float = 20.0) -> float:
    """Clamp a floating-point value to [lo, hi]."""
    return max(lo, min(hi, value))


class UPAORubricMapper:
    """Map collaboration metrics to UPAO rubric scores (0–20 scale).

    Parameters
    ----------
    config : FusionConfig
        Fusion pipeline configuration.
    """

    def __init__(self, config: FusionConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def map_student_to_rubric(
        self,
        student: StudentMetrics,
        group_metrics: GroupMetrics,
    ) -> RubricScores:
        """Map a single student's metrics to UPAO rubric scores.

        Parameters
        ----------
        student : StudentMetrics
        group_metrics : GroupMetrics
            Group-level context needed for relative scoring.

        Returns
        -------
        RubricScores
        """
        total_students = max(group_metrics.total_students, 1)
        equal_share = 1.0 / total_students  # ideal participation share

        collaboration = self._score_collaboration(student, group_metrics)
        communication = self._score_communication(student, group_metrics)
        responsibility = self._score_responsibility(student, equal_share)
        leadership = self._score_leadership(student, group_metrics)
        technical = self._score_technical_contribution(student, group_metrics)

        return RubricScores(
            student_id=student.student_id,
            collaboration=_clamp(collaboration),
            communication=_clamp(communication),
            responsibility=_clamp(responsibility),
            leadership=_clamp(leadership),
            technical_contribution=_clamp(technical),
        )

    def map_group_to_rubric(
        self,
        group_metrics: GroupMetrics,
    ) -> GroupRubricScores:
        """Map group metrics to group-level UPAO rubric scores.

        Per-student scores are computed first, then group scores are the mean
        across all students.

        Parameters
        ----------
        group_metrics : GroupMetrics

        Returns
        -------
        GroupRubricScores
        """
        per_student: List[RubricScores] = [
            self.map_student_to_rubric(sm, group_metrics)
            for sm in group_metrics.per_student_metrics
        ]

        n = len(per_student) or 1
        avg = lambda dim: sum(getattr(s, dim) for s in per_student) / n  # noqa: E731

        return GroupRubricScores(
            collaboration=_clamp(avg("collaboration")),
            communication=_clamp(avg("communication")),
            responsibility=_clamp(avg("responsibility")),
            leadership=_clamp(avg("leadership")),
            technical_contribution=_clamp(avg("technical_contribution")),
            per_student_scores=per_student,
        )

    # ------------------------------------------------------------------
    # Dimension scoring helpers
    # ------------------------------------------------------------------

    def _score_collaboration(
        self, student: StudentMetrics, group: GroupMetrics
    ) -> float:
        """Collaboration: turn equity + gaze contact.

        High score = low group CV + student gazes at peers/camera often.
        """
        weights = self.config.rubric_weights.get("collaboration", {})
        w_equity = weights.get("turn_equity", 0.5)
        w_gaze = weights.get("gaze_contact", 0.5)

        # Turn equity: invert CV (0 CV → full score; CV ≥ 1 → 0)
        equity_score = max(0.0, 1.0 - group.participation_cv) * 20.0

        # Gaze contact: 100% gaze → 20; 0% → 0
        gaze_score = (student.gaze_contact_percentage / 100.0) * 20.0

        return w_equity * equity_score + w_gaze * gaze_score

    def _score_communication(
        self, student: StudentMetrics, group: GroupMetrics
    ) -> float:
        """Communication: low interruption rate + high turn synchronization."""
        weights = self.config.rubric_weights.get("communication", {})
        w_intr = weights.get("low_interruption_rate", 0.5)
        w_sync = weights.get("turn_synchronization", 0.5)

        # Low interruption: 0% rate → 20; 100% rate → 0
        intr_score = (1.0 - group.disruptive_interruption_rate) * 20.0

        # Turn synchronization: 1.0 → 20; 0.0 → 0
        sync_score = group.turn_synchronization_score * 20.0

        return w_intr * intr_score + w_sync * sync_score

    def _score_responsibility(
        self, student: StudentMetrics, equal_share: float
    ) -> float:
        """Responsibility: participation close to equal share.

        Students who carry their expected weight get a higher score.
        """
        weights = self.config.rubric_weights.get("responsibility", {})
        w_consist = weights.get("participation_consistency", 0.6)
        w_cv = weights.get("cv_penalty", 0.4)

        # Consistency: ratio of actual to expected participation, capped at 1
        if equal_share > 0:
            ratio = min(student.participation_ratio / equal_share, 1.0)
        else:
            ratio = 1.0

        consist_score = ratio * 20.0

        # CV penalty: the individual deviation from equal share
        deviation = abs(student.participation_ratio - equal_share)
        cv_score = max(0.0, 1.0 - deviation * 5.0) * 20.0

        return w_consist * consist_score + w_cv * cv_score

    def _score_leadership(
        self, student: StudentMetrics, group: GroupMetrics
    ) -> float:
        """Leadership: speaking time share + topic initiation rate."""
        weights = self.config.rubric_weights.get("leadership", {})
        w_share = weights.get("speaking_share", 0.4)
        w_init = weights.get("initiation_rate", 0.6)

        # Speaking share relative to equal share
        total_students = max(group.total_students, 1)
        equal_share = 1.0 / total_students
        share_ratio = min(student.participation_ratio / equal_share, 2.0) / 2.0
        share_score = share_ratio * 20.0

        # Initiation rate: initiations per turn; cap at 0.5 (leaders start ~half turns)
        if student.turn_count > 0:
            init_ratio = min(student.initiation_count / student.turn_count, 0.5) / 0.5
        else:
            init_ratio = 0.0
        init_score = init_ratio * 20.0

        return w_share * share_score + w_init * init_score

    def _score_technical_contribution(
        self, student: StudentMetrics, group: GroupMetrics
    ) -> float:
        """Technical contribution: turn count share + (future) keyword density.

        The keyword density component is a placeholder: without access to the
        raw transcript text here, we use turn_count as a proxy.  When the full
        transcript text is available it should be passed down and used instead.
        """
        weights = self.config.rubric_weights.get("technical_contribution", {})
        w_turns = weights.get("turn_count_share", 0.5)
        w_keyword = weights.get("keyword_density", 0.5)

        # Turn count share relative to equal share
        total_students = max(group.total_students, 1)
        total_turns = sum(m.turn_count for m in group.per_student_metrics) or 1
        turn_share = student.turn_count / total_turns
        equal_turn_share = 1.0 / total_students
        turn_ratio = min(turn_share / equal_turn_share, 2.0) / 2.0
        turn_score = turn_ratio * 20.0

        # Keyword density placeholder: proportional to turn count share
        # (to be replaced with actual transcript analysis)
        keyword_score = turn_score

        return w_turns * turn_score + w_keyword * keyword_score
