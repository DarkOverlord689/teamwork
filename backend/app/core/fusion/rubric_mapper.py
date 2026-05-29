"""VALUE rubric mapping for the multimodal fusion pipeline (Module 2.3).

``VALUERubricMapper`` converts raw collaboration metrics into the five AAC&U
VALUE Teamwork Rubric dimensions on a 0–20 scale:

1. Contributes to Team Meetings
2. Facilitates the Contributions of Team Members
3. Individual Contributions Outside of Team Meetings (placeholder)
4. Fosters Constructive Team Climate
5. Responds to Conflict

Each dimension maps to VALUE levels:
  Benchmark (1) → 5.0   | Milestone 2 (2) → 10.0
  Milestone 3 (3) → 15.0 | Capstone (4) → 20.0
"""

from __future__ import annotations

import logging
from typing import Dict, List

from app.core.fusion.config import FusionConfig
from app.core.fusion.data_types import (
    GroupMetrics,
    GroupRubricScores,
    RubricScores,
    StudentMetrics,
)

logger = logging.getLogger(__name__)


def _clamp(value: float, lo: float = 0.0, hi: float = 20.0) -> float:
    """Clamp a floating-point value to [lo, hi]."""
    return max(lo, min(hi, value))


def _sub_score(raw: float, threshold: float, cap: float = 1.0) -> float:
    """Normalise a raw metric to a 0–1 sub-score capped at *cap*.

    ``raw`` is the metric value (e.g. participation ratio),
    ``threshold`` is the "perfect" target value,
    ``cap`` limits how much above 1.0 the result can go.
    """
    if threshold == 0:
        return 0.0
    return min(raw / threshold, cap)


class VALUERubricMapper:
    """Map collaboration metrics to AAC&U VALUE teamwork rubric scores (0–20).

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
        """Map a single student's metrics to VALUE rubric scores.

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
        equal_share = 1.0 / total_students

        return RubricScores(
            student_id=student.student_id,
            contributes_to_team_meetings=_clamp(
                self._score_contributes_to_team_meetings(
                    student, group_metrics, equal_share
                )
            ),
            facilitates_contributions=_clamp(
                self._score_facilitates_contributions(
                    student, group_metrics, equal_share
                )
            ),
            fosters_constructive_climate=_clamp(
                self._score_fosters_constructive_climate(student, group_metrics)
            ),
            responds_to_conflict=_clamp(
                self._score_responds_to_conflict(student, group_metrics)
            ),
            individual_contributions_outside=0.0,
        )

    def map_group_to_rubric(
        self,
        group_metrics: GroupMetrics,
    ) -> GroupRubricScores:
        """Map group metrics to group-level VALUE rubric scores.

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

        if per_student:
            n = len(per_student)
            avg_ctm = sum(s.contributes_to_team_meetings for s in per_student) / n
            avg_fc = sum(s.facilitates_contributions for s in per_student) / n
            avg_fcc = sum(s.fosters_constructive_climate for s in per_student) / n
            avg_rc = sum(s.responds_to_conflict for s in per_student) / n
            avg_ico = sum(s.individual_contributions_outside for s in per_student) / n
        else:
            avg_ctm = avg_fc = avg_fcc = avg_rc = avg_ico = 0.0

        return GroupRubricScores(
            contributes_to_team_meetings=_clamp(avg_ctm),
            facilitates_contributions=_clamp(avg_fc),
            fosters_constructive_climate=_clamp(avg_fcc),
            responds_to_conflict=_clamp(avg_rc),
            individual_contributions_outside=_clamp(avg_ico),
            per_student_scores=per_student,
        )

    # ------------------------------------------------------------------
    # VALUE Dimension 1: Contributes to Team Meetings
    # ------------------------------------------------------------------

    def _score_contributes_to_team_meetings(
        self,
        student: StudentMetrics,
        group: GroupMetrics,
        equal_share: float,
    ) -> float:
        """Contributes to Team Meetings – active participation + idea contribution.

        VALUE levels:
          1 (5pt):  Shares ideas but does not advance the work.
          2 (10pt): Offers new suggestions to advance the work.
          3 (15pt): Offers alternative solutions that build on others' ideas.
          4 (20pt): Articulates merits of alternative ideas, moves team forward.

        Sub-metrics:
          * participation_engagement (40%): ratio of actual to equal share,
            penalised when dominating (>2x share) or barely present (<0.5x).
          * idea_contribution (35%): initiation_count / turn_count – higher
            ratio means the student brings new topics, not just reacts.
          * turn_frequency (25%): turn_count relative to the group average –
            engaged contributors speak at a healthy frequency.
        """
        w = self.config.rubric_weights.get("contributes_to_team_meetings", {})

        # --- participation_engagement ---
        if equal_share > 0:
            ratio = student.participation_ratio / equal_share
        else:
            ratio = 0.0

        if ratio > 2.0:  # dominating the conversation
            engagement = 0.6
        elif ratio > 1.0:  # slightly above fair share, fine
            engagement = 1.0 - (ratio - 1.0) * 0.2
        elif ratio > 0.5:  # decent participation
            engagement = 0.5 + (ratio - 0.5) * 0.5
        else:  # barely participating
            engagement = ratio

        engagement = max(engagement, 0.0)

        # --- idea_contribution ---
        if student.turn_count > 0:
            init_rate = student.initiation_count / student.turn_count
        else:
            init_rate = 0.0
        idea = min(init_rate / 0.5, 1.0)  # cap at 50% initiation rate

        # --- turn_frequency ---
        total_students = max(group.total_students, 1)
        other_turns = (
            sum(m.turn_count for m in group.per_student_metrics) + 1
        )  # +1 avoids div-zero
        expected_turns = other_turns / total_students
        turn_ratio = min(student.turn_count / max(expected_turns, 1), 1.5) / 1.5
        turn_ratio = max(turn_ratio, 0.0)

        w_engage = w.get("participation_engagement", 0.40)
        w_idea = w.get("idea_contribution", 0.35)
        w_turn = w.get("turn_frequency", 0.25)

        return (w_engage * engagement + w_idea * idea + w_turn * turn_ratio) * 20.0

    # ------------------------------------------------------------------
    # VALUE Dimension 2: Facilitates the Contributions of Team Members
    # ------------------------------------------------------------------

    def _score_facilitates_contributions(
        self,
        student: StudentMetrics,
        group: GroupMetrics,
        equal_share: float,
    ) -> float:
        """Facilitates Contributions – engaging + building on others' ideas.

        VALUE levels:
          1 (5pt):  Takes turns and listens without interrupting.
          2 (10pt): Restates views of others and/or asks questions.
          3 (15pt): Builds upon the contributions of others.
          4 (20pt): Facilitates contributions by building/synthesising and
                    inviting non-participants to engage.

        Sub-metrics:
          * active_listening (35%): back_channel_count relative to turns +
            gaze_contact when others are speaking (proxied by overall gaze).
          * cooperative_engagement (40%): low disruptive interruption count +
            presence of cooperative overlaps + gaze towards peers.
          * engagement_signals (25%): gaze_contact_percentage as proxy for
            sustained attention to others.
        """
        w = self.config.rubric_weights.get("facilitates_contributions", {})

        # --- active_listening ---
        if student.turn_count > 0:
            bc_ratio = student.back_channel_count / student.turn_count
        else:
            bc_ratio = 0.0
        bc_score = min(bc_ratio / 0.5, 1.0)  # good listener: ~50% of turns have BC
        gaze_fraction = student.gaze_contact_percentage / 100.0
        listening = 0.6 * bc_score + 0.4 * gaze_fraction

        # --- cooperative_engagement ---
        # Fewer disruptive interruptions + better gaze = more constructive
        disruption_penalty = min(
            student.interruption_count / max(student.turn_count, 1), 1.0
        )
        # Invert: no disruptions = 1.0, many = 0.0
        cooperation = max(0.0, 1.0 - disruption_penalty)

        # Mix: gaze shows attention to others, cooperation shows constructive intent
        coop_engage = 0.4 * gaze_fraction + 0.6 * cooperation

        # --- engagement_signals ---
        # Gaze contact as primary signal of engagement with others
        engagement = gaze_fraction

        w_listen = w.get("active_listening", 0.35)
        w_coop = w.get("cooperative_engagement", 0.40)
        w_engage = w.get("engagement_signals", 0.25)

        return (
            w_listen * listening + w_coop * coop_engage + w_engage * engagement
        ) * 20.0

    # ------------------------------------------------------------------
    # VALUE Dimension 3: Individual Contributions Outside Team Meetings
    # ------------------------------------------------------------------

    def _score_individual_contributions_outside(
        self,
        student: StudentMetrics,
        group: GroupMetrics,
    ) -> float:
        """Individual Contributions Outside of Team Meetings – placeholder.

        This VALUE dimension evaluates whether the student completed assigned
        tasks, conducted research, and brought materials to the session.
        It CANNOT be measured from video/audio data alone – it requires
        external data sources (LMS submissions, GitHub commits, etc.).

        Always returns 0.0.
        """
        return 0.0

    # ------------------------------------------------------------------
    # VALUE Dimension 4: Fosters Constructive Team Climate
    # ------------------------------------------------------------------

    def _score_fosters_constructive_climate(
        self,
        student: StudentMetrics,
        group: GroupMetrics,
    ) -> float:
        """Fosters Constructive Team Climate – respect, positive tone, body language.

        VALUE levels:
          1 (5pt):  Supports a constructive climate by doing ONE of: treating
                    respectfully, using positive tone, motivating teammates.
          2 (10pt): Does any TWO of the above.
          3 (15pt): Does any THREE of the above.
          4 (20pt): Does ALL FOUR: respectfully, positive tone, positive body
                    language, motivates teammates.

        Sub-metrics:
          * emotional_tone (35%): dominant_emotion – positive emotions
            (attentive, happy) score high; negative (angry, sad) score low.
          * respect_signals (35%): low disruptive interruptions + high gaze
            contact = respectful engagement.
          * body_language (30%): body_orientation (facing group = engaged) +
            gesture data (nod = agreement, positive).
        """
        w = self.config.rubric_weights.get("fosters_constructive_climate", {})

        # --- emotional_tone ---
        emotion = student.dominant_emotion.lower()
        positive_emotions = self.config.positive_emotions
        negative_emotions = self.config.negative_emotions

        if emotion in {"attentive", "happy"}:
            emotion_score = 0.9
        elif emotion in {"surprised", "neutral"}:
            emotion_score = 0.7
        elif emotion in {"sad", "fearful"}:
            emotion_score = 0.35
        elif emotion in {"angry", "disgusted", "contemptuous"}:
            emotion_score = 0.1
        else:
            emotion_score = 0.5  # unknown

        # --- respect_signals ---
        disrupt_ratio = student.interruption_count / max(student.turn_count, 1)
        disruption_penalty = min(disrupt_ratio, 1.0)
        respect = max(0.0, 1.0 - disruption_penalty * 1.5)  # amplify penalty

        gaze_fraction = student.gaze_contact_percentage / 100.0
        respect_score = 0.5 * respect + 0.5 * gaze_fraction

        # --- body_language ---
        # Body orientation: 0° = facing forward/engaged, 90° = turned away
        # Normalise: 0° -> 1.0, 90° -> 0.0
        orientation_score = max(0.0, 1.0 - student.avg_body_orientation / 90.0)

        w_emotion = w.get("emotional_tone", 0.35)
        w_respect = w.get("respect_signals", 0.35)
        w_body = w.get("body_language", 0.30)

        return (
            w_emotion * emotion_score
            + w_respect * respect_score
            + w_body * orientation_score
        ) * 20.0

    # ------------------------------------------------------------------
    # VALUE Dimension 5: Responds to Conflict
    # ------------------------------------------------------------------

    def _score_responds_to_conflict(
        self,
        student: StudentMetrics,
        group: GroupMetrics,
    ) -> float:
        """Responds to Conflict – addressing disagreements constructively.

        VALUE levels:
          1 (5pt):  Passively accepts alternate viewpoints.
          2 (10pt): Redirects focus toward common ground, away from conflict.
          3 (15pt): Identifies and acknowledges conflict, stays engaged.
          4 (20pt): Addresses conflict directly and constructively,
                    helps manage/resolve it.

        Sub-metrics:
          * conflict_management (50%): low disruptive interruption rate at
            group level + high turn synchronization = healthy group dynamics.
          * constructive_engagement (50%): balance between interrupting and
            being interrupted. A student who engages in conflict constructively
            has a balanced profile – they're not a passive victim nor an
            aggressive dominator.
        """
        w = self.config.rubric_weights.get("responds_to_conflict", {})

        # --- conflict_management ---
        # Group-level: low disruptive rate + high turn sync = healthy
        disruptive_score = max(0.0, 1.0 - group.disruptive_interruption_rate * 2.0)
        sync_score = group.turn_synchronization_score
        conflict_mgmt = 0.5 * disruptive_score + 0.5 * sync_score

        # --- constructive_engagement ---
        # Personal balance: ratio of interruptions done vs received
        # Good: near 1.0 (balanced), Bad: >>1 (aggressor) or <<1 (victim)
        total_intr = student.interruption_count + student.interrupted_count
        if total_intr > 0:
            balance = student.interruption_count / total_intr
            # Score peaks at 0.5 (balanced), penalises extremes
            constructive = 1.0 - abs(balance - 0.5) * 2.0
            constructive = max(constructive, 0.0)
        else:
            constructive = 1.0  # no interruptions at all = neutral

        w_mgmt = w.get("conflict_management", 0.50)
        w_construct = w.get("constructive_engagement", 0.50)

        return (w_mgmt * conflict_mgmt + w_construct * constructive) * 20.0
