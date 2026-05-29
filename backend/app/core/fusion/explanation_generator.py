"""Narrative explanation generation for Module 2.3.

``ExplanationGenerator`` uses the OpenAI Chat Completions API to call a
hosted LLM (gpt-4o-mini by default) and generate a structured narrative
explanation of the group's collaboration performance.  If the API is
unavailable it falls back gracefully to a deterministic rule-based
explanation.

The ``openai`` SDK is used for the API call.  OpenAI is NOT a hard
dependency — the module runs without it (rule-based fallback is always
available).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Optional

from openai import AsyncOpenAI

from app.core.fusion.config import FusionConfig
from app.core.fusion.data_types import (
    ExplanationResult,
    GroupMetrics,
    GroupRubricScores,
)

if TYPE_CHECKING:
    from app.core.audio.data_types import AudioResult

logger = logging.getLogger(__name__)


class ExplanationGenerator:
    """Generate narrative explanations of collaboration performance.

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

    async def generate(
        self,
        group_metrics: GroupMetrics,
        rubric_scores: GroupRubricScores,
        audio_result: Optional["AudioResult"] = None,
    ) -> ExplanationResult:
        """Generate a narrative explanation.

        Tries Ollama first; falls back to rule-based explanation on any error.

        Parameters
        ----------
        group_metrics : GroupMetrics
        rubric_scores : GroupRubricScores
        audio_result : AudioResult, optional
            Used to build per-student transcript excerpts for topic_description
            and intervention_summary generation.

        Returns
        -------
        ExplanationResult
        """
        if not self.config.enable_explanation:
            logger.info(
                "Explanation generation disabled by config; using rule-based fallback"
            )
            return self._rule_based_explanation(group_metrics, rubric_scores)

        try:
            return await self._llm_explanation(
                group_metrics, rubric_scores, audio_result
            )
        except Exception as exc:
            logger.warning(
                "LLM explanation failed (%s); falling back to rule-based explanation",
                exc,
            )
            return self._rule_based_explanation(group_metrics, rubric_scores)

    # ------------------------------------------------------------------
    # LLM path
    # ------------------------------------------------------------------

    async def _llm_explanation(
        self,
        group_metrics: GroupMetrics,
        rubric_scores: GroupRubricScores,
        audio_result: Optional["AudioResult"] = None,
    ) -> ExplanationResult:
        """Call OpenAI Chat Completions and parse a structured explanation."""
        prompt = self._build_prompt(group_metrics, rubric_scores, audio_result)

        client = AsyncOpenAI(
            api_key=self.config.openai_api_key,
            timeout=self.config.llm_timeout_seconds,
        )
        response = await client.chat.completions.create(
            model=self.config.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config.llm_temperature,
            max_tokens=self.config.max_explanation_tokens,
        )

        raw_text: str = (response.choices[0].message.content or "").strip()
        if not raw_text:
            raise ValueError("Empty response from LLM")

        return self._parse_llm_response(raw_text)

    @staticmethod
    def _build_transcript_excerpt(
        audio_result: Optional["AudioResult"], max_chars: int = 2000
    ) -> str:
        """Build a condensed transcript excerpt grouped by speaker.

        Returns an empty string when no transcripts are available.
        """
        if audio_result is None or not audio_result.transcripts:
            return ""

        # Group transcript text by speaker, keeping chronological order
        by_speaker: Dict[str, List[str]] = {}
        for seg in audio_result.transcripts:
            sid = seg.speaker_id
            if sid not in by_speaker:
                by_speaker[sid] = []
            text = seg.text.strip()
            if text:
                by_speaker[sid].append(text)

        lines = []
        for sid, texts in by_speaker.items():
            excerpt = " ".join(texts)
            if len(excerpt) > 400:
                excerpt = excerpt[:400] + "..."
            lines.append(f"  {sid}: {excerpt}")

        full = "\n".join(lines)
        if len(full) > max_chars:
            full = full[:max_chars] + "\n  [transcripción truncada]"
        return full

    def _build_prompt(
        self,
        group_metrics: GroupMetrics,
        rubric_scores: GroupRubricScores,
        audio_result: Optional["AudioResult"] = None,
    ) -> str:
        """Build the structured prompt sent to the LLM."""
        cv = group_metrics.participation_cv
        equity_label = "equitativa" if cv < self.config.cv_threshold else "desigual"

        student_lines = "\n".join(
            f"  - {m.student_id}: habla {m.speaking_time_seconds:.1f}s, "
            f"{m.turn_count} turnos, {m.gaze_contact_percentage:.0f}% contacto visual, "
            f"emoción dominante: {m.dominant_emotion}"
            for m in group_metrics.per_student_metrics
        )

        score_lines = "\n".join(
            f"  - {s.student_id}: "
            f"contribuye_a_reuniones={s.contributes_to_team_meetings:.1f}, "
            f"facilita_contribuciones={s.facilitates_contributions:.1f}, "
            f"clima_constructivo={s.fosters_constructive_climate:.1f}, "
            f"responde_a_conflictos={s.responds_to_conflict:.1f}, "
            f"contribuciones_fuera={s.individual_contributions_outside:.1f}"
            for s in rubric_scores.per_student_scores
        )

        student_ids = [m.student_id for m in group_metrics.per_student_metrics]
        intervention_placeholders = "\n".join(
            f"INTERVENCION_{sid}:\n[1-2 oraciones describiendo la contribución de {sid}]"
            for sid in student_ids
        )

        transcript_section = self._build_transcript_excerpt(audio_result)
        transcript_block = (
            f"\n=== FRAGMENTO DE TRANSCRIPCIÓN ===\n{transcript_section}\n"
            if transcript_section
            else ""
        )

        return f"""Eres un evaluador académico del sistema SMATC-UPAO que analiza trabajo colaborativo
en grupos de ingeniería.  Con base en las métricas y transcripción de la sesión, genera un análisis en español.

=== MÉTRICAS DEL GRUPO ===
- Número de estudiantes: {group_metrics.total_students}
- Duración de la sesión: {group_metrics.duration_seconds:.0f} segundos
- CV de participación: {cv:.3f} (participación {equity_label}, meta < 0.30)
- Tasa de interrupciones disruptivas: {group_metrics.disruptive_interruption_rate:.2%}
- Score de sincronización de turnos: {group_metrics.turn_synchronization_score:.2f} / 1.00
- Contacto visual promedio: {group_metrics.avg_gaze_contact_percentage:.1f}%
- Ratio de silencio: {group_metrics.silence_ratio:.2%}

=== MÉTRICAS POR ESTUDIANTE ===
{student_lines}

=== PUNTUACIONES RÚBRICA VALUE (escala 0-20) ===
{score_lines}
{transcript_block}
=== INSTRUCCIONES ===
Genera un análisis estructurado con exactamente estas secciones en español.
Sé específico y usa los datos proporcionados.  Máximo 4 oraciones por sección.

TEMA:
[2-3 oraciones describiendo el tema o tópico principal discutido en la sesión, basado en la transcripción y contexto]

NARRATIVA:
[párrafo de análisis general del desempeño del grupo]

FORTALEZAS:
- [fortaleza 1]
- [fortaleza 2]
- [fortaleza 3]

ÁREAS DE MEJORA:
- [área 1]
- [área 2]

RECOMENDACIONES:
- [recomendación específica 1]
- [recomendación específica 2]
- [recomendación específica 3]

{intervention_placeholders}
"""

    def _parse_llm_response(self, raw: str) -> ExplanationResult:
        """Parse the structured LLM response into an ExplanationResult."""
        sections: Dict[str, str] = {}
        current_key: str | None = None
        current_lines: List[str] = []

        # Fixed section markers
        fixed_markers = {
            "TEMA:": "topic",
            "NARRATIVA:": "narrative",
            "FORTALEZAS:": "strengths",
            "ÁREAS DE MEJORA:": "improvements",
            "RECOMENDACIONES:": "recommendations",
        }

        for line in raw.splitlines():
            stripped = line.strip()
            matched = False
            upper = stripped.upper()

            # Check fixed markers first
            for marker, key in fixed_markers.items():
                if upper.startswith(marker):
                    if current_key and current_lines:
                        sections[current_key] = "\n".join(current_lines).strip()
                    current_key = key
                    current_lines = []
                    matched = True
                    break

            # Check dynamic INTERVENCION_{student_id}: markers
            if not matched and upper.startswith("INTERVENCION_"):
                section_key = stripped.rstrip(":").strip()
                if current_key and current_lines:
                    sections[current_key] = "\n".join(current_lines).strip()
                current_key = section_key
                current_lines = []
                matched = True

            if not matched and current_key:
                current_lines.append(stripped)

        if current_key and current_lines:
            sections[current_key] = "\n".join(current_lines).strip()

        narrative = sections.get("narrative", raw[:300])
        topic_description = sections.get("topic") or None

        def _parse_bullets(text: str) -> List[str]:
            return [
                line.lstrip("- •*").strip()
                for line in text.splitlines()
                if line.strip().startswith(("-", "•", "*"))
            ]

        # Collect per-student intervention summaries
        intervention_summaries: Dict[str, str] = {}
        for key, value in sections.items():
            if key.upper().startswith("INTERVENCION_"):
                student_id = key[len("INTERVENCION_") :]
                if value:
                    intervention_summaries[student_id] = value

        return ExplanationResult(
            narrative_text=narrative,
            strengths=_parse_bullets(sections.get("strengths", "")),
            improvements=_parse_bullets(sections.get("improvements", "")),
            recommendations=_parse_bullets(sections.get("recommendations", "")),
            generated_by="openai",
            model_used=self.config.llm_model,
            topic_description=topic_description,
            intervention_summaries=intervention_summaries,
        )

    # ------------------------------------------------------------------
    # Rule-based fallback
    # ------------------------------------------------------------------

    def _rule_based_explanation(
        self,
        group_metrics: GroupMetrics,
        rubric_scores: GroupRubricScores,
    ) -> ExplanationResult:
        """Generate a deterministic rule-based explanation without LLM."""
        cv = group_metrics.participation_cv
        sync = group_metrics.turn_synchronization_score
        intr = group_metrics.disruptive_interruption_rate
        gaze = group_metrics.avg_gaze_contact_percentage

        # Narrative
        equity_desc = "equitativa" if cv < self.config.cv_threshold else "desigual"
        narrative = (
            f"El grupo de {group_metrics.total_students} estudiantes mostró una participación "
            f"{equity_desc} (CV = {cv:.2f}, meta < {self.config.cv_threshold}). "
            f"La sincronización de turnos fue de {sync:.0%} y la tasa de interrupciones "
            f"disruptivas fue de {intr:.0%}. "
            f"El contacto visual promedio fue del {gaze:.0f}%."
        )

        # Strengths
        strengths: List[str] = []
        if cv < self.config.cv_threshold:
            strengths.append(
                "Participación equilibrada entre todos los integrantes del grupo."
            )
        if sync >= 0.7:
            strengths.append(
                "Los turnos de habla se coordinaron de forma fluida y ordenada."
            )
        if gaze >= 60.0:
            strengths.append(
                "Buen nivel de contacto visual, indicando atención y compromiso."
            )
        if intr <= 0.1:
            strengths.append(
                "Pocas interrupciones disruptivas; respeto por los turnos de habla."
            )

        if not strengths:
            strengths.append("El grupo completó la actividad colaborativa.")

        # Areas of improvement
        improvements: List[str] = []
        if cv >= self.config.cv_threshold:
            improvements.append(
                "Mejorar el equilibrio de participación; algunos estudiantes hablan "
                "significativamente más que otros."
            )
        if sync < 0.7:
            improvements.append(
                "Trabajar en la transición entre turnos para reducir solapamientos y silencios largos."
            )
        if intr > 0.15:
            improvements.append(
                "Reducir las interrupciones disruptivas respetando el turno del hablante."
            )
        if gaze < 40.0:
            improvements.append(
                "Aumentar el contacto visual con el equipo para demostrar atención activa."
            )

        # Recommendations
        recommendations: List[str] = [
            "Establecer un turno rotativo de moderador para equilibrar la participación.",
            "Practicar la escucha activa: no hablar mientras otro compañero esté haciendo un punto.",
            "Utilizar señales no verbales (asentir, mirar al hablante) para mostrar compromiso.",
        ]

        # Per-student intervention summaries (rule-based fallback)
        intervention_summaries: Dict[str, str] = {}
        for m in group_metrics.per_student_metrics:
            ratio_pct = m.participation_ratio * 100
            emotion = m.dominant_emotion
            intervention_summaries[m.student_id] = (
                f"{m.student_id} participó durante {m.speaking_time_seconds:.0f} segundos "
                f"({ratio_pct:.0f}% del tiempo total) en {m.turn_count} turnos, "
                f"con emoción dominante '{emotion}'."
            )

        return ExplanationResult(
            narrative_text=narrative,
            strengths=strengths,
            improvements=improvements,
            recommendations=recommendations,
            generated_by="rule_based",
            model_used=None,
            topic_description=None,
            intervention_summaries=intervention_summaries,
        )
