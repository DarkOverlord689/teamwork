"""PDF report generator for SMATC-UPAO analysis results (Module 2.4).

Generates a structured PDF containing group summary, per-student metrics,
UPAO rubric scores, narrative explanation, strengths and improvements.
"""

from __future__ import annotations

import io
from datetime import datetime
from typing import Any, Dict, List

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


class PDFReportGenerator:
    """Generate a PDF report from fusion analysis data.

    Parameters
    ----------
    analysis_data:
        Dictionary containing the full fusion result structure (group_metrics,
        rubric_scores, explanation) plus metadata (group_name, course, date).
    """

    def __init__(self, analysis_data: Dict[str, Any]) -> None:
        self.data = analysis_data
        self.styles = getSampleStyleSheet()
        self._register_custom_styles()

    # ------------------------------------------------------------------
    # Custom styles
    # ------------------------------------------------------------------

    def _register_custom_styles(self) -> None:
        self.styles.add(
            ParagraphStyle(
                name="ReportTitle",
                parent=self.styles["Title"],
                fontSize=18,
                leading=22,
                textColor=colors.HexColor("#1976d2"),
                alignment=TA_CENTER,
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="SectionHeader",
                parent=self.styles["Heading2"],
                fontSize=12,
                leading=16,
                textColor=colors.HexColor("#1976d2"),
                spaceAfter=6,
                spaceBefore=12,
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="BulletItem",
                parent=self.styles["Normal"],
                fontSize=10,
                leading=14,
                leftIndent=12,
                bulletIndent=0,
                spaceAfter=3,
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="FooterStyle",
                parent=self.styles["Normal"],
                fontSize=8,
                textColor=colors.grey,
                alignment=TA_CENTER,
            )
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self) -> bytes:
        """Generate the PDF and return raw bytes."""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            leftMargin=2.5 * cm,
            rightMargin=2.5 * cm,
            topMargin=2 * cm,
            bottomMargin=2.5 * cm,
        )

        story: List[Any] = []

        story.extend(self._build_header())
        story.extend(self._build_group_summary())
        story.extend(self._build_student_metrics_table())
        story.extend(self._build_rubric_table())
        story.extend(self._build_narrative())
        story.extend(self._build_strengths_improvements())
        story.extend(self._build_footer())

        doc.build(story)
        return buffer.getvalue()

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    def _build_header(self) -> List[Any]:
        group_name = self.data.get("group_name", "Grupo")
        course = self.data.get("course", "")
        date_str = self.data.get("date", datetime.utcnow().strftime("%Y-%m-%d"))

        elements: List[Any] = [
            Paragraph("SMATC-UPAO", self.styles["ReportTitle"]),
            Paragraph(
                "Sistema Multimodal de Análisis de Trabajo Colaborativo",
                ParagraphStyle(
                    "SubTitle",
                    parent=self.styles["Normal"],
                    alignment=TA_CENTER,
                    fontSize=10,
                    textColor=colors.grey,
                    spaceAfter=4,
                ),
            ),
            HRFlowable(width="100%", thickness=2, color=colors.HexColor("#1976d2")),
            Spacer(1, 0.4 * cm),
        ]

        header_data = [
            ["Grupo:", group_name],
            ["Curso:", course or "—"],
            ["Fecha:", date_str],
        ]
        header_table = Table(header_data, colWidths=[3 * cm, 12 * cm])
        header_table.setStyle(
            TableStyle(
                [
                    ("FONT", (0, 0), (-1, -1), "Helvetica", 10),
                    ("FONT", (0, 0), (0, -1), "Helvetica-Bold", 10),
                    ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#1976d2")),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ]
            )
        )
        elements.append(header_table)
        elements.append(Spacer(1, 0.5 * cm))
        return elements

    def _build_group_summary(self) -> List[Any]:
        group_metrics = self.data.get("group_metrics", {})
        elements: List[Any] = [
            Paragraph("1. Resumen del Grupo", self.styles["SectionHeader"]),
        ]

        total_students = group_metrics.get("total_students", 0)
        duration_s = group_metrics.get("duration_seconds", 0)
        duration_min = round(duration_s / 60, 1)
        cv = group_metrics.get("participation_cv", 0.0)

        rows = [
            ["Métrica", "Valor"],
            ["Total de estudiantes", str(total_students)],
            ["Duración de la sesión", f"{duration_min} min ({int(duration_s)} seg)"],
            ["CV de participación general", f"{cv:.3f}"],
        ]
        tbl = self._make_simple_table(rows)
        elements.append(tbl)
        elements.append(Spacer(1, 0.3 * cm))
        return elements

    def _build_student_metrics_table(self) -> List[Any]:
        group_metrics = self.data.get("group_metrics", {})
        per_student: List[Dict[str, Any]] = group_metrics.get("per_student_metrics", [])

        elements: List[Any] = [
            Paragraph("2. Métricas por Estudiante", self.styles["SectionHeader"]),
        ]

        header = ["Estudiante", "Tiempo Habla (s)", "Turnos", "Interrupciones", "Contacto Visual (%)"]
        rows = [header]
        for sm in per_student:
            rows.append(
                [
                    sm.get("student_id", "—"),
                    f"{sm.get('speaking_time_seconds', 0):.1f}",
                    str(sm.get("turn_count", 0)),
                    str(sm.get("interruption_count", 0)),
                    f"{sm.get('gaze_contact_percentage', 0):.1f}%",
                ]
            )

        if len(rows) == 1:
            rows.append(["Sin datos", "—", "—", "—", "—"])

        tbl = self._make_simple_table(rows, col_widths=[4 * cm, 3.5 * cm, 2 * cm, 3 * cm, 3.5 * cm])
        elements.append(tbl)
        elements.append(Spacer(1, 0.3 * cm))
        return elements

    def _build_rubric_table(self) -> List[Any]:
        rubric_scores = self.data.get("rubric_scores", {})
        per_student_scores: List[Dict[str, Any]] = rubric_scores.get("per_student_scores", [])

        elements: List[Any] = [
            Paragraph("3. Puntuaciones Rúbrica UPAO (0–20)", self.styles["SectionHeader"]),
            Paragraph(
                "Color: verde ≥ 16 | amarillo 8–15.9 | rojo < 8",
                ParagraphStyle(
                    "Legend",
                    parent=self.styles["Normal"],
                    fontSize=8,
                    textColor=colors.grey,
                    spaceAfter=4,
                ),
            ),
        ]

        header = [
            "Estudiante",
            "Colaboración",
            "Comunicación",
            "Responsabilidad",
            "Liderazgo",
            "Contribución Técnica",
        ]
        rows = [header]
        for ps in per_student_scores:
            rows.append(
                [
                    ps.get("student_id", "—"),
                    f"{ps.get('collaboration', 0):.1f}",
                    f"{ps.get('communication', 0):.1f}",
                    f"{ps.get('responsibility', 0):.1f}",
                    f"{ps.get('leadership', 0):.1f}",
                    f"{ps.get('technical_contribution', 0):.1f}",
                ]
            )

        if len(rows) == 1:
            rows.append(["Sin datos", "—", "—", "—", "—", "—"])

        col_w = [4 * cm, 2.5 * cm, 2.5 * cm, 3 * cm, 2.5 * cm, 3.5 * cm]
        tbl = Table(rows, colWidths=col_w)

        style_cmds = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1976d2")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 9),
            ("FONT", (0, 1), (-1, -1), "Helvetica", 9),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
        ]

        for row_idx, row in enumerate(rows[1:], start=1):
            for col_idx, val in enumerate(row[1:], start=1):
                try:
                    score = float(val.rstrip("%"))
                    if score >= 16:
                        bg = colors.HexColor("#c8e6c9")
                    elif score >= 8:
                        bg = colors.HexColor("#fff9c4")
                    else:
                        bg = colors.HexColor("#ffcdd2")
                    style_cmds.append(("BACKGROUND", (col_idx, row_idx), (col_idx, row_idx), bg))
                except (ValueError, AttributeError):
                    pass

        tbl.setStyle(TableStyle(style_cmds))
        elements.append(tbl)
        elements.append(Spacer(1, 0.3 * cm))
        return elements

    def _build_narrative(self) -> List[Any]:
        explanation = self.data.get("explanation", {})
        narrative = explanation.get("narrative_text", "Sin análisis narrativo disponible.")

        elements: List[Any] = [
            Paragraph("4. Análisis Narrativo", self.styles["SectionHeader"]),
            Paragraph(
                narrative,
                ParagraphStyle(
                    "NarrativeBody",
                    parent=self.styles["Normal"],
                    fontSize=10,
                    leading=15,
                    spaceAfter=6,
                ),
            ),
            Spacer(1, 0.3 * cm),
        ]
        return elements

    def _build_strengths_improvements(self) -> List[Any]:
        explanation = self.data.get("explanation", {})
        strengths: List[str] = explanation.get("strengths", [])
        improvements: List[str] = explanation.get("improvements", [])

        elements: List[Any] = [
            Paragraph("5. Fortalezas y Áreas de Mejora", self.styles["SectionHeader"]),
            Paragraph("Fortalezas:", self.styles["Heading3"]),
        ]

        if strengths:
            for s in strengths:
                elements.append(Paragraph(f"• {s}", self.styles["BulletItem"]))
        else:
            elements.append(Paragraph("• Sin fortalezas registradas.", self.styles["BulletItem"]))

        elements.append(Spacer(1, 0.2 * cm))
        elements.append(Paragraph("Áreas de Mejora:", self.styles["Heading3"]))

        if improvements:
            for imp in improvements:
                elements.append(Paragraph(f"• {imp}", self.styles["BulletItem"]))
        else:
            elements.append(Paragraph("• Sin áreas de mejora registradas.", self.styles["BulletItem"]))

        elements.append(Spacer(1, 0.4 * cm))
        return elements

    def _build_footer(self) -> List[Any]:
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        return [
            HRFlowable(width="100%", thickness=1, color=colors.lightgrey),
            Spacer(1, 0.2 * cm),
            Paragraph(
                f"Generado el {timestamp} | Generated by SMATC-UPAO",
                self.styles["FooterStyle"],
            ),
        ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_simple_table(
        self,
        rows: List[List[str]],
        col_widths: List[float] | None = None,
    ) -> Table:
        tbl = Table(rows, colWidths=col_widths)
        tbl.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1976d2")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 10),
                    ("FONT", (0, 1), (-1, -1), "Helvetica", 10),
                    ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                    ("ALIGN", (0, 0), (0, -1), "LEFT"),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        return tbl
