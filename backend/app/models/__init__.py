import uuid
from datetime import datetime

from sqlalchemy import String, Integer, Float, DateTime, ForeignKey, Enum as SQLEnum
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Group(Base):
    """Grupo de estudiantes que trabajan juntos en una sesión."""

    __tablename__ = "groups"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    course_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=True)
    name: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    students: Mapped[list["Student"]] = relationship("Student", back_populates="group")
    sessions: Mapped[list["AnalysisSession"]] = relationship(
        "AnalysisSession", back_populates="group"
    )


class Student(Base):
    """Estudiante registrado en el sistema."""

    __tablename__ = "students"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    group_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("groups.id")
    )
    name: Mapped[str] = mapped_column(String(255))
    student_code: Mapped[str] = mapped_column(String(50), nullable=True)

    group: Mapped["Group"] = relationship("Group", back_populates="students")


class AnalysisSession(Base):
    """Una sesión de análisis de video."""

    __tablename__ = "analysis_sessions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    group_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("groups.id")
    )
    video_path: Mapped[str] = mapped_column(String(500))
    duration_seconds: Mapped[int] = mapped_column(Integer, nullable=True)
    processed_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    status: Mapped[str] = mapped_column(String(50), default="pending")
    session_metrics: Mapped[dict] = mapped_column(JSONB, nullable=True)
    result_data: Mapped[dict] = mapped_column(JSONB, nullable=True)
    topic_description: Mapped[str] = mapped_column(String, nullable=True)

    group: Mapped["Group"] = relationship("Group", back_populates="sessions")
    visual_metrics: Mapped[list["VisualMetrics"]] = relationship(
        "VisualMetrics", back_populates="session"
    )
    audio_metrics: Mapped[list["AudioMetrics"]] = relationship(
        "AudioMetrics", back_populates="session"
    )
    rubric_scores: Mapped[list["RubricScore"]] = relationship(
        "RubricScore", back_populates="session"
    )
    explanations: Mapped[list["Explanation"]] = relationship(
        "Explanation", back_populates="session"
    )


class VisualMetrics(Base):
    """Métricas visuales calculadas para un estudiante en una sesión."""

    __tablename__ = "visual_metrics"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("analysis_sessions.id")
    )
    student_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("students.id")
    )
    gaze_contact_percentage: Mapped[float] = mapped_column(Float, nullable=True)
    affirmative_gestures_count: Mapped[int] = mapped_column(Integer, nullable=True)
    body_orientation_score: Mapped[float] = mapped_column(Float, nullable=True)
    attention_score: Mapped[float] = mapped_column(Float, nullable=True)

    session: Mapped["AnalysisSession"] = relationship(
        "AnalysisSession", back_populates="visual_metrics"
    )


class AudioMetrics(Base):
    """Métricas de audio calculadas para un estudiante en una sesión."""

    __tablename__ = "audio_metrics"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("analysis_sessions.id")
    )
    student_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("students.id")
    )
    speaking_time_seconds: Mapped[float] = mapped_column(Float, nullable=True)
    turn_count: Mapped[int] = mapped_column(Integer, nullable=True)
    interruption_count: Mapped[int] = mapped_column(Integer, nullable=True)
    interrupted_count: Mapped[int] = mapped_column(Integer, nullable=True)
    avg_turn_duration: Mapped[float] = mapped_column(Float, nullable=True)

    session: Mapped["AnalysisSession"] = relationship(
        "AnalysisSession", back_populates="audio_metrics"
    )


class RubricScore(Base):
    """Puntuación de rúbrica para un estudiante en una sesión."""

    __tablename__ = "rubric_scores"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("analysis_sessions.id")
    )
    student_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("students.id")
    )
    collaboration_score: Mapped[float] = mapped_column(Float, nullable=True)
    communication_score: Mapped[float] = mapped_column(Float, nullable=True)
    responsibility_score: Mapped[float] = mapped_column(Float, nullable=True)
    leadership_score: Mapped[float] = mapped_column(Float, nullable=True)
    technical_contribution_score: Mapped[float] = mapped_column(Float, nullable=True)
    overall_score: Mapped[float] = mapped_column(Float, nullable=True)
    evaluator_type: Mapped[str] = mapped_column(String(50), default="system")
    intervention_summary: Mapped[str] = mapped_column(String, nullable=True)

    session: Mapped["AnalysisSession"] = relationship(
        "AnalysisSession", back_populates="rubric_scores"
    )


class Explanation(Base):
    """Explicación generada para un estudiante."""

    __tablename__ = "explanations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("analysis_sessions.id")
    )
    student_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("students.id")
    )
    narrative_text: Mapped[str] = mapped_column(String, nullable=True)
    strengths: Mapped[dict] = mapped_column(JSONB, nullable=True)
    improvements: Mapped[dict] = mapped_column(JSONB, nullable=True)
    generated_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)

    session: Mapped["AnalysisSession"] = relationship(
        "AnalysisSession", back_populates="explanations"
    )
