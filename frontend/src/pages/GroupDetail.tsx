import { useState, useEffect } from "react";
import { useParams } from "react-router-dom";
import { useDispatch } from "react-redux";
import {
  Container,
  Typography,
  Grid,
  Card,
  CardContent,
  Box,
  Chip,
  CircularProgress,
  Tabs,
  Tab,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  TextField,
  Button,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Alert,
  Paper,
} from "@mui/material";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import CheckCircleOutlineIcon from "@mui/icons-material/CheckCircleOutline";
import WarningAmberIcon from "@mui/icons-material/WarningAmber";

import { showSnackbar } from "../store/slices/uiSlice";
import { groupService, validationService } from "../services/api";
import type { ValidationPayload } from "../services/api";
import ParticipationChart from "../components/charts/ParticipationChart";
import RubricRadarChart from "../components/charts/RubricRadarChart";
import TranscriptView from '../components/audit/TranscriptView';
import AuditTimeline from '../components/audit/AuditTimeline';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface StudentMetric {
  student_id: string;
  speaking_time_seconds: number;
  turn_count: number;
  interruption_count: number;
  interrupted_count: number;
  participation_ratio: number;
  gaze_contact_percentage: number;
  avg_body_orientation: number;
  dominant_emotion: string;
  initiation_count: number;
  back_channel_count: number;
  intervention_summary?: string | null;
}

interface RubricScoreEntry {
  student_id: string;
  collaboration: number;
  communication: number;
  responsibility: number;
  leadership: number;
  technical_contribution: number;
}

interface ExplanationData {
  narrative_text: string;
  strengths: string[];
  improvements: string[];
  recommendations: string[];
  topic_description?: string | null;
  intervention_summaries?: Record<string, string> | null;
}

interface FusionSession {
  id: string;
  status: string;
  topic_description?: string | null;
  group_metrics?: {
    total_students: number;
    duration_seconds: number;
    participation_cv: number;
    per_student_metrics: StudentMetric[];
  };
  rubric_scores?: {
    per_student_scores: RubricScoreEntry[];
  };
  explanation?: ExplanationData;
}

interface ValidationFormState {
  [studentId: string]: {
    collaboration: string;
    communication: string;
    responsibility: string;
    leadership: string;
    technical_contribution: string;
    teacher_note: string;
  };
}

const STATUS_COLORS: Record<string, "success" | "warning" | "default" | "error"> = {
  completed: "success",
  processing: "warning",
  pending: "default",
  error: "error",
};

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function GroupDetail() {
  const { groupId } = useParams<{ groupId: string }>();
  const dispatch = useDispatch();
  const [loading, setLoading] = useState(true);
  const [sessions, setSessions] = useState<FusionSession[]>([]);
  const [activeSession, setActiveSession] = useState<FusionSession | null>(null);
  const [tab, setTab] = useState(0);
  const [validationForm, setValidationForm] = useState<ValidationFormState>({});
  const [submittingId, setSubmittingId] = useState<string | null>(null);

  useEffect(() => {
    if (groupId) loadGroupData();
  }, [groupId]);

  const loadGroupData = async () => {
    setLoading(true);
    try {
      const data = await groupService.getAnalysis(groupId!);
      const sessionList: FusionSession[] = Array.isArray(data) ? data : data.sessions ?? [];
      setSessions(sessionList);
      const completed = sessionList.find((s) => s.status === "completed") ?? sessionList[0] ?? null;
      setActiveSession(completed);
      if (completed) {
        initValidationForm(completed);
      }
    } catch (error) {
      console.error("Error loading group:", error);
      dispatch(showSnackbar({ message: "Error al cargar el grupo", severity: "error" }));
    } finally {
      setLoading(false);
    }
  };

  const initValidationForm = (session: FusionSession) => {
    const scores = session.rubric_scores?.per_student_scores ?? [];
    const initial: ValidationFormState = {};
    for (const s of scores) {
      initial[s.student_id] = {
        collaboration: s.collaboration != null ? s.collaboration.toFixed(2) : "",
        communication: s.communication != null ? s.communication.toFixed(2) : "",
        responsibility: s.responsibility != null ? s.responsibility.toFixed(2) : "",
        leadership: s.leadership != null ? s.leadership.toFixed(2) : "",
        technical_contribution: s.technical_contribution != null ? s.technical_contribution.toFixed(2) : "",
        teacher_note: "",
      };
    }
    setValidationForm(initial);
  };

  const handleValidationChange = (
    studentId: string,
    field: keyof ValidationFormState[string],
    value: string,
  ) => {
    setValidationForm((prev) => ({
      ...prev,
      [studentId]: { ...prev[studentId], [field]: value },
    }));
  };

  const handleSubmitValidation = async (studentId: string) => {
    if (!activeSession) return;
    const form = validationForm[studentId] ?? {};
    const parseScore = (v: string) => (v !== "" ? parseFloat(v) : undefined);

    const payload: ValidationPayload = {
      student_id: studentId,
      rubric_corrections: {
        collaboration: parseScore(form.collaboration),
        communication: parseScore(form.communication),
        responsibility: parseScore(form.responsibility),
        leadership: parseScore(form.leadership),
        technical_contribution: parseScore(form.technical_contribution),
      },
      teacher_note: form.teacher_note ?? "",
    };

    setSubmittingId(studentId);
    try {
      await validationService.submitCorrections(activeSession.id, payload);
      dispatch(showSnackbar({ message: "Correcciones guardadas exitosamente", severity: "success" }));
    } catch (error) {
      console.error("Error submitting validation:", error);
      dispatch(showSnackbar({ message: "Error al guardar las correcciones", severity: "error" }));
    } finally {
      setSubmittingId(null);
    }
  };

  const getScoreColor = (score: number): "success" | "warning" | "error" => {
    if (score >= 16) return "success";
    if (score >= 8) return "warning";
    return "error";
  };

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress />
      </Box>
    );
  }

  if (!activeSession) {
    return (
      <Container maxWidth="lg" sx={{ mt: 4 }}>
        <Alert severity="info">No hay sesiones disponibles para este grupo.</Alert>
      </Container>
    );
  }

  const studentMetrics = activeSession.group_metrics?.per_student_metrics ?? [];
  const rubricScores = activeSession.rubric_scores?.per_student_scores ?? [];
  const explanation = activeSession.explanation ?? {
    narrative_text: "",
    strengths: [],
    improvements: [],
    recommendations: [],
  };

  const participationStudents = studentMetrics.map((sm) => ({
    name: sm.student_id,
    speaking_time: sm.speaking_time_seconds,
  }));

  return (
    <Container maxWidth="lg" sx={{ mt: 4 }}>
      <Box sx={{ mb: 3, display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
        <Box>
          <Typography variant="h4">Detalle de Grupo</Typography>
          <Box sx={{ mt: 1, display: "flex", gap: 1, flexWrap: "wrap" }}>
            {sessions.map((s) => (
              <Chip
                key={s.id}
                label={`${s.status} - ${s.id.slice(0, 8)}...`}
                color={STATUS_COLORS[s.status] ?? "default"}
                size="small"
                onClick={() => {
                  setActiveSession(s);
                  initValidationForm(s);
                }}
                variant={activeSession.id === s.id ? "filled" : "outlined"}
              />
            ))}
          </Box>
        </Box>
      </Box>

      {/* ------------------------------------------------------------------ */}
      {/* Topic description (read-only, LLM-generated)                        */}
      {/* ------------------------------------------------------------------ */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Tema / Descripción de la Exposición
          </Typography>
          <Paper
            variant="outlined"
            sx={{ p: 2, bgcolor: "background.default", minHeight: 56 }}
          >
            <Typography variant="body1" sx={{ whiteSpace: "pre-wrap", lineHeight: 1.7 }}>
              {activeSession.topic_description ||
                activeSession.explanation?.topic_description ||
                "Pendiente de análisis..."}
            </Typography>
          </Paper>
        </CardContent>
      </Card>

      <Tabs value={tab} onChange={(_, v) => setTab(v)} sx={{ mb: 3 }}>
        <Tab label="Participación" />
        <Tab label="Rúbrica UPAO" />
        <Tab label="Análisis Narrativo" />
        <Tab label="Transcripción" />
        <Tab label="Auditoría" />
      </Tabs>

      {/* ------------------------------------------------------------------ */}
      {/* Tab 0: Participation                                                 */}
      {/* ------------------------------------------------------------------ */}
      {tab === 0 && (
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Tiempo de Habla por Estudiante
                </Typography>
                {participationStudents.length > 0 ? (
                  <ParticipationChart students={participationStudents} />
                ) : (
                  <Typography color="text.secondary">Sin datos de participación.</Typography>
                )}
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Métricas Detalladas
                </Typography>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Estudiante</TableCell>
                      <TableCell align="right">Tiempo Habla (s)</TableCell>
                      <TableCell align="right">Turnos</TableCell>
                      <TableCell align="right">Interrupciones</TableCell>
                      <TableCell align="right">Contacto Visual (%)</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {studentMetrics.length === 0 ? (
                      <TableRow>
                        <TableCell colSpan={5} align="center">
                          Sin datos
                        </TableCell>
                      </TableRow>
                    ) : (
                      studentMetrics.map((sm) => (
                        <TableRow key={sm.student_id}>
                          <TableCell>{sm.student_id}</TableCell>
                          <TableCell align="right">{sm.speaking_time_seconds.toFixed(1)}</TableCell>
                          <TableCell align="right">{sm.turn_count}</TableCell>
                          <TableCell align="right">{sm.interruption_count}</TableCell>
                          <TableCell align="right">{sm.gaze_contact_percentage.toFixed(1)}%</TableCell>
                        </TableRow>
                      ))
                    )}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          </Grid>

          {/* Intervention summaries per student (read-only, LLM-generated) */}
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Resumen de Intervención por Participante
                </Typography>
                {studentMetrics.length === 0 ? (
                  <Typography color="text.secondary">Sin participantes registrados.</Typography>
                ) : (
                  <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
                    {studentMetrics.map((sm) => {
                      const summary =
                        sm.intervention_summary ||
                        explanation.intervention_summaries?.[sm.student_id] ||
                        null;
                      return (
                        <Box key={sm.student_id}>
                          <Typography variant="subtitle2" fontWeight="bold" sx={{ mb: 0.5 }}>
                            {sm.student_id}
                          </Typography>
                          <Paper
                            variant="outlined"
                            sx={{ p: 1.5, bgcolor: "background.default", minHeight: 48 }}
                          >
                            <Typography variant="body2" sx={{ whiteSpace: "pre-wrap" }}>
                              {summary ?? "Pendiente de análisis..."}
                            </Typography>
                          </Paper>
                        </Box>
                      );
                    })}
                  </Box>
                )}
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}

      {/* ------------------------------------------------------------------ */}
      {/* Tab 1: Rubric scores + teacher validation                            */}
      {/* ------------------------------------------------------------------ */}
      {tab === 1 && (
        <Grid container spacing={3}>
          {/* Radar charts per student */}
          {rubricScores.map((rs) => (
            <Grid item xs={12} sm={6} md={4} key={rs.student_id}>
              <Card>
                <CardContent>
                  <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
                    {rs.student_id}
                  </Typography>
                  <RubricRadarChart
                    scores={{
                      collaboration: rs.collaboration,
                      communication: rs.communication,
                      responsibility: rs.responsibility,
                      leadership: rs.leadership,
                      technical_contribution: rs.technical_contribution,
                    }}
                    label={rs.student_id}
                  />
                </CardContent>
              </Card>
            </Grid>
          ))}

          {/* Summary table */}
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Tabla Resumen de Puntajes UPAO
                </Typography>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Estudiante</TableCell>
                      <TableCell align="center">Colaboración</TableCell>
                      <TableCell align="center">Comunicación</TableCell>
                      <TableCell align="center">Responsabilidad</TableCell>
                      <TableCell align="center">Liderazgo</TableCell>
                      <TableCell align="center">Contrib. Técnica</TableCell>
                      <TableCell align="center">Promedio</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {rubricScores.length === 0 ? (
                      <TableRow>
                        <TableCell colSpan={7} align="center">
                          Sin datos de rúbrica.
                        </TableCell>
                      </TableRow>
                    ) : (
                      rubricScores.map((rs) => {
                        const avg =
                          (rs.collaboration +
                            rs.communication +
                            rs.responsibility +
                            rs.leadership +
                            rs.technical_contribution) /
                          5;
                        return (
                          <TableRow key={rs.student_id}>
                            <TableCell>{rs.student_id}</TableCell>
                            {[
                              rs.collaboration,
                              rs.communication,
                              rs.responsibility,
                              rs.leadership,
                              rs.technical_contribution,
                            ].map((score, idx) => (
                              <TableCell align="center" key={idx}>
                                <Chip
                                  label={score.toFixed(1)}
                                  color={getScoreColor(score)}
                                  size="small"
                                />
                              </TableCell>
                            ))}
                            <TableCell align="center">
                              <Chip
                                label={avg.toFixed(1)}
                                color={getScoreColor(avg)}
                                size="small"
                              />
                            </TableCell>
                          </TableRow>
                        );
                      })
                    )}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          </Grid>

          {/* Teacher validation form */}
          <Grid item xs={12}>
            <Accordion>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Typography variant="h6">Corrección Docente</Typography>
              </AccordionSummary>
              <AccordionDetails>
                {rubricScores.length === 0 ? (
                  <Typography color="text.secondary">
                    No hay puntajes de rúbrica para corregir.
                  </Typography>
                ) : (
                  rubricScores.map((rs) => {
                    const form = validationForm[rs.student_id] ?? {
                      collaboration: "",
                      communication: "",
                      responsibility: "",
                      leadership: "",
                      technical_contribution: "",
                      teacher_note: "",
                    };
                    return (
                      <Box key={rs.student_id} sx={{ mb: 4 }}>
                        <Typography variant="subtitle1" fontWeight="bold" sx={{ mb: 1 }}>
                          {rs.student_id}
                        </Typography>
                        <Grid container spacing={2}>
                          {(
                            [
                              ["collaboration", "Colaboración"],
                              ["communication", "Comunicación"],
                              ["responsibility", "Responsabilidad"],
                              ["leadership", "Liderazgo"],
                              ["technical_contribution", "Contrib. Técnica"],
                            ] as [keyof typeof form, string][]
                          ).map(([field, label]) => (
                            <Grid item xs={6} sm={4} md={2} key={field}>
                              <TextField
                                label={label}
                                size="small"
                                type="number"
                                inputProps={{ min: 0, max: 20, step: 0.5 }}
                                value={form[field]}
                                onChange={(e) =>
                                  handleValidationChange(rs.student_id, field, e.target.value)
                                }
                                helperText={`Sistema: ${(rs as unknown as Record<string, number>)[field]?.toFixed(2) ?? "—"}`}
                                fullWidth
                              />
                            </Grid>
                          ))}
                          <Grid item xs={12}>
                            <TextField
                              label="Nota Docente"
                              size="small"
                              multiline
                              rows={2}
                              fullWidth
                              value={form.teacher_note}
                              onChange={(e) =>
                                handleValidationChange(rs.student_id, "teacher_note", e.target.value)
                              }
                            />
                          </Grid>
                          <Grid item xs={12}>
                            <Button
                              variant="contained"
                              disabled={submittingId === rs.student_id}
                              onClick={() => handleSubmitValidation(rs.student_id)}
                            >
                              {submittingId === rs.student_id ? "Guardando..." : "Guardar Corrección"}
                            </Button>
                          </Grid>
                        </Grid>
                      </Box>
                    );
                  })
                )}
              </AccordionDetails>
            </Accordion>
          </Grid>
        </Grid>
      )}

      {/* ------------------------------------------------------------------ */}
      {/* Tab 2: Narrative analysis                                            */}
      {/* ------------------------------------------------------------------ */}
      {tab === 2 && (
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Análisis Narrativo
                </Typography>
                <Typography variant="body1" sx={{ whiteSpace: "pre-wrap", lineHeight: 1.7 }}>
                  {explanation.narrative_text || "Sin análisis narrativo disponible."}
                </Typography>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom sx={{ color: "success.main" }}>
                  Fortalezas
                </Typography>
                {explanation.strengths.length === 0 ? (
                  <Typography color="text.secondary">Sin fortalezas registradas.</Typography>
                ) : (
                  <List dense>
                    {explanation.strengths.map((s, i) => (
                      <ListItem key={i} disableGutters>
                        <ListItemIcon sx={{ minWidth: 32 }}>
                          <CheckCircleOutlineIcon color="success" fontSize="small" />
                        </ListItemIcon>
                        <ListItemText primary={s} />
                      </ListItem>
                    ))}
                  </List>
                )}
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom sx={{ color: "warning.main" }}>
                  Áreas de Mejora
                </Typography>
                {explanation.improvements.length === 0 ? (
                  <Typography color="text.secondary">Sin áreas de mejora registradas.</Typography>
                ) : (
                  <List dense>
                    {explanation.improvements.map((s, i) => (
                      <ListItem key={i} disableGutters>
                        <ListItemIcon sx={{ minWidth: 32 }}>
                          <WarningAmberIcon color="warning" fontSize="small" />
                        </ListItemIcon>
                        <ListItemText primary={s} />
                      </ListItem>
                    ))}
                  </List>
                )}
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}

      {/* ------------------------------------------------------------------ */}
      {/* Tab 3: Transcript                                                    */}
      {/* ------------------------------------------------------------------ */}
      {tab === 3 && activeSession && <TranscriptView sessionId={activeSession.id} />}

      {/* ------------------------------------------------------------------ */}
      {/* Tab 4: Audit Timeline                                                */}
      {/* ------------------------------------------------------------------ */}
      {tab === 4 && activeSession && <AuditTimeline sessionId={activeSession.id} />}
    </Container>
  );
}
