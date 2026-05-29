/* GroupDetail.tsx - Página de detalle de un grupo
 *
 * Muestra los resultados completos del análisis multimodal de un grupo:
 * - Pestañas: Resumen, Audit, Video, Docente
 * - Gráficos de participación, radar de rúbrica, transcripciones
 * - Línea de tiempo con timeline de audio y visión
 * - Formulario para que el docente corrija las rúbricas
 */

import { useState, useEffect } from "react";
import { useParams } from "react-router-dom";
import {
  Container, Typography, Grid, Card, CardContent, Box, Chip,
  CircularProgress, Tabs, Tab, Table, TableBody, TableCell, TableHead,
  TableRow, Accordion, AccordionSummary, AccordionDetails, TextField,
  List, ListItem, ListItemIcon, ListItemText, Alert, Paper,
} from "@mui/material";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import CheckCircleOutlineIcon from "@mui/icons-material/CheckCircleOutline";
import WarningAmberIcon from "@mui/icons-material/WarningAmber";

import { groupService } from "../services/api";
import AuditTimeline from '../components/audit/AuditTimeline';

/* Métricas por estudiante que vienen del backend */
interface StudentMetric {
  student_id: string;
  speaking_time_seconds: number;
  turn_count: number;
  interruption_count: number;
  avg_turn_duration: number;
  participation_ratio: number;
  gaze_contact_percentage: number;
  dominant_emotion: string;
  attention_score: number;
}

/* Métricas generales del grupo */
interface GroupMetrics {
  total_students: number;
  total_speaking_time: number;
  participation_cv: number;
  turn_synchronization_score: number;
  per_student_metrics: StudentMetric[];
}

/* Puntajes de rúbrica VALUE (AAC&U Teamwork) */
interface RubricScores {
  contributes_to_team_meetings: number;
  facilitates_contributions: number;
  fosters_constructive_climate: number;
  responds_to_conflict: number;
  individual_contributions_outside: number;
  overall_score?: number;
}

/* Explicación narrativa generada por el LLM */
interface Explanation {
  narrative_text: string;
  generated_by: string;
  strengths: string[];
  improvements: string[];
}

/* Respuesta completa del análisis */
interface AnalysisData {
  id: string;
  group_id: string;
  video_path: string;
  duration_seconds: number;
  processed_at: string;
  status: string;
  topic_description?: string;
  group_metrics?: GroupMetrics;
  rubric_scores?: RubricScores;
  explanation?: Explanation;
}
export default function GroupDetail() {
  const { groupId } = useParams<{ groupId: string }>();

  const [analysisData, setAnalysisData] = useState<AnalysisData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState(0);
  const [selectedSession, setSelectedSession] = useState<AnalysisData | null>(null);

  /* Carga los datos de análisis del grupo */
  useEffect(() => {
    if (!groupId) return;
    loadAnalysis(groupId);
  }, [groupId]);

  const loadAnalysis = async (gid: string) => {
    setLoading(true);
    try {
      const data: AnalysisData[] = await groupService.getAnalysis(gid);
      setAnalysisData(data);
      if (data.length > 0) {
        setSelectedSession(data[0]);
      }
    } catch (err) {
      console.error("Error loading analysis:", err);
      setError("Error al cargar los datos del grupo");
    } finally {
      setLoading(false);
    }
  };

  /* Maneja el cambio de pestaña */
  const handleTabChange = (_: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };

  /* Pantalla de carga */
  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress />
      </Box>
    );
  }

  /* Mensaje de error */
  if (error) {
    return (
      <Container maxWidth="lg" sx={{ mt: 4 }}>
        <Alert severity="error">{error}</Alert>
      </Container>
    );
  }

  /* Mensaje si no hay datos */
  if (!selectedSession) {
    return (
      <Container maxWidth="lg" sx={{ mt: 4 }}>
        <Alert severity="info">No hay datos de análisis para este grupo</Alert>
      </Container>
    );
  }

  const metrics = selectedSession.group_metrics;
  const rubric = selectedSession.rubric_scores;
  const explanation = selectedSession.explanation;

  /* Renderiza cada pestaña según su índice */
  const renderTabContent = () => {
    switch (activeTab) {
      case 0:
        return (
          <>
            {/* Tarjetas de métricas generales */}
            <Grid container spacing={3} sx={{ mb: 4 }}>
              <Grid item xs={12} sm={6} md={3}>
                <Card><CardContent>
                  <Typography color="text.secondary" variant="overline">Estudiantes</Typography>
                  <Typography variant="h4">{metrics?.total_students ?? "—"}</Typography>
                </CardContent></Card>
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <Card><CardContent>
                  <Typography color="text.secondary" variant="overline">Tiempo Total</Typography>
                  <Typography variant="h4">{metrics?.total_speaking_time?.toFixed(0) ?? "—"}s</Typography>
                </CardContent></Card>
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <Card><CardContent>
                  <Typography color="text.secondary" variant="overline">CV Participación</Typography>
                  <Typography variant="h4">{metrics?.participation_cv?.toFixed(3) ?? "—"}</Typography>
                </CardContent></Card>
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <Card><CardContent>
                  <Typography color="text.secondary" variant="overline">Sincronización</Typography>
                  <Typography variant="h4">{metrics?.turn_synchronization_score?.toFixed(2) ?? "—"}</Typography>
                </CardContent></Card>
              </Grid>
            </Grid>

            {/* Tabla de métricas por estudiante */}
            <Card sx={{ mb: 4 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>Métricas por Estudiante</Typography>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Estudiante</TableCell>
                      <TableCell align="right">Tiempo (s)</TableCell>
                      <TableCell align="right">Turnos</TableCell>
                      <TableCell align="right">Interrupciones</TableCell>
                      <TableCell align="right">Mirada (%)</TableCell>
                      <TableCell align="right">Atención</TableCell>
                      <TableCell>Emoción</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {metrics?.per_student_metrics?.map((sm) => (
                      <TableRow key={sm.student_id}>
                        <TableCell>{sm.student_id}</TableCell>
                        <TableCell align="right">{sm.speaking_time_seconds?.toFixed(1)}</TableCell>
                        <TableCell align="right">{sm.turn_count}</TableCell>
                        <TableCell align="right">{sm.interruption_count}</TableCell>
                        <TableCell align="right">{sm.gaze_contact_percentage?.toFixed(1)}%</TableCell>
                        <TableCell align="right">{sm.attention_score?.toFixed(2)}</TableCell>
                        <TableCell>
                          <Chip label={sm.dominant_emotion} size="small" variant="outlined" />
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>

            {/* Puntajes de rúbrica */}
            {rubric && (
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>Puntajes Rúbrica VALUE (AAC&U)</Typography>
                  <Table size="small">
                    <TableBody>
                      <TableRow>
                        <TableCell>Contribuye en Reuniones</TableCell>
                        <TableCell align="right">{rubric.contributes_to_team_meetings?.toFixed(1) ?? "—"}</TableCell>
                        <TableCell>Facilita Contribuciones</TableCell>
                        <TableCell align="right">{rubric.facilitates_contributions?.toFixed(1) ?? "—"}</TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Clima Constructivo</TableCell>
                        <TableCell align="right">{rubric.fosters_constructive_climate?.toFixed(1) ?? "—"}</TableCell>
                        <TableCell>Responde a Conflictos</TableCell>
                        <TableCell align="right">{rubric.responds_to_conflict?.toFixed(1) ?? "—"}</TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Contrib. Fuera de Reuniones</TableCell>
                        <TableCell align="right">{rubric.individual_contributions_outside?.toFixed(1) ?? "—"}</TableCell>
                        <TableCell>Puntaje General</TableCell>
                        <TableCell align="right">
                          <strong>{rubric.overall_score?.toFixed(1) ?? "—"}</strong>
                        </TableCell>
                      </TableRow>
                    </TableBody>
                  </Table>
                </CardContent>
              </Card>
            )}
          </>
        );

      case 1:
        /* Pestaña de auditoría: timeline y transcripciones */
        return (
          <>
            {selectedSession && (
              <AuditTimeline sessionId={selectedSession.id} />
            )}
          </>
        );

      case 2:
        /* Pestaña de video: reproductor y análisis frame a frame */
        return (
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>Reproducción de Video</Typography>
              {selectedSession?.video_path && (
                <Box sx={{ width: "100%", mb: 2 }}>
                  <video
                    controls
                    width="100%"
                    src={selectedSession.video_path}
                    style={{ borderRadius: 8 }}
                  />
                </Box>
              )}
              <Typography variant="body2" color="text.secondary">
                Duración: {selectedSession?.duration_seconds ?? 0}s | Estado: {selectedSession?.status}
              </Typography>
            </CardContent>
          </Card>
        );

      case 3:
        /* Pestaña de docente: corrección de rúbricas y notas */
        return (
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>Intervenciones por Estudiante</Typography>
                  {/* Formularios de corrección por estudiante */}
                  {metrics?.per_student_metrics?.map((sm) => (
                    <Accordion key={sm.student_id}>
                      <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                        <Typography>{sm.student_id}</Typography>
                      </AccordionSummary>
                      <AccordionDetails>
                        <Table size="small" sx={{ mb: 2 }}>
                          <TableBody>
                             <TableRow>
                               <TableCell>Contribuye en Reuniones</TableCell>
                               <TableCell>
                                 <TextField type="number" size="small" fullWidth
                                   inputProps={{ min: 0, max: 20, step: 0.5 }}
                                   defaultValue={rubric?.contributes_to_team_meetings?.toFixed(1) ?? 10} />
                               </TableCell>
                             </TableRow>
                             <TableRow>
                               <TableCell>Facilita Contribuciones</TableCell>
                               <TableCell>
                                 <TextField type="number" size="small" fullWidth
                                   inputProps={{ min: 0, max: 20, step: 0.5 }}
                                   defaultValue={rubric?.facilitates_contributions?.toFixed(1) ?? 10} />
                               </TableCell>
                             </TableRow>
                             <TableRow>
                               <TableCell>Clima Constructivo</TableCell>
                               <TableCell>
                                 <TextField type="number" size="small" fullWidth
                                   inputProps={{ min: 0, max: 20, step: 0.5 }}
                                   defaultValue={rubric?.fosters_constructive_climate?.toFixed(1) ?? 10} />
                               </TableCell>
                             </TableRow>
                             <TableRow>
                               <TableCell>Responde a Conflictos</TableCell>
                               <TableCell>
                                 <TextField type="number" size="small" fullWidth
                                   inputProps={{ min: 0, max: 20, step: 0.5 }}
                                   defaultValue={rubric?.responds_to_conflict?.toFixed(1) ?? 10} />
                               </TableCell>
                             </TableRow>
                             <TableRow>
                               <TableCell>Contrib. Fuera de Reuniones</TableCell>
                               <TableCell>
                                 <TextField type="number" size="small" fullWidth
                                   inputProps={{ min: 0, max: 20, step: 0.5 }}
                                   defaultValue={rubric?.individual_contributions_outside?.toFixed(1) ?? 10} />
                               </TableCell>
                             </TableRow>
                          </TableBody>
                        </Table>
                        <TextField label="Nota del docente" multiline rows={3} fullWidth size="small" />
                      </AccordionDetails>
                    </Accordion>
                  ))}
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} md={6}>
              <Card><CardContent>
                <Typography variant="h6" gutterBottom>Explicación Generada</Typography>
                {explanation ? (
                  <>
                    <Typography variant="body2" paragraph>{explanation.narrative_text}</Typography>
                    <Typography variant="subtitle2" sx={{ mt: 2, mb: 1 }}>Fortalezas:</Typography>
                    <List dense>
                      {explanation.strengths?.map((s, i) => (
                        <ListItem key={i}>
                          <ListItemIcon><CheckCircleOutlineIcon color="success" fontSize="small" /></ListItemIcon>
                          <ListItemText primary={s} />
                        </ListItem>
                      ))}
                    </List>
                    <Typography variant="subtitle2" sx={{ mt: 1, mb: 1 }}>Mejoras:</Typography>
                    <List dense>
                      {explanation.improvements?.map((s, i) => (
                        <ListItem key={i}>
                          <ListItemIcon><WarningAmberIcon color="warning" fontSize="small" /></ListItemIcon>
                          <ListItemText primary={s} />
                        </ListItem>
                      ))}
                    </List>
                    <Typography variant="caption" color="text.secondary">
                      Generado por: {explanation.generated_by}
                    </Typography>
                  </>
                ) : (
                  <Typography color="text.secondary">No hay explicación disponible</Typography>
                )}
              </CardContent></Card>
            </Grid>
          </Grid>
        );

      default:
        return null;
    }
  };

  return (
    <Container maxWidth="lg" sx={{ mt: 4 }}>
      <Typography variant="h4" gutterBottom>Detalle del Grupo</Typography>
      <Chip label={`Sesiones: ${analysisData.length}`} size="small" sx={{ mb: 2 }} />

      {/* Selector de sesiones si hay más de una */}
      {analysisData.length > 1 && (
        <Box sx={{ mb: 2 }}>
          {analysisData.map((session) => (
            <Chip
              key={session.id}
              label={new Date(session.processed_at).toLocaleDateString("es-PE")}
              color={selectedSession?.id === session.id ? "primary" : "default"}
              onClick={() => setSelectedSession(session)}
              sx={{ mr: 1, mb: 1 }}
            />
          ))}
        </Box>
      )}

      {/* Pestañas: Resumen, Audit, Video, Docente */}
      <Paper sx={{ mb: 3 }}>
        <Tabs value={activeTab} onChange={handleTabChange}>
          <Tab label="Resumen" />
          <Tab label="Audit" />
          <Tab label="Video" />
          <Tab label="Docente" />
        </Tabs>
      </Paper>

      {renderTabContent()}
    </Container>
  );
}