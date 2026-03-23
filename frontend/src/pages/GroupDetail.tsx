import { useState, useEffect } from "react";
import { useParams } from "react-router-dom";
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
} from "@mui/material";
import { groupService } from "../services/api";

interface StudentMetrics {
  id: string;
  name: string;
  speaking_time: number;
  turn_count: number;
  interruption_count: number;
  gaze_contact: number;
  collaboration_score: number;
  communication_score: number;
  responsibility_score: number;
  leadership_score: number;
}

export default function GroupDetail() {
  const { groupId } = useParams<{ groupId: string }>();
  const [loading, setLoading] = useState(true);
  const [group, setGroup] = useState<{ name: string; status: string } | null>(null);
  const [students, setStudents] = useState<StudentMetrics[]>([]);
  const [tab, setTab] = useState(0);

  useEffect(() => {
    if (groupId) loadGroupData();
  }, [groupId]);

  const loadGroupData = async () => {
    setLoading(true);
    try {
      const analysis = await groupService.getAnalysis(groupId!);
      setGroup(analysis.group);
      setStudents(analysis.students || []);
    } catch (error) {
      console.error("Error loading group:", error);
    } finally {
      setLoading(false);
    }
  };

  const getScoreColor = (score: number) => {
    if (score >= 4) return "success";
    if (score >= 3) return "warning";
    return "error";
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Container maxWidth="lg" sx={{ mt: 4 }}>
      <Box sx={{ mb: 3 }}>
        <Typography variant="h4">{group?.name || "Grupo"}</Typography>
        <Chip label={group?.status || "pending"} color="primary" sx={{ mt: 1 }} />
      </Box>

      <Tabs value={tab} onChange={(_, v) => setTab(v)} sx={{ mb: 3 }}>
        <Tab label="Métricas de Participación" />
        <Tab label="Puntuaciones por Rúbrica" />
        <Tab label="Análisis Narrativo" />
      </Tabs>

      {tab === 0 && (
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>Tiempo de Habla (min)</Typography>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Estudiante</TableCell>
                      <TableCell align="right">Tiempo</TableCell>
                      <TableCell align="right">Turnos</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {students.map((s) => (
                      <TableRow key={s.id}>
                        <TableCell>{s.name}</TableCell>
                        <TableCell align="right">{(s.speaking_time / 60).toFixed(1)}</TableCell>
                        <TableCell align="right">{s.turn_count}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>Interrupciones</Typography>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Estudiante</TableCell>
                      <TableCell align="right">Realizadas</TableCell>
                      <TableCell align="right">Recibidas</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {students.map((s) => (
                      <TableRow key={s.id}>
                        <TableCell>{s.name}</TableCell>
                        <TableCell align="right">{s.interruption_count}</TableCell>
                        <TableCell align="right">{s.interruption_count}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}

      {tab === 1 && (
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>Puntuaciones UPAO</Typography>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Estudiante</TableCell>
                  <TableCell align="center">Colaboración</TableCell>
                  <TableCell align="center">Comunicación</TableCell>
                  <TableCell align="center">Responsabilidad</TableCell>
                  <TableCell align="center">Liderazgo</TableCell>
                  <TableCell align="center">Promedio</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {students.map((s) => {
                  const avg = (s.collaboration_score + s.communication_score + s.responsibility_score + s.leadership_score) / 4;
                  return (
                    <TableRow key={s.id}>
                      <TableCell>{s.name}</TableCell>
                      <TableCell align="center">
                        <Chip label={s.collaboration_score.toFixed(1)} color={getScoreColor(s.collaboration_score)} size="small" />
                      </TableCell>
                      <TableCell align="center">
                        <Chip label={s.communication_score.toFixed(1)} color={getScoreColor(s.communication_score)} size="small" />
                      </TableCell>
                      <TableCell align="center">
                        <Chip label={s.responsibility_score.toFixed(1)} color={getScoreColor(s.responsibility_score)} size="small" />
                      </TableCell>
                      <TableCell align="center">
                        <Chip label={s.leadership_score.toFixed(1)} color={getScoreColor(s.leadership_score)} size="small" />
                      </TableCell>
                      <TableCell align="center">
                        <Chip label={avg.toFixed(1)} color={getScoreColor(avg)} size="small" />
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      )}

      {tab === 2 && (
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>Análisis Generado por LLM</Typography>
            {students.map((s) => (
              <Box key={s.id} sx={{ mb: 3 }}>
                <Typography variant="subtitle1" fontWeight="bold">{s.name}</Typography>
                <Typography variant="body2" color="text.secondary">
                  Análisis narrativo basado en métricas de participación...
                </Typography>
              </Box>
            ))}
          </CardContent>
        </Card>
      )}
    </Container>
  );
}