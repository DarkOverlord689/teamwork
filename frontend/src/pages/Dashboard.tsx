import { useEffect, useState } from "react";
import { useDispatch } from "react-redux";
import { useNavigate } from "react-router-dom";
import {
  Container,
  Typography,
  Grid,
  Card,
  CardContent,
  Box,
  Chip,
  CircularProgress,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  IconButton,
  Tooltip,
} from "@mui/material";
import DownloadIcon from "@mui/icons-material/Download";
import { Bar } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip as ChartTooltip,
  Legend,
} from "chart.js";

import { setLoading } from "../store/slices/analysisSlice";
import { showSnackbar } from "../store/slices/uiSlice";
import { teacherService, reportService } from "../services/api";

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, ChartTooltip, Legend);

interface DashboardData {
  total_groups: number;
  completed_sessions: number;
  avg_participation_cv: number;
  avg_rubric_overall: number;
  groups_by_status: {
    completed: number;
    processing: number;
    pending: number;
    error: number;
  };
  recent_sessions: Array<{
    session_id: string;
    group_name: string;
    status: string;
    processed_at: string | null;
    overall_score: number;
  }>;
}

const STATUS_COLORS: Record<string, "success" | "warning" | "default" | "error"> = {
  completed: "success",
  processing: "warning",
  pending: "default",
  error: "error",
};

export default function Dashboard() {
  const dispatch = useDispatch();
  const navigate = useNavigate();
  const [dashboardData, setDashboardData] = useState<DashboardData | null>(null);
  const [loading, setLocalLoading] = useState(true);
  const [downloadingId, setDownloadingId] = useState<string | null>(null);

  useEffect(() => {
    loadDashboard();
  }, []);

  const loadDashboard = async () => {
    dispatch(setLoading(true));
    setLocalLoading(true);
    try {
      const data: DashboardData = await teacherService.getDashboard();
      setDashboardData(data);
    } catch (error) {
      console.error("Error loading dashboard:", error);
      dispatch(showSnackbar({ message: "Error al cargar el dashboard", severity: "error" }));
    } finally {
      dispatch(setLoading(false));
      setLocalLoading(false);
    }
  };

  const handleDownloadReport = async (sessionId: string) => {
    setDownloadingId(sessionId);
    try {
      await reportService.download(sessionId, "pdf");
      dispatch(showSnackbar({ message: "Descarga iniciada", severity: "success" }));
    } catch (error) {
      console.error("Error downloading report:", error);
      dispatch(showSnackbar({ message: "Error al descargar el reporte", severity: "error" }));
    } finally {
      setDownloadingId(null);
    }
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress />
      </Box>
    );
  }

  const stats = dashboardData?.groups_by_status ?? { completed: 0, processing: 0, pending: 0, error: 0 };

  const chartData = {
    labels: ["Completados", "Procesando", "Pendientes", "Error"],
    datasets: [
      {
        label: "Sesiones",
        data: [stats.completed, stats.processing, stats.pending, stats.error],
        backgroundColor: ["#4caf50", "#ff9800", "#9e9e9e", "#f44336"],
      },
    ],
  };

  const metricCards = [
    {
      label: "Total Grupos",
      value: dashboardData?.total_groups ?? 0,
    },
    {
      label: "Sesiones Completadas",
      value: dashboardData?.completed_sessions ?? 0,
    },
    {
      label: "CV Participación Promedio",
      value: dashboardData?.avg_participation_cv?.toFixed(3) ?? "—",
    },
    {
      label: "Puntaje Rúbrica Promedio",
      value: dashboardData?.avg_rubric_overall?.toFixed(2) ?? "—",
    },
  ];

  return (
    <Container maxWidth="lg" sx={{ mt: 4 }}>
      <Typography variant="h4" gutterBottom>
        Dashboard - SMATC-UPAO
      </Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 4 }}>
        Sistema Multimodal de Análisis de Trabajo Colaborativo
      </Typography>

      <Grid container spacing={3} sx={{ mb: 4 }}>
        {metricCards.map((stat) => (
          <Grid item xs={12} sm={6} md={3} key={stat.label}>
            <Card>
              <CardContent>
                <Typography color="text.secondary" variant="overline">
                  {stat.label}
                </Typography>
                <Typography variant="h4">{stat.value}</Typography>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      <Grid container spacing={3}>
        <Grid item xs={12} md={5}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Distribución de Estados
              </Typography>
              <Bar
                data={chartData}
                options={{
                  responsive: true,
                  plugins: { legend: { display: false } },
                }}
              />
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={7}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Sesiones Recientes
              </Typography>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Grupo</TableCell>
                    <TableCell>Estado</TableCell>
                    <TableCell>Fecha</TableCell>
                    <TableCell align="center">Puntaje</TableCell>
                    <TableCell align="center">Reporte</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {(dashboardData?.recent_sessions ?? []).length === 0 ? (
                    <TableRow>
                      <TableCell colSpan={5} align="center">
                        <Typography variant="body2" color="text.secondary">
                          Sin sesiones recientes
                        </Typography>
                      </TableCell>
                    </TableRow>
                  ) : (
                    (dashboardData?.recent_sessions ?? []).map((s) => (
                      <TableRow
                        key={s.session_id}
                        hover
                        sx={{ cursor: "pointer" }}
                      >
                        <TableCell
                          onClick={() => navigate(`/groups/${s.session_id}`)}
                        >
                          {s.group_name}
                        </TableCell>
                        <TableCell>
                          <Chip
                            label={s.status}
                            color={STATUS_COLORS[s.status] ?? "default"}
                            size="small"
                          />
                        </TableCell>
                        <TableCell>
                          {s.processed_at
                            ? new Date(s.processed_at).toLocaleDateString("es-PE")
                            : "—"}
                        </TableCell>
                        <TableCell align="center">{s.overall_score?.toFixed(2)}</TableCell>
                        <TableCell align="center">
                          <Tooltip title="Descargar PDF">
                            <span>
                              <IconButton
                                size="small"
                                disabled={s.status !== "completed" || downloadingId === s.session_id}
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handleDownloadReport(s.session_id);
                                }}
                              >
                                <DownloadIcon fontSize="small" />
                              </IconButton>
                            </span>
                          </Tooltip>
                        </TableCell>
                      </TableRow>
                    ))
                  )}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Container>
  );
}
