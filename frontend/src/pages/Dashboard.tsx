import { useEffect, useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import {
  Container,
  Typography,
  Grid,
  Card,
  CardContent,
  Box,
  Chip,
  CircularProgress,
} from "@mui/material";
import { Bar } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";

import { setGroups, setLoading } from "../store/slices/analysisSlice";
import { groupService } from "../services/api";
import type { RootState } from "../store";

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

export default function Dashboard() {
  const dispatch = useDispatch();
  const { groups, loading } = useSelector((state: RootState) => state.analysis);
  const [stats, setStats] = useState({ total: 0, completed: 0, processing: 0, pending: 0 });

  useEffect(() => {
    loadGroups();
  }, []);

  const loadGroups = async () => {
    dispatch(setLoading(true));
    try {
      const data = await groupService.list();
      dispatch(setGroups(data));
      calculateStats(data);
    } catch (error) {
      console.error("Error loading groups:", error);
    } finally {
      dispatch(setLoading(false));
    }
  };

  const calculateStats = (groups: unknown[]) => {
    setStats({
      total: groups.length,
      completed: groups.filter((g: unknown) => (g as { status?: string }).status === "completed").length,
      processing: groups.filter((g: unknown) => (g as { status?: string }).status === "processing").length,
      pending: groups.filter((g: unknown) => (g as { status?: string }).status === "pending").length,
    });
  };

  const chartData = {
    labels: ["Total", "Completados", "Procesando", "Pendientes"],
    datasets: [
      {
        label: "Grupos",
        data: [stats.total, stats.completed, stats.processing, stats.pending],
        backgroundColor: ["#1976d2", "#4caf50", "#ff9800", "#9e9e9e"],
      },
    ],
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
      <Typography variant="h4" gutterBottom>
        Dashboard - SMATC-UPAO
      </Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 4 }}>
        Sistema Multimodal de Análisis de Trabajo Colaborativo
      </Typography>

      <Grid container spacing={3} sx={{ mb: 4 }}>
        {[
          { label: "Total Grupos", value: stats.total, color: "primary" },
          { label: "Completados", value: stats.completed, color: "success" },
          { label: "Procesando", value: stats.processing, color: "warning" },
          { label: "Pendientes", value: stats.pending, color: "default" },
        ].map((stat) => (
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
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Distribución de Estados
              </Typography>
              <Bar data={chartData} options={{ responsive: true }} />
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Grupos Recientes
              </Typography>
              {groups.slice(0, 5).map((group) => (
                <Box key={group.id} sx={{ mb: 2, display: "flex", justifyContent: "space-between" }}>
                  <Typography variant="body2">{group.name}</Typography>
                  <Chip label={group.status || "pending"} size="small" />
                </Box>
              ))}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Container>
  );
}