/* Reports.tsx - Página de reportes de análisis
 *
 * Muestra todos los reportes generados agrupados por grupo.
 * Cada reporte se presenta en una tarjeta con:
 * - Nombre del grupo y estado del análisis
 * - Fecha de creación y duración del video
 * - Botones para descargar en PDF o JSON
 * - Enlace para ver el análisis completo
 *
 * Los reportes se cargan recorriendo todos los grupos registrados.
 */

import { useState, useEffect } from "react";
import { useDispatch } from "react-redux";
import { useNavigate } from "react-router-dom";
import {
  Container, Typography, Grid, Card, CardContent, Button, Box,
  Chip, CircularProgress, Alert,
} from "@mui/material";
import DownloadIcon from "@mui/icons-material/Download";
import OpenInNewIcon from "@mui/icons-material/OpenInNew";

import { showSnackbar } from "../store/slices/uiSlice";
import { reportService, groupService } from "../services/api";

/* Datos de un reporte */
interface Report {
  id: string;
  group_id: string;
  group_name: string;
  status: string;
  created_at: string | null;
  duration_seconds: number | null;
}

/* Colores de los chips según el estado del reporte */
const STATUS_COLORS: Record<string, "success" | "warning" | "default" | "error"> = {
  completed: "success",
  processing: "warning",
  pending: "default",
  error: "error",
};

export default function Reports() {
  const dispatch = useDispatch();
  const navigate = useNavigate();
  const [reports, setReports] = useState<Report[]>([]);
  const [loading, setLoading] = useState(true);
  const [downloadingId, setDownloadingId] = useState<string | null>(null);

  useEffect(() => {
    loadReports();
  }, []);

  /* Carga todos los reportes de todos los grupos */
  const loadReports = async () => {
    setLoading(true);
    try {
      const groups = await groupService.list();
      const groupList = Array.isArray(groups) ? groups : groups.groups ?? [];

      const allReports: Report[] = [];
      await Promise.all(
        groupList.map(async (g: { id: string }) => {
          try {
            const data = await reportService.getReports(g.id);
            const items: Report[] = data.reports ?? [];
            allReports.push(...items);
          } catch {
            /* si un grupo falla, se salta y sigue con los demás */
          }
        }),
      );

      /* Ordena los reportes del más reciente al más antiguo */
      allReports.sort((a, b) => {
        if (!a.created_at) return 1;
        if (!b.created_at) return -1;
        return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
      });

      setReports(allReports);
    } catch (error) {
      console.error("Error loading reports:", error);
      dispatch(showSnackbar({ message: "Error al cargar los reportes", severity: "error" }));
    } finally {
      setLoading(false);
    }
  };

  /* Descarga un reporte en el formato especificado */
  const handleDownload = async (reportId: string, format: "pdf" | "json") => {
    setDownloadingId(`${reportId}-${format}`);
    try {
      await reportService.download(reportId, format);
      dispatch(showSnackbar({ message: `Descarga ${format.toUpperCase()} iniciada`, severity: "success" }));
    } catch (error) {
      console.error("Error downloading report:", error);
      dispatch(showSnackbar({ message: "Error al descargar el reporte", severity: "error" }));
    } finally {
      setDownloadingId(null);
    }
  };

  /* Pantalla de carga */
  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Container maxWidth="lg" sx={{ mt: 4 }}>
      <Typography variant="h4" gutterBottom>Reportes</Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 4 }}>
        Descargue los análisis completados en formato PDF o JSON
      </Typography>

      <Grid container spacing={3}>
        {reports.length === 0 ? (
          <Grid item xs={12}>
            <Alert severity="info">
              No hay reportes disponibles. Suba un video y complete el análisis para generar reportes.
            </Alert>
          </Grid>
        ) : (
          reports.map((report) => (
            <Grid item xs={12} sm={6} md={4} key={report.id}>
              <Card sx={{
                height: "100%", display: "flex", flexDirection: "column",
                opacity: report.status === "completed" ? 1 : 0.7,
              }}>
                <CardContent sx={{ flexGrow: 1 }}>
                  <Typography variant="h6" gutterBottom noWrap>{report.group_name}</Typography>
                  <Box sx={{ mb: 1 }}>
                    <Chip label={report.status} color={STATUS_COLORS[report.status] ?? "default"} size="small" />
                  </Box>
                  <Typography variant="caption" color="text.secondary" display="block">
                    {report.created_at ? new Date(report.created_at).toLocaleString("es-PE") : "Fecha desconocida"}
                  </Typography>
                  {report.duration_seconds != null && (
                    <Typography variant="caption" color="text.secondary" display="block">
                      Duración: {Math.round(report.duration_seconds / 60)} min
                    </Typography>
                  )}
                </CardContent>
                <Box sx={{ px: 2, pb: 2, display: "flex", gap: 1, flexWrap: "wrap" }}>
                  <Button size="small" variant="contained" startIcon={<DownloadIcon />}
                    disabled={report.status !== "completed" || downloadingId === `${report.id}-pdf`}
                    onClick={() => handleDownload(report.id, "pdf")} color="primary">
                    PDF
                  </Button>
                  <Button size="small" variant="outlined" startIcon={<DownloadIcon />}
                    disabled={report.status !== "completed" || downloadingId === `${report.id}-json`}
                    onClick={() => handleDownload(report.id, "json")}>
                    JSON
                  </Button>
                  <Button size="small" variant="outlined" color="secondary" startIcon={<OpenInNewIcon />}
                    onClick={() => navigate(`/groups/${report.group_id}`)}>
                    Ver análisis
                  </Button>
                </Box>
              </Card>
            </Grid>
          ))
        )}
      </Grid>
    </Container>
  );
}