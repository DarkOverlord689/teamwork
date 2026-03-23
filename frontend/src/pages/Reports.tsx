import { useState, useEffect } from "react";
import {
  Container,
  Typography,
  Grid,
  Card,
  CardContent,
  Button,
  Box,
  Chip,
} from "@mui/material";
import DownloadIcon from "@mui/icons-material/Download";
import { reportService } from "../services/api";

interface Report {
  id: string;
  group_name: string;
  status: string;
  created_at: string;
}

export default function Reports() {
  const [reports, setReports] = useState<Report[]>([]);

  useEffect(() => {
    loadReports();
  }, []);

  const loadReports = async () => {
    try {
      const data = await reportService.getReports("");
      setReports(data.reports || []);
    } catch (error) {
      console.error("Error loading reports:", error);
    }
  };

  const handleDownload = async (reportId: string, format: "pdf" | "json") => {
    try {
      const data = await reportService.download(reportId, format);
      window.open(data.download_url, "_blank");
    } catch (error) {
      console.error("Error downloading report:", error);
    }
  };

  return (
    <Container maxWidth="lg" sx={{ mt: 4 }}>
      <Typography variant="h4" gutterBottom>
        Reportes
      </Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 4 }}>
        Descargue los análisis completados
      </Typography>

      <Grid container spacing={3}>
        {reports.length === 0 ? (
          <Grid item xs={12}>
            <Card>
              <CardContent sx={{ textAlign: "center", py: 8 }}>
                <Typography color="text.secondary">
                  No hay reportes disponibles. Suba un video para comenzar el análisis.
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        ) : (
          reports.map((report) => (
            <Grid item xs={12} sm={6} md={4} key={report.id}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    {report.group_name}
                  </Typography>
                  <Box sx={{ mb: 2 }}>
                    <Chip
                      label={report.status}
                      color={report.status === "completed" ? "success" : "default"}
                      size="small"
                    />
                  </Box>
                  <Typography variant="caption" color="text.secondary" display="block">
                    {new Date(report.created_at).toLocaleDateString()}
                  </Typography>
                  <Box sx={{ mt: 2, display: "flex", gap: 1 }}>
                    <Button
                      size="small"
                      startIcon={<DownloadIcon />}
                      onClick={() => handleDownload(report.id, "pdf")}
                    >
                      PDF
                    </Button>
                    <Button
                      size="small"
                      startIcon={<DownloadIcon />}
                      onClick={() => handleDownload(report.id, "json")}
                    >
                      JSON
                    </Button>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          ))
        )}
      </Grid>
    </Container>
  );
}