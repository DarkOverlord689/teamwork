import { useState } from "react";
import {
  Container,
  Typography,
  Box,
  Paper,
  TextField,
  Button,
  LinearProgress,
  Alert,
} from "@mui/material";
import CloudUploadIcon from "@mui/icons-material/CloudUpload";
import { uploadService } from "../services/api";

export default function Upload() {
  const [file, setFile] = useState<File | null>(null);
  const [groupName, setGroupName] = useState("");
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [message, setMessage] = useState<{ type: "success" | "error"; text: string } | null>(null);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = event.target.files?.[0];
    if (selectedFile) {
      const validTypes = ["video/mp4", "video/avi", "video/quicktime"];
      if (!validTypes.includes(selectedFile.type)) {
        setMessage({ type: "error", text: "Formato no válido. Use MP4, AVI o MOV" });
        return;
      }
      if (selectedFile.size > 500 * 1024 * 1024) {
        setMessage({ type: "error", text: "El archivo excede 500MB" });
        return;
      }
      setFile(selectedFile);
      setMessage(null);
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setMessage({ type: "error", text: "Seleccione un video" });
      return;
    }

    setUploading(true);
    setProgress(0);

    try {
      const interval = setInterval(() => {
        setProgress((p) => Math.min(p + 10, 90));
      }, 500);

      await uploadService.uploadVideo(file, groupName || undefined);

      clearInterval(interval);
      setProgress(100);
      setMessage({ type: "success", text: "Video subido correctamente. El procesamiento началось." });
      setFile(null);
      setGroupName("");
    } catch (error) {
      setMessage({ type: "error", text: "Error al subir el video" });
    } finally {
      setUploading(false);
    }
  };

  return (
    <Container maxWidth="md" sx={{ mt: 4 }}>
      <Typography variant="h4" gutterBottom>
        Subir Video
      </Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 4 }}>
        Cargue el video de la sesión de trabajo grupal para análisis
      </Typography>

      {message && (
        <Alert severity={message.type} sx={{ mb: 3 }} onClose={() => setMessage(null)}>
          {message.text}
        </Alert>
      )}

      <Paper sx={{ p: 4 }}>
        <Box sx={{ mb: 3 }}>
          <TextField
            fullWidth
            label="Nombre del Grupo (opcional)"
            value={groupName}
            onChange={(e) => setGroupName(e.target.value)}
            variant="outlined"
          />
        </Box>

        <Box
          sx={{
            border: "2px dashed #ccc",
            borderRadius: 2,
            p: 4,
            textAlign: "center",
            mb: 3,
            cursor: uploading ? "not-allowed" : "pointer",
            "&:hover": { borderColor: "primary.main" },
          }}
        >
          <input
            type="file"
            accept="video/mp4,video/avi,video/quicktime"
            onChange={handleFileChange}
            style={{ display: "none" }}
            id="video-upload"
            disabled={uploading}
          />
          <label htmlFor="video-upload">
            <CloudUploadIcon sx={{ fontSize: 48, color: "primary.main", mb: 1 }} />
            <Typography variant="body1">
              {file ? file.name : "Arrastre un video o haga clic para seleccionar"}
            </Typography>
            <Typography variant="caption" color="text.secondary">
              Formatos: MP4, AVI, MOV (máx 500MB)
            </Typography>
          </label>
        </Box>

        {uploading && (
          <Box sx={{ mb: 3 }}>
            <LinearProgress variant="determinate" value={progress} />
            <Typography variant="caption" color="text.secondary">
              {progress}% - Procesando video...
            </Typography>
          </Box>
        )}

        <Button
          variant="contained"
          size="large"
          onClick={handleUpload}
          disabled={!file || uploading}
          fullWidth
        >
          {uploading ? "Subiendo..." : "Iniciar Análisis"}
        </Button>
      </Paper>
    </Container>
  );
}