/* Upload.tsx - Página de subida de videos
 *
 * Permite al docente:
 * 1. Seleccionar un archivo de video (MP4, AVI, MOV, máx 500MB)
 * 2. Asignar un nombre opcional al grupo
 * 3. Subir el video al servidor para su análisis
 *
 * Después de subir el video, muestra el progreso del procesamiento
 * usando un stepper y polling al backend cada 3 segundos.
 */

import { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import {
  Container, Typography, Box, Paper, TextField, Button, LinearProgress,
  Alert, Card, CardContent, Divider, Stepper, Step, StepLabel, CircularProgress,
} from "@mui/material";
import CloudUploadIcon from "@mui/icons-material/CloudUpload";
import CheckCircleOutlineIcon from "@mui/icons-material/CheckCircleOutline";
import ErrorOutlineIcon from "@mui/icons-material/ErrorOutline";
import AccessTimeIcon from "@mui/icons-material/AccessTime";
import { uploadService } from "../services/api";

/* Estado de la subida: guarda la info de la sesión creada en el backend */
interface UploadStatus {
  sessionId: string;
  groupId: string;
  status: "pending" | "processing" | "completed" | "error";
  progressMessage: string;
}

/* Pasos del stepper que reflejan el progreso del análisis */
const STEPS = ["Subiendo archivo", "En cola", "Analizando video", "Completado"];

/* Texto explicativo para cada paso del procesamiento */
const STEP_SUBTEXTS: Record<number, string> = {
  1: "Tu video está en la cola de procesamiento. El análisis comenzará en breve.",
  2: "Analizando el video. Esto puede tomar varios minutos dependiendo de la duración.",
};

/* Calcula en qué paso del stepper estamos según el estado actual */
function getActiveStep(uploading: boolean, status: UploadStatus["status"] | null): number {
  if (uploading) return 0;
  if (status === "pending") return 1;
  if (status === "processing") return 2;
  if (status === "completed") return 3;
  return -1; /* estado de error */
}

export default function Upload() {
  const navigate = useNavigate();

  /* Estado del formulario de subida */
  const [file, setFile] = useState<File | null>(null);
  const [groupName, setGroupName] = useState("");
  const [uploading, setUploading] = useState(false);
  const [formMessage, setFormMessage] = useState<{ type: "success" | "error"; text: string } | null>(null);

  /* Estado de la tarjeta de progreso */
  const [uploadStatus, setUploadStatus] = useState<UploadStatus | null>(null);
  const [showStatusCard, setShowStatusCard] = useState(false);

  /* Temporizador del tiempo transcurrido */
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const elapsedRef = useRef<ReturnType<typeof setInterval> | null>(null);

  /* Referencia al intervalo de polling */
  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null);

  /* Inicia el contador de tiempo transcurrido */
  const startElapsedTimer = () => {
    setElapsedSeconds(0);
    if (elapsedRef.current) clearInterval(elapsedRef.current);
    elapsedRef.current = setInterval(() => setElapsedSeconds((s) => s + 1), 1000);
  };

  /* Detiene el contador de tiempo transcurrido */
  const stopElapsedTimer = () => {
    if (elapsedRef.current) {
      clearInterval(elapsedRef.current);
      elapsedRef.current = null;
    }
  };

  /* Limpia los intervalos al desmontar el componente */
  useEffect(() => {
    return () => {
      stopElapsedTimer();
      if (pollingRef.current) clearInterval(pollingRef.current);
    };
  }, []);

  /* Detiene el contador cuando se llega a un estado terminal */
  useEffect(() => {
    if (uploadStatus?.status === "completed" || uploadStatus?.status === "error") {
      stopElapsedTimer();
    }
  }, [uploadStatus?.status]);

  /* Inicia el polling periódico al backend para conocer el estado */
  const startPolling = (sessionId: string) => {
    if (pollingRef.current) clearInterval(pollingRef.current);

    pollingRef.current = setInterval(async () => {
      try {
        const data = await uploadService.getStatus(sessionId);
        setUploadStatus({
          sessionId: data.session_id,
          groupId: data.group_id,
          status: data.status,
          progressMessage: data.progress_message,
        });

        if (data.status === "completed" || data.status === "error" || data.status === "failed") {
          if (pollingRef.current) {
            clearInterval(pollingRef.current);
            pollingRef.current = null;
          }
        }
      } catch {
        /* Si falla el polling, se ignora y se sigue intentando */
      }
    }, 3000);
  };

  /* Maneja la selección del archivo y valida formato y tamaño */
  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = event.target.files?.[0];
    if (selectedFile) {
      const validTypes = ["video/mp4", "video/avi", "video/quicktime"];
      if (!validTypes.includes(selectedFile.type)) {
        setFormMessage({ type: "error", text: "Formato no válido. Use MP4, AVI o MOV" });
        return;
      }
      if (selectedFile.size > 500 * 1024 * 1024) {
        setFormMessage({ type: "error", text: "El archivo excede 500MB" });
        return;
      }
      setFile(selectedFile);
      setFormMessage(null);
    }
  };

  /* Envía el video al backend para su análisis */
  const handleUpload = async () => {
    if (!file) {
      setFormMessage({ type: "error", text: "Seleccione un video" });
      return;
    }

    startElapsedTimer();
    setShowStatusCard(true);
    setUploading(true);
    setFormMessage(null);

    try {
      const data = await uploadService.uploadVideo(file, groupName || undefined);

      const initialStatus: UploadStatus = {
        sessionId: data.session_id,
        groupId: data.group_id,
        status: data.status ?? "pending",
        progressMessage: data.message ?? "En cola...",
      };
      setUploadStatus(initialStatus);
      startPolling(data.session_id);

      /* Limpia el formulario para permitir otra subida */
      setFile(null);
      setGroupName("");
    } catch {
      setFormMessage({ type: "error", text: "Error al subir el video. Intente de nuevo." });
    } finally {
      setUploading(false);
    }
  };

  /* Reinicia el estado para intentar una nueva subida */
  const handleRetry = () => {
    setUploadStatus(null);
    setShowStatusCard(false);
    stopElapsedTimer();
    setElapsedSeconds(0);
    if (pollingRef.current) {
      clearInterval(pollingRef.current);
      pollingRef.current = null;
    }
  };

  const activeStep = getActiveStep(uploading, uploadStatus?.status ?? null);
  const isTerminal = uploadStatus?.status === "completed" || uploadStatus?.status === "error";
  const showElapsed = (uploading || uploadStatus?.status === "pending" || uploadStatus?.status === "processing") && !isTerminal;

  const elapsedFormatted = `${Math.floor(elapsedSeconds / 60)}m ${elapsedSeconds % 60}s`;

  return (
    <Container maxWidth="md" sx={{ mt: 4 }}>
      <Typography variant="h4" gutterBottom>Subir Video</Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 4 }}>
        Cargue el video de la sesión de trabajo grupal para análisis
      </Typography>

      {/* Formulario de subida */}
      <Paper sx={{ p: 4, mb: 4 }}>
        {formMessage && (
          <Alert severity={formMessage.type} sx={{ mb: 3 }} onClose={() => setFormMessage(null)}>
            {formMessage.text}
          </Alert>
        )}

        <Box sx={{ mb: 3 }}>
          <TextField
            fullWidth label="Nombre del Grupo (opcional)" value={groupName}
            onChange={(e) => setGroupName(e.target.value)} variant="outlined" disabled={uploading}
          />
        </Box>

        {/* Área de arrastrar y soltar archivo */}
        <Box sx={{
          border: "2px dashed #ccc", borderRadius: 2, p: 4, textAlign: "center", mb: 3,
          cursor: uploading ? "not-allowed" : "pointer",
          "&:hover": { borderColor: "primary.main" },
        }}>
          <input type="file" accept="video/mp4,video/avi,video/quicktime"
            onChange={handleFileChange} style={{ display: "none" }} id="video-upload" disabled={uploading} />
          <label htmlFor="video-upload" style={{ cursor: uploading ? "not-allowed" : "pointer" }}>
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
            <LinearProgress />
            <Typography variant="caption" color="text.secondary">Subiendo video...</Typography>
          </Box>
        )}

        <Button variant="contained" size="large" onClick={handleUpload}
          disabled={!file || uploading} fullWidth>
          {uploading ? "Subiendo..." : "Iniciar Análisis"}
        </Button>
      </Paper>

      {/* Tarjeta de estado del análisis (aparece después de subir) */}
      {(showStatusCard || uploadStatus) && (
        <Card variant="outlined">
          <CardContent>
            <Typography variant="h6" gutterBottom>Estado del análisis</Typography>
            <Divider sx={{ mb: 3 }} />

            {activeStep >= 0 ? (
              <>
                {/* Stepper con los pasos del procesamiento */}
                <Stepper activeStep={activeStep} alternativeLabel sx={{ mb: 3 }}>
                  {STEPS.map((label, index) => (
                    <Step key={label}>
                      <StepLabel>
                        <Box sx={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 0.5 }}>
                          <span>{label}</span>
                          {activeStep === index && !isTerminal && <CircularProgress size={14} thickness={5} />}
                        </Box>
                      </StepLabel>
                    </Step>
                  ))}
                </Stepper>

                {STEP_SUBTEXTS[activeStep] && (
                  <Typography variant="body2" color="text.secondary" align="center" sx={{ mb: 2 }}>
                    {STEP_SUBTEXTS[activeStep]}
                  </Typography>
                )}

                {/* Tiempo transcurrido */}
                {showElapsed && (
                  <Box sx={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 0.5, mb: 2 }}>
                    <AccessTimeIcon fontSize="small" color="action" />
                    <Typography variant="caption" color="text.secondary">
                      Tiempo transcurrido: {elapsedFormatted}
                    </Typography>
                  </Box>
                )}

                {/* Estado completado: botón para ver resultados */}
                {uploadStatus?.status === "completed" && (
                  <Box sx={{ textAlign: "center" }}>
                    <Box sx={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 1, mb: 2 }}>
                      <CheckCircleOutlineIcon color="success" />
                      <Typography color="success.main" fontWeight="medium">{uploadStatus.progressMessage}</Typography>
                    </Box>
                    <Button variant="contained" onClick={() => navigate(`/groups/${uploadStatus.groupId}`)}>
                      Ver resultados
                    </Button>
                  </Box>
                )}
              </>
            ) : (
              /* Estado de error */
              <Box sx={{ textAlign: "center" }}>
                <Box sx={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 1, mb: 2 }}>
                  <ErrorOutlineIcon color="error" />
                  <Typography color="error.main" fontWeight="medium">
                    {uploadStatus?.progressMessage ?? "Error en el procesamiento"}
                  </Typography>
                </Box>
                <Button variant="outlined" color="error" onClick={handleRetry}>Intentar de nuevo</Button>
              </Box>
            )}

            {/* ID de la sesión */}
            {uploadStatus && (
              <Typography variant="caption" color="text.secondary" display="block" align="center" sx={{ mt: 2 }}>
                Sesión: {uploadStatus.sessionId}
              </Typography>
            )}
          </CardContent>
        </Card>
      )}
    </Container>
  );
}