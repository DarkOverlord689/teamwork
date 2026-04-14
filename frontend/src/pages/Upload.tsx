import { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import {
  Container,
  Typography,
  Box,
  Paper,
  TextField,
  Button,
  LinearProgress,
  Alert,
  Card,
  CardContent,
  Divider,
  Stepper,
  Step,
  StepLabel,
  CircularProgress,
} from "@mui/material";
import CloudUploadIcon from "@mui/icons-material/CloudUpload";
import CheckCircleOutlineIcon from "@mui/icons-material/CheckCircleOutline";
import ErrorOutlineIcon from "@mui/icons-material/ErrorOutline";
import AccessTimeIcon from "@mui/icons-material/AccessTime";
import { uploadService } from "../services/api";

interface UploadStatus {
  sessionId: string;
  groupId: string;
  status: "pending" | "processing" | "completed" | "error";
  progressMessage: string;
}

const STEPS = ["Subiendo archivo", "En cola", "Analizando video", "Completado"];

const STEP_SUBTEXTS: Record<number, string> = {
  1: "Tu video está en la cola de procesamiento. El análisis comenzará en breve.",
  2: "Analizando el video. Esto puede tomar varios minutos dependiendo de la duración.",
};

function getActiveStep(uploading: boolean, status: UploadStatus["status"] | null): number {
  if (uploading) return 0;
  if (status === "pending") return 1;
  if (status === "processing") return 2;
  if (status === "completed") return 3;
  return -1; // error state
}

export default function Upload() {
  const navigate = useNavigate();

  // Form state
  const [file, setFile] = useState<File | null>(null);
  const [groupName, setGroupName] = useState("");
  const [uploading, setUploading] = useState(false);
  const [formMessage, setFormMessage] = useState<{ type: "success" | "error"; text: string } | null>(null);

  // Status card state (persists across new uploads)
  const [uploadStatus, setUploadStatus] = useState<UploadStatus | null>(null);
  const [showStatusCard, setShowStatusCard] = useState(false);

  // Elapsed time state
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const elapsedRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Elapsed timer helpers
  const startElapsedTimer = () => {
    setElapsedSeconds(0);
    if (elapsedRef.current) clearInterval(elapsedRef.current);
    elapsedRef.current = setInterval(() => setElapsedSeconds((s) => s + 1), 1000);
  };

  const stopElapsedTimer = () => {
    if (elapsedRef.current) {
      clearInterval(elapsedRef.current);
      elapsedRef.current = null;
    }
  };

  // Clean up on unmount
  useEffect(() => {
    return () => {
      stopElapsedTimer();
      if (pollingRef.current) clearInterval(pollingRef.current);
    };
  }, []);

  // Stop elapsed timer when reaching terminal states
  useEffect(() => {
    if (uploadStatus?.status === "completed" || uploadStatus?.status === "error") {
      stopElapsedTimer();
    }
  }, [uploadStatus?.status]);

  const startPolling = (sessionId: string, _groupId: string) => {
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

        if (
          data.status === "completed" ||
          data.status === "error" ||
          data.status === "failed"
        ) {
          if (pollingRef.current) {
            clearInterval(pollingRef.current);
            pollingRef.current = null;
          }
        }
      } catch {
        // polling errors are non-fatal — keep trying
      }
    }, 3000);
  };

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
      startPolling(data.session_id, data.group_id);

      // Reset form so the user can upload another video
      setFile(null);
      setGroupName("");
    } catch {
      setFormMessage({ type: "error", text: "Error al subir el video. Intente de nuevo." });
    } finally {
      setUploading(false);
    }
  };

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
  const isTerminal =
    uploadStatus?.status === "completed" || uploadStatus?.status === "error";
  const showElapsed =
    (uploading || uploadStatus?.status === "pending" || uploadStatus?.status === "processing") &&
    !isTerminal;

  const elapsedFormatted = `${Math.floor(elapsedSeconds / 60)}m ${elapsedSeconds % 60}s`;

  return (
    <Container maxWidth="md" sx={{ mt: 4 }}>
      <Typography variant="h4" gutterBottom>
        Subir Video
      </Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 4 }}>
        Cargue el video de la sesión de trabajo grupal para análisis
      </Typography>

      {/* Upload Form */}
      <Paper sx={{ p: 4, mb: 4 }}>
        {formMessage && (
          <Alert severity={formMessage.type} sx={{ mb: 3 }} onClose={() => setFormMessage(null)}>
            {formMessage.text}
          </Alert>
        )}

        <Box sx={{ mb: 3 }}>
          <TextField
            fullWidth
            label="Nombre del Grupo (opcional)"
            value={groupName}
            onChange={(e) => setGroupName(e.target.value)}
            variant="outlined"
            disabled={uploading}
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
            <Typography variant="caption" color="text.secondary">
              Subiendo video...
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

      {/* Status Card — visible once upload has been initiated */}
      {(showStatusCard || uploadStatus) && (
        <Card variant="outlined">
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Estado del análisis
            </Typography>
            <Divider sx={{ mb: 3 }} />

            {/* Stepper */}
            {activeStep >= 0 ? (
              <>
                <Stepper activeStep={activeStep} alternativeLabel sx={{ mb: 3 }}>
                  {STEPS.map((label, index) => (
                    <Step key={label}>
                      <StepLabel>
                        <Box sx={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 0.5 }}>
                          <span>{label}</span>
                          {activeStep === index && !isTerminal && (
                            <CircularProgress size={14} thickness={5} />
                          )}
                        </Box>
                      </StepLabel>
                    </Step>
                  ))}
                </Stepper>

                {/* Active step subtext */}
                {STEP_SUBTEXTS[activeStep] && (
                  <Typography
                    variant="body2"
                    color="text.secondary"
                    align="center"
                    sx={{ mb: 2 }}
                  >
                    {STEP_SUBTEXTS[activeStep]}
                  </Typography>
                )}

                {/* Elapsed time */}
                {showElapsed && (
                  <Box
                    sx={{
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      gap: 0.5,
                      mb: 2,
                    }}
                  >
                    <AccessTimeIcon fontSize="small" color="action" />
                    <Typography variant="caption" color="text.secondary">
                      Tiempo transcurrido: {elapsedFormatted}
                    </Typography>
                  </Box>
                )}

                {/* Completed state */}
                {uploadStatus?.status === "completed" && (
                  <Box sx={{ textAlign: "center" }}>
                    <Box sx={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 1, mb: 2 }}>
                      <CheckCircleOutlineIcon color="success" />
                      <Typography color="success.main" fontWeight="medium">
                        {uploadStatus.progressMessage}
                      </Typography>
                    </Box>
                    <Button
                      variant="contained"
                      onClick={() => navigate(`/groups/${uploadStatus.groupId}`)}
                    >
                      Ver resultados
                    </Button>
                  </Box>
                )}
              </>
            ) : (
              /* Error state — activeStep === -1 */
              <Box sx={{ textAlign: "center" }}>
                <Box sx={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 1, mb: 2 }}>
                  <ErrorOutlineIcon color="error" />
                  <Typography color="error.main" fontWeight="medium">
                    {uploadStatus?.progressMessage ?? "Error en el procesamiento"}
                  </Typography>
                </Box>
                <Button variant="outlined" color="error" onClick={handleRetry}>
                  Intentar de nuevo
                </Button>
              </Box>
            )}

            {uploadStatus && (
              <Typography
                variant="caption"
                color="text.secondary"
                display="block"
                align="center"
                sx={{ mt: 2 }}
              >
                Sesión: {uploadStatus.sessionId}
              </Typography>
            )}
          </CardContent>
        </Card>
      )}
    </Container>
  );
}
