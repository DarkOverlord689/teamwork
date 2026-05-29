/* api.ts - Servicios de comunicación con el backend
 *
 * Define los servicios para cada módulo del API:
 * - authService:    autenticación de usuarios
 * - groupService:   gestión de grupos (listar, crear, obtener análisis)
 * - uploadService:  subida de videos y consulta de estado
 * - reportService:  descarga de reportes PDF
 * - metricsService: métricas de estudiantes
 * - validationService: validación y corrección de rúbricas por el docente
 * - teacherService: dashboard y comparativa de grupos
 * - audioService:   resultados de transcripción y análisis de audio
 * - visionService:  resultados del análisis de video
 *
 * Todas las llamadas usan axios con la URL base configurada por VITE_API_URL.
 */

import axios from "axios";
import type { AudioResult, TranscriptSegment, VisionResult } from "../types/audit";

/* URL base del API: usa variable de entorno o ruta por defecto */
const API_BASE_URL = import.meta.env.VITE_API_URL || "/api/v1";

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: { "Content-Type": "application/json" },
});

/* Interceptor que agrega el token JWT a cada petición */
api.interceptors.request.use((config) => {
  const token = localStorage.getItem("access_token");
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

/* Servicio de autenticación: login con usuario y contraseña */
export const authService = {
  login: async (username: string, password: string) => {
    const formData = new FormData();
    formData.append("username", username);
    formData.append("password", password);
    const response = await api.post("/auth/login", formData, {
      headers: { "Content-Type": "multipart/form-data" },
    });
    return response.data;
  },
};

/* Servicio de grupos: listar, crear y obtener análisis */
export const groupService = {
  list: async () => {
    const response = await api.get("/analysis/groups");
    return response.data;
  },
  create: async (data: { name: string; course_id?: string }) => {
    const response = await api.post("/analysis/groups", data);
    return response.data;
  },
  getAnalysis: async (groupId: string) => {
    const response = await api.get(`/analysis/groups/${groupId}/analysis`);
    return response.data;
  },
};

/* Servicio de subida de videos */
export const uploadService = {
  uploadVideo: async (file: File, groupName?: string, groupId?: string) => {
    const formData = new FormData();
    formData.append("file", file);
    if (groupId) formData.append("group_id", groupId);
    if (groupName) formData.append("group_name", groupName);
    const response = await api.post("/upload/", formData, {
      headers: { "Content-Type": undefined },
    });
    return response.data;
  },
  getStatus: async (sessionId: string) => {
    const response = await api.get(`/upload/${sessionId}/status`);
    return response.data;
  },
};

/* Servicio de reportes: obtener y descargar en PDF */
export const reportService = {
  getReports: async (groupId: string) => {
    const response = await api.get(`/reports/${groupId}`);
    return response.data;
  },
  download: async (reportId: string, format: "pdf" | "json") => {
    const response = await api.get(`/reports/${reportId}/download`, {
      params: { format },
      responseType: "blob",
    });

    /* Extrae el nombre del archivo del header Content-Disposition */
    const contentDisposition: string = response.headers["content-disposition"] || "";
    const filenameMatch = contentDisposition.match(/filename="?([^"]+)"?/);
    const filename = filenameMatch ? filenameMatch[1] : `report.${format}`;

    /* Crea un enlace temporal y dispara la descarga en el navegador */
    const url = window.URL.createObjectURL(new Blob([response.data]));
    const link = document.createElement("a");
    link.href = url;
    link.setAttribute("download", filename);
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.URL.revokeObjectURL(url);
  },
};

/* Servicio de métricas de estudiantes */
export const metricsService = {
  getStudentMetrics: async (studentId: string) => {
    const response = await api.get(`/metrics/${studentId}`);
    return response.data;
  },
};

/* Datos que se envían al corregir una rúbrica */
export interface ValidationPayload {
  student_id: string;
  rubric_corrections: {
    contributes_to_team_meetings?: number;
    facilitates_contributions?: number;
    fosters_constructive_climate?: number;
    responds_to_conflict?: number;
    individual_contributions_outside?: number;
  };
  teacher_note: string;
}

/* Servicio de validación: obtener y enviar correcciones docentes */
export const validationService = {
  getValidation: async (sessionId: string) => {
    const response = await api.get(`/validate/${sessionId}`);
    return response.data;
  },
  submitCorrections: async (sessionId: string, corrections: ValidationPayload) => {
    const response = await api.post(`/validate/${sessionId}`, corrections);
    return response.data;
  },
};

/* Servicio de docente: dashboard y comparativa de grupos */
export const teacherService = {
  getDashboard: async () => {
    const response = await api.get("/teacher/dashboard");
    return response.data;
  },
  getGroupComparison: async (groupId: string) => {
    const response = await api.get(`/teacher/groups/${groupId}/comparison`);
    return response.data;
  },
};

/* Servicio de audio: transcripciones y resultados del análisis */
export const audioService = {
  getTranscripts: (sessionId: string): Promise<TranscriptSegment[]> =>
    api.get(`/audio/results/${sessionId}/transcripts`).then((r) => r.data),
  getResults: (sessionId: string): Promise<AudioResult> =>
    api.get(`/audio/results/${sessionId}`).then((r) => r.data),
};

/* Servicio de visión: resultados del análisis de video */
export const visionService = {
  getResults: (sessionId: string): Promise<VisionResult> =>
    api.get(`/vision/results/${sessionId}`).then((r) => r.data),
};

export default api;