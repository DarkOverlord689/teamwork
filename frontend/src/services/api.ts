import axios from "axios";

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000/api/v1";

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

api.interceptors.request.use((config) => {
  const token = localStorage.getItem("access_token");
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

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

export const uploadService = {
  uploadVideo: async (file: File, groupId?: string) => {
    const formData = new FormData();
    formData.append("file", file);
    if (groupId) formData.append("group_id", groupId);
    const response = await api.post("/upload/", formData, {
      headers: { "Content-Type": "multipart/form-data" },
    });
    return response.data;
  },
};

export const reportService = {
  getReports: async (groupId: string) => {
    const response = await api.get(`/reports/${groupId}`);
    return response.data;
  },
  download: async (reportId: string, format: "pdf" | "json") => {
    const response = await api.get(`/reports/${reportId}/download?format=${format}`);
    return response.data;
  },
};

export const metricsService = {
  getStudentMetrics: async (studentId: string) => {
    const response = await api.get(`/metrics/${studentId}`);
    return response.data;
  },
};

export const validationService = {
  validate: async (analysisId: string, corrections: Record<string, unknown>) => {
    const response = await api.post(`/validate/${analysisId}`, corrections);
    return response.data;
  },
};

export default api;