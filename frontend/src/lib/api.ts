import axios from 'axios';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export const api = axios.create({
  baseURL: API_URL,
  timeout: 300000,
});

api.interceptors.request.use((config) => {
  if (typeof window !== 'undefined') {
    const token = localStorage.getItem('token');
    if (token && config.headers) {
      config.headers.Authorization = `Bearer ${token}`;
    }
  }
  return config;
});

api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      if (typeof window !== 'undefined') {
        localStorage.removeItem('token');
        localStorage.removeItem('username');
      }
    }
    return Promise.reject(error);
  }
);
export const endpoints = {
  root: '/',
  health: '/health',

  login: '/auth/token',
  register: '/auth/register',
  me: '/auth/me',
  myStats: '/auth/me/stats',

  predict: '/analysis/predict',
  predictBatch: '/analysis/predict/batch',
  analyzeFolder: '/analysis/folder',
  history: '/analysis/history',

  analysis: (id: number) => `/analysis/${id}`,
  analysisImage: (id: number) => `/analysis/${id}/image`,

  heatmap: (id: number, download = false) =>
    `/analysis/${id}/heatmap${download ? '?download=true' : ''}`,
  saveHeatmap: (id: number) => `/analysis/${id}/heatmap/save`,

  batchStart: '/analysis/batch/start',
  batchStatus: (jobId: string) => `/analysis/batch/${jobId}`,
  batchStream: (jobId: string) => `/analysis/batch/${jobId}/stream`,

  modelInfo: '/analysis/model/info',
  availableMetrics: '/analysis/metrics/available',

  adminUsers: '/admin/users',
  adminUser: (id: number) => `/admin/users/${id}`,
  adminToggleUser: (id: number) => `/admin/users/${id}/toggle-active`,
  adminToggleAdmin: (id: number) => `/admin/users/${id}/toggle-admin`,
  adminStats: '/admin/stats',
  adminCleanup: '/admin/cleanup',
  adminReloadMetrics: '/admin/metrics/reload',
  adminOptimizeDb: '/admin/optimize-db',
  adminUploadModel: '/admin/upload-model',
  adminBootstrap: '/admin/bootstrap',
};
export interface BatchUploadOptions {
  onProgress?: (percent: number) => void;
}

export async function uploadBatch(
  files: File[],
  options?: BatchUploadOptions
) {
  const formData = new FormData();
  files.forEach(file => formData.append('files', file));

  const response = await api.post(endpoints.predictBatch, formData, {
    onUploadProgress: (progressEvent) => {
      if (progressEvent.total && options?.onProgress) {
        const percent = Math.round((progressEvent.loaded / progressEvent.total) * 100);
        options.onProgress(percent);
      }
    }
  });

  return response.data;
}

export interface FolderAnalysisOptions {
  recursive?: boolean;
  maxImages?: number;
}

export async function analyzeFolder(
  path: string,
  options?: FolderAnalysisOptions
) {
  const response = await api.post(endpoints.analyzeFolder, {
    path,
    recursive: options?.recursive ?? false,
    max_images: options?.maxImages ?? 100
  });

  return response.data;
}