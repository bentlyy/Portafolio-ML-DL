import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:3000';

const api = axios.create({
  baseURL: `${API_URL}/api/ml`,
});

export interface ModelInfo {
  model_id: string;
  name: string;
  model_type: string;
  description: string;
  category: string;
  supported_tasks: string[];
  hyperparameters: Record<string, Hyperparameter>;
}

export interface Hyperparameter {
  type: string;
  default: string | number | boolean;
  min?: number;
  max?: number;
  options?: string[];
  description: string;
}

export interface DatasetInfo {
  filename: string;
  rows: number;
  columns: number;
  column_names: string[];
  dtypes: Record<string, string>;
  head: Record<string, unknown>[];
  describe: Record<string, unknown>;
  missing_values: Record<string, number>;
}

export interface PerClassMetrics {
  precision: number;
  recall: number;
  'f1-score': number;
  support: number;
}

export interface TrainingResult {
  model_id: string;
  metrics: Record<string, number>;
  training_time_seconds: number;
  timestamp: string;
  feature_importance?: Record<string, number>;
  confusion_matrix?: number[][];
  classification_report?: Record<string, PerClassMetrics | number>;
  algorithm_details?: Record<string, unknown>;
}

export interface PredictionResult {
  predictions: (number | string)[];
  probabilities?: number[][];
  classes?: (number | string)[];
}

export const mlApi = {
  getHealth: () => api.get('/health'),

  getModels: (): Promise<{ models: ModelInfo[]; categories: Record<string, string[]> }> =>
    api.get('/models').then(r => r.data),

  getModelInfo: (modelId: string): Promise<ModelInfo> =>
    api.get(`/models/${modelId}`).then(r => r.data),

  getHyperparameters: (modelId: string): Promise<{ model_id: string; hyperparameters: Record<string, Hyperparameter> }> =>
    api.get(`/models/${modelId}/hyperparameters`).then(r => r.data),

  uploadDataset: (file: File): Promise<DatasetInfo> => {
    const formData = new FormData();
    formData.append('file', file);
    return api.post('/upload-dataset', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    }).then(r => r.data);
  },

  trainModel: (
    modelId: string,
    file: File,
    targetColumn?: string,
    hyperparameters?: Record<string, unknown>
  ): Promise<TrainingResult> => {
    const formData = new FormData();
    formData.append('file', file);
    if (targetColumn) formData.append('target_column', targetColumn);
    if (hyperparameters) formData.append('hyperparameters', JSON.stringify(hyperparameters));
    return api.post(`/models/${modelId}/train`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    }).then(r => r.data);
  },

  predict: (modelId: string, data: number[][]): Promise<PredictionResult> =>
    api.post('/predict', { model_id: modelId, data }).then(r => r.data),

  predictFromFile: (
    modelId: string,
    file: File,
    targetColumn?: string
  ): Promise<PredictionResult> => {
    const formData = new FormData();
    formData.append('file', file);
    if (targetColumn) formData.append('target_column', targetColumn);
    return api.post(`/models/${modelId}/predict-from-file`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    }).then(r => r.data);
  },
};
