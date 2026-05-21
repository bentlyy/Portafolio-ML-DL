import axios from 'axios';

const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://localhost:8000';

interface PredictionResponse {
  predictions: (number | string)[];
  probabilities?: number[][];
  classes?: (number | string)[];
}

export const predict = async (modelId: string, data: number[][]): Promise<PredictionResponse> => {
  try {
    const response = await axios.post<PredictionResponse>(`${ML_SERVICE_URL}/predict`, {
      model_id: modelId,
      data,
    });
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      throw new Error(`Error del servicio ML: ${error.message}`);
    }
    throw error;
  }
};
