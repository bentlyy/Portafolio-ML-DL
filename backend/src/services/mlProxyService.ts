import axios from 'axios';

const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://localhost:8000';

interface PredictionResponse {
  prediction: number[];
}

export const predict = async (input: number[]): Promise<PredictionResponse> => {
  try {
    const response = await axios.post<PredictionResponse>(`${ML_SERVICE_URL}/predict`, {
      input,
    });
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      throw new Error(`ML service error: ${error.message}`);
    }
    throw error;
  }
};
