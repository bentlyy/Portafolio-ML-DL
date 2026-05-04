import axios from 'axios';

const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://localhost:8000';

export const predict = async (input: number[]) => {
  const response = await axios.post(`${ML_SERVICE_URL}/predict`, {
    input
  });

  return response.data;
};