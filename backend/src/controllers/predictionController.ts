import { Request, Response } from 'express';
import { predict } from '../services/mlProxyService';

export const getPrediction = async (req: Request, res: Response): Promise<void> => {
  try {
    const { input } = req.body;

    if (!input || !Array.isArray(input)) {
      res.status(400).json({ error: 'Invalid input: expected an array of numbers' });
      return;
    }

    if (!input.every((val: unknown) => typeof val === 'number')) {
      res.status(400).json({ error: 'Invalid input: all values must be numbers' });
      return;
    }

    const result = await predict(input);
    res.json(result);
  } catch (error: unknown) {
    if (error instanceof Error && 'response' in error) {
      const axiosError = error as { response?: { status?: number; data?: unknown } };
      if (axiosError.response?.status === 503) {
        res.status(503).json({ error: 'ML service unavailable' });
        return;
      }
    }
    res.status(500).json({ error: 'Internal server error during prediction' });
  }
};
