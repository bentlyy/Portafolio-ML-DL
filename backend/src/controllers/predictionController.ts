import { Request, Response } from 'express';
import { predict } from '../services/mlProxyService';

export const getPrediction = async (req: Request, res: Response): Promise<void> => {
  try {
    const { modelId, data } = req.body;

    if (!modelId || typeof modelId !== 'string') {
      res.status(400).json({ error: 'Entrada inválida: se esperaba un modelId string' });
      return;
    }

    if (!data || !Array.isArray(data) || !data.every((row: unknown) => Array.isArray(row))) {
      res.status(400).json({ error: 'Entrada inválida: se esperaba data como arreglo de arreglos de números' });
      return;
    }

    const result = await predict(modelId, data);
    res.json(result);
  } catch (error: unknown) {
    if (error instanceof Error && 'response' in error) {
      const axiosError = error as { response?: { status?: number; data?: unknown } };
      if (axiosError.response?.status === 503) {
        res.status(503).json({ error: 'Servicio ML no disponible' });
        return;
      }
    }
    res.status(500).json({ error: 'Error interno del servidor durante la predicción' });
  }
};
