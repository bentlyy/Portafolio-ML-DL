import { Request, Response } from 'express';
import { predict } from '../services/mlProxyService';

export const getPrediction = async (req: Request, res: Response) => {
  const { input } = req.body;

  const result = await predict(input);

  res.json(result);
};