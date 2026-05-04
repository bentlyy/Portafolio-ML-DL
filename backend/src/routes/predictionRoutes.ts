import { Router } from 'express';
import { getPrediction } from '../controllers/predictionController';

const router = Router();

router.post('/predict', getPrediction);

export default router;