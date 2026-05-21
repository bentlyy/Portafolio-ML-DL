import dotenv from 'dotenv';
import express from 'express';
import cors from 'cors';
import { createProxyMiddleware } from 'http-proxy-middleware';
import predictionRoutes from './routes/predictionRoutes';

dotenv.config();

const app = express();
const PORT = process.env.PORT || 3000;
const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://localhost:8000';

app.use(cors());

app.use('/api/ml', createProxyMiddleware({
  target: ML_SERVICE_URL,
  changeOrigin: true,
  pathRewrite: { '^/api/ml': '' },
  on: {
    error: (err, req, res) => {
      console.error('[ML Proxy] Error:', err.message);
      if ('status' in res) {
        (res as express.Response).status(503).json({ error: 'Servicio ML no disponible' });
      }
    },
  },
}));

app.use(express.json());

app.use('/api', predictionRoutes);

app.get('/', (req, res) => {
  res.json({ status: 'ok', message: 'API de ML Portafolio funcionando' });
});

app.get('/health', (req, res) => {
  res.json({ status: 'healthy', timestamp: new Date().toISOString() });
});

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});

export default app;
