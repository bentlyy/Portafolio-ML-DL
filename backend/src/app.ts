import dotenv from 'dotenv';
import express from 'express';
import cors from 'cors';
import predictionRoutes from './routes/predictionRoutes';

dotenv.config();

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());

app.use('/api', predictionRoutes);

app.get('/', (req, res) => {
  res.json({ status: 'ok', message: 'ML Portfolio API running' });
});

app.get('/health', (req, res) => {
  res.json({ status: 'healthy', timestamp: new Date().toISOString() });
});

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});

export default app;
