import express from 'express';
import cors from 'cors';
import predictionRoutes from './routes/predictionRoutes';

const app = express();

app.use(cors());
app.use(express.json());

app.use('/api', predictionRoutes);

app.get('/', (req, res) => {
  res.send('ML Portfolio API running');
});

app.listen(3000, () => {
  console.log('CAMBIO EN BACKEND: Server running on port 3000');
});