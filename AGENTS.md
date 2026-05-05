# AGENTS.md - ML Portfolio Project

Development guidelines for AI assistants working on this project.

## Project Overview

Full-stack ML platform with React frontend, Express backend, and FastAPI ML service. Supports 12 ML models across 4 categories (classification, regression, clustering, neural networks).

## Architecture

```
Frontend (Vite) :3001 --> Backend (Express) :3000 --> ML Service (FastAPI) :8000
```

## Tech Stack

| Component | Framework | Port |
|-----------|-----------|------|
| Frontend | React 19, TypeScript, Vite 6 | 3001 |
| Backend | Express 5, TypeScript | 3000 |
| ML Service | FastAPI, Python 3.10, scikit-learn, PyTorch | 8000 |

## Commands

### Development

```bash
# Start all services (recommended)
cd infra/docker && docker-compose up --build

# Manual start
# ML Service
cd ml-services/main-service
source venv/Scripts/activate  # Windows
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Backend
cd backend && npm run dev

# Frontend
cd frontend && npm run dev
```

### Linting & Typecheck

```bash
# Frontend
cd frontend && npm run lint

# Backend
cd backend && npx tsc --noEmit
```

### Docker

```bash
cd infra/docker
docker-compose up --build
docker-compose down
```

## Project Structure

```
ml-portafolio/
├── frontend/          # React + TypeScript (Vite)
│   ├── src/
│   │   ├── api/ml.ts              # ML API client
│   │   ├── components/           # UI components
│   │   │   ├── ModelSelector.tsx
│   │   │   ├── DatasetUpload.tsx
│   │   │   ├── TrainingPanel.tsx
│   │   │   ├── MetricsDisplay.tsx
│   │   │   └── PredictPanel.tsx
│   │   └── App.tsx
│   └── package.json
├── backend/           # Express API
│   └── src/
│       ├── app.ts
│       ├── routes/
│       └── controllers/
├── ml-services/
│   └── main-service/ # FastAPI ML service
│       ├── app/
│       │   ├── main.py          # FastAPI endpoints
│       │   └── models/
│       │       ├── base.py     # BaseModel interface
│       │       ├── registry.py # Model registry
│       │       ├── classifiers.py
│       │       ├── regressors.py
│       │       ├── clustering.py
│       │       └── neural_networks.py
└── infra/docker/    # Docker Compose
```

## Available Models

### Classification (4)
- random_forest, gradient_boosting, svm, logistic_regression

### Regression (4)
- linear_regression, ridge_regression, random_forest_regressor, gradient_boosting_regressor

### Clustering (3)
- kmeans, dbscan, agglomerative

### Neural Networks (1)
- mlp (PyTorch)

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/ml/health` | Health check |
| GET | `/api/ml/models` | List models |
| GET | `/api/ml/models/{id}` | Model info |
| POST | `/api/ml/upload-dataset` | Upload CSV |
| POST | `/api/ml/models/{id}/train` | Train model |
| POST | `/api/ml/predict` | Predict |

## Adding New Models

1. **Create model class** in appropriate `ml-services/main-service/app/models/` file:
   - Extend `BaseModel` interface
   - Implement `train()`, `predict()`, `get_hyperparameters()`

2. **Register model** in `registry.py`:
   ```python
   registry.register("model_name", ModelClass, ModelCategory)
   ```

3. **Update frontend** `src/api/ml.ts` - add model metadata if needed

## Key Conventions

- Use existing patterns in each component
- Add type hints (TypeScript frontend/backend, Python type hints)
- No comments unless explicitly requested
- Follow naming: camelCase (JS/TS), snake_case (Python)
- Environment variables via `.env` in backend
- Ports defined in docker-compose.yml

## Common Tasks

| Task | Location | Notes |
|------|----------|-------|
| Add new model | `ml-services/main-service/app/models/` | Extend BaseModel |
| Add API endpoint | `ml-services/main-service/app/main.py` | FastAPI route |
| Add frontend component | `frontend/src/components/` | TSX + CSS |
| Add backend route | `backend/src/routes/` | Express router |