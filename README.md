# ML Portfolio

A full-stack machine learning platform with interactive model training, dataset upload, and real-time predictions. Demonstrates expertise in ML algorithms, deep learning, and full-stack development.

## Architecture

```
┌─────────────┐      ┌─────────────┐      ┌─────────────────────────┐
│  Frontend   │─────▶│   Backend   │─────▶│    ML Service (FastAPI) │
│  Vite 6     │      │ Express 5   │      │                         │
│  Port 3001  │      │  Port 3000  │      │  Port 8000              │
└─────────────┘      └─────────────┘      └─────────────────────────┘
                                                  │
                                    ┌─────────────┼─────────────┐
                                    │             │             │
                              Classification  Regression   Clustering
                              Neural Network
```

## Available Models (12 total)

### Classification (4 models)
| Model | Type | Best For |
|-------|------|----------|
| Random Forest | Ensemble | Non-linear patterns, robust baseline |
| Gradient Boosting | Ensemble | Competition-grade tabular performance |
| SVM | Kernel | High-dimensional spaces, complex boundaries |
| Logistic Regression | Linear | Interpretable baseline, fast training |

### Regression (4 models)
| Model | Type | Best For |
|-------|------|----------|
| Linear Regression | Linear | Interpretable relationships |
| Ridge Regression | Linear | Multicollinearity, L2 regularization |
| Random Forest Regressor | Ensemble | Non-linear relationships |
| Gradient Boosting Regressor | Ensemble | Complex feature interactions |

### Clustering (3 models)
| Model | Type | Best For |
|-------|------|----------|
| K-Means | Centroid | Spherical clusters, fast |
| DBSCAN | Density | Arbitrary shapes, outlier detection |
| Agglomerative | Hierarchical | Tree-based cluster exploration |

### Neural Networks (1 model)
| Model | Framework | Best For |
|-------|-----------|----------|
| MLP (Multi-Layer Perceptron) | PyTorch | Deep learning fundamentals, custom architectures |

## Project Structure

```
ml-portafolio/
├── frontend/                      # React + TypeScript (Vite)
│   ├── src/
│   │   ├── api/ml.ts              # ML service API client
│   │   ├── components/
│   │   │   ├── ModelSelector.tsx  # Model category browser
│   │   │   ├── DatasetUpload.tsx  # CSV upload + preview
│   │   │   ├── TrainingPanel.tsx  # Hyperparameter config
│   │   │   ├── MetricsDisplay.tsx # Results + visualizations
│   │   │   └── PredictPanel.tsx   # Live predictions
│   │   └── App.tsx                # Main app with routing
│   └── vite.config.ts
├── backend/                       # Express API proxy
│   └── src/
│       ├── app.ts                 # Main server
│       ├── routes/
│       └── controllers/
├── ml-services/
│   └── main-service/              # Unified ML service
│       ├── app/
│       │   ├── main.py            # FastAPI app + endpoints
│       │   └── models/
│       │       ├── base.py        # BaseModel interface
│       │       ├── registry.py    # Model registry pattern
│       │       ├── classifiers.py # 4 classification models
│       │       ├── regressors.py  # 4 regression models
│       │       ├── clustering.py  # 3 clustering models
│       │       └── neural_networks.py  # PyTorch MLP
│       └── scripts/
│           └── generate_datasets.py  # Sample dataset generator
├── infra/docker/                  # Docker Compose
├── data/                          # Datasets storage
└── docs/                          # Documentation
```

## Quick Start

### Docker Compose (Recommended)

```bash
cd infra/docker
docker-compose up --build
```

- Frontend: http://localhost:3001
- Backend API: http://localhost:3000
- ML Service: http://localhost:8000
- API Docs: http://localhost:8000/docs

### Manual Setup

```bash
# 1. ML Service
cd ml-services/main-service
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload

# 2. Backend
cd backend
npm install
npm run dev

# 3. Frontend
cd frontend
npm install
npm run dev
```

## API Endpoints

### ML Service (via `/api/ml`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/ml/health` | Service health check |
| GET | `/api/ml/models` | List all available models |
| GET | `/api/ml/models/{id}` | Get model info + hyperparameters |
| POST | `/api/ml/upload-dataset` | Upload CSV, get dataset analysis |
| POST | `/api/ml/models/{id}/train` | Train model with dataset |
| POST | `/api/ml/predict` | Predict with raw data |
| POST | `/api/ml/models/{id}/predict-from-file` | Predict from CSV |

### Training Request

```bash
curl -X POST http://localhost:3000/api/ml/models/random_forest/train \
  -F "file=@iris.csv" \
  -F "target_column=species" \
  -F 'hyperparameters={"n_estimators": 200, "max_depth": 15}'
```

### Sample Datasets

Generate pre-built datasets (Iris, Wine, Breast Cancer, Titanic, Housing):

```bash
cd ml-services/main-service
python scripts/generate_datasets.py
```

Datasets saved to `ml-services/main-service/scripts/datasets/`

## Features

- **12 ML/DL Models** across 4 categories (classification, regression, clustering, neural networks)
- **Interactive Playground** - Select model → Upload dataset → Configure hyperparameters → Train → Predict
- **Real-time Metrics** - Accuracy, Precision, Recall, F1, MSE, RMSE, R², Silhouette Score
- **Visualizations** - Confusion matrix, feature importance bars, classification reports
- **CSV Upload** - Drag & drop with dataset preview, column analysis, missing value detection
- **Hyperparameter Tuning** - Configure model parameters via UI with type validation
- **PyTorch MLP** - Deep learning model with configurable architecture, dropout, batch normalization

## Tech Stack

- **Frontend**: React 19, TypeScript, Vite 6
- **Backend**: Express 5, TypeScript, Axios
- **ML Service**: FastAPI, Python 3.10, scikit-learn, XGBoost, PyTorch
- **Infrastructure**: Docker, Docker Compose
- **ML Libraries**: scikit-learn, pandas, numpy, matplotlib, seaborn, torch
