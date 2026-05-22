# ML Portfolio

![React](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=white)
![TypeScript](https://img.shields.io/badge/TypeScript-5-3178C6?logo=typescript&logoColor=white)
![Vite](https://img.shields.io/badge/Vite-6-646CFF?logo=vite&logoColor=white)
![Express](https://img.shields.io/badge/Express-5-000000?logo=express&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6-F7931E?logo=scikitlearn&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)

Plataforma full-stack de machine learning con entrenamiento interactivo de modelos, carga de datasets y predicciones en tiempo real. Demuestra experiencia en algoritmos ML, deep learning y desarrollo full-stack.

## Arquitectura

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

## Modelos Disponibles (12 en total)

### Clasificación (4 modelos)
| Modelo | Tipo | Ideal Para |
|--------|------|------------|
| Random Forest | Ensemble | Patrones no lineales, baseline robusto |
| Gradient Boosting | Ensemble | Rendimiento competitivo en tabulares |
| SVM | Kernel | Espacios de alta dimensión, fronteras complejas |
| Logistic Regression | Lineal | Baseline interpretable, entrenamiento rápido |

### Regresión (4 modelos)
| Modelo | Tipo | Ideal Para |
|--------|------|------------|
| Linear Regression | Lineal | Relaciones interpretables |
| Ridge Regression | Lineal | Multicollinealidad, regularización L2 |
| Random Forest Regressor | Ensemble | Relaciones no lineales |
| Gradient Boosting Regressor | Ensemble | Interacciones complejas entre features |

### Clustering (3 modelos)
| Modelo | Tipo | Ideal Para |
|--------|------|------------|
| K-Means | Centroide | Clusters esféricos, rápido |
| DBSCAN | Densidad | Formas arbitrarias, detección de outliers |
| Agglomerative | Jerárquico | Exploración de clusters en árbol |

### Redes Neuronales (1 modelo)
| Modelo | Framework | Ideal Para |
|--------|-----------|------------|
| MLP (Multi-Layer Perceptron) | scikit-learn | Red neuronal con capas ocultas configurables |

## Estructura del Proyecto

```
ml-portafolio/
├── frontend/                      # React + TypeScript (Vite)
│   ├── src/
│   │   ├── api/ml.ts              # Cliente API del servicio ML
│   │   ├── components/
│   │   │   ├── ModelSelector.tsx      # Navegador de categorías de modelos
│   │   │   ├── DatasetUpload.tsx      # Carga CSV + previsualización
│   │   │   ├── TrainingPanel.tsx      # Configuración de hiperparámetros
│   │   │   ├── MetricsDisplay.tsx     # Resultados + visualizaciones
│   │   │   ├── PredictPanel.tsx       # Predicciones en vivo
│   │   │   └── AlgorithmExplanations.tsx  # Explicaciones de algoritmos
│   │   └── App.tsx                # App principal con rutas
│   └── vite.config.ts
├── backend/                       # Proxy API Express
│   └── src/
│       ├── app.ts                 # Servidor principal
│       ├── routes/
│       │   └── predictionRoutes.ts
│       ├── controllers/
│       │   └── predictionController.ts
│       └── services/
│           └── mlProxyService.ts  # Proxy al ML Service
├── ml-services/
│   └── main-service/              # Servicio ML unificado
│       ├── app/
│       │   ├── main.py            # FastAPI app + endpoints
│       │   └── models/
│       │       ├── __init__.py    # Package marker
│       │       ├── base.py        # Interfaz BaseModel
│       │       ├── registry.py    # Patrón de registro de modelos
│       │       ├── classifiers.py # 4 modelos de clasificación
│       │       ├── regressors.py  # 4 modelos de regresión
│       │       ├── clustering.py  # 3 modelos de clustering
│       │       └── neural_networks.py  # MLP de scikit-learn
│       └── scripts/
│           ├── generate_datasets.py  # Generador de datasets de ejemplo
│           └── datasets/             # Datasets generados (CSV)
└── infra/docker/                  # Docker Compose
```

## Inicio Rápido

### Docker Compose (Recomendado)

```bash
cd infra/docker
docker-compose up --build
```

- Frontend: http://localhost:3001
- Backend API: http://localhost:3000
- ML Service: http://localhost:8000
- API Docs: http://localhost:8000/docs

### Instalación Manual

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

## Endpoints de API

### ML Service (vía `/api/ml`)

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| GET | `/api/ml/health` | Verificar estado del servicio |
| GET | `/api/ml/models` | Listar todos los modelos disponibles |
| GET | `/api/ml/models/{id}` | Obtener info del modelo + hiperparámetros |
| POST | `/api/ml/upload-dataset` | Cargar CSV, obtener análisis del dataset |
| POST | `/api/ml/models/{id}/train` | Entrenar modelo con dataset |
| POST | `/api/ml/predict` | Predecir con datos en bruto |
| POST | `/api/ml/models/{id}/predict-from-file` | Predecir desde CSV |

### Ejemplo de Entrenamiento

```bash
curl -X POST http://localhost:3000/api/ml/models/random_forest/train \
  -F "file=@iris.csv" \
  -F "target_column=species" \
  -F 'hyperparameters={"n_estimators": 200, "max_depth": 15}'
```

### Datasets de Ejemplo

Genera datasets preconstruidos (Iris, Wine, Breast Cancer, Titanic, Housing):

```bash
cd ml-services/main-service
python scripts/generate_datasets.py
```

Los datasets se guardan en `ml-services/main-service/scripts/datasets/`

## Funcionalidades

- **12 Modelos ML** en 4 categorías (clasificación, regresión, clustering, redes neuronales)
- **Playground Interactivo** - Seleccionar modelo → Cargar dataset → Configurar hiperparámetros → Entrenar → Predecir
- **Métricas en Tiempo Real** - Accuracy, Precision, Recall, F1, MSE, RMSE, R², Silhouette Score
- **Visualizaciones** - Matriz de confusión, barras de importancia de features, reportes de clasificación
- **Carga CSV** - Arrastrar y soltar con previsualización, análisis de columnas, detección de valores faltantes
- **Ajuste de Hiperparámetros** - Configurar parámetros del modelo vía UI con validación de tipos
- **Red Neuronal** - MLP de scikit-learn con capas ocultas configurables

## Capturas de Pantalla

> *(Agrega aquí capturas de pantalla de la aplicación)*

| Vista | Descripción |
|-------|-------------|
| ![Selector de Modelos](https://via.placeholder.com/400x250?text=Model+Selector) | Panel de selección con 12 modelos en 4 categorías |
| ![Entrenamiento](https://via.placeholder.com/400x250?text=Training) | Configuración de hiperparámetros y entrenamiento |
| ![Métricas](https://via.placeholder.com/400x250?text=Metrics) | Resultados con métricas y visualizaciones |
| ![Predicciones](https://via.placeholder.com/400x250?text=Predictions) | Predicciones en tiempo real sobre nuevos datos |

## Stack Tecnológico

- **Frontend**: React 19, TypeScript, Vite 6, React Router 7, Axios, Vitest
- **Backend**: Express 5, TypeScript, Axios, http-proxy-middleware, cors, dotenv
- **ML Service**: FastAPI, Python 3.10, scikit-learn, uvicorn
- **Infraestructura**: Docker, Docker Compose
- **Librerías ML**: scikit-learn, pandas, numpy, scipy, joblib
