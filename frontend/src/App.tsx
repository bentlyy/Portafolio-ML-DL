import { useState, useEffect } from 'react';
import { BrowserRouter, Routes, Route, Link, useLocation } from 'react-router-dom';
import { mlApi, type ModelInfo, type DatasetInfo, type TrainingResult } from './api/ml';
import './App.css';
import ModelSelector from './components/ModelSelector';
import DatasetUpload from './components/DatasetUpload';
import TrainingPanel from './components/TrainingPanel';
import MetricsDisplay from './components/MetricsDisplay';
import PredictPanel from './components/PredictPanel';

import AlgorithmExplanations from './components/AlgorithmExplanations';

function Navigation() {
  const location = useLocation();
  return (
    <nav className="navbar">
      <div className="nav-brand">ML Portafolio</div>
      <div className="nav-links">
        <Link to="/" className={location.pathname === '/' ? 'active' : ''}>Laboratorio</Link>
        <Link to="/models" className={location.pathname === '/models' ? 'active' : ''}>Modelos</Link>
      </div>
    </nav>
  );
}

function PlaygroundPage() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [categories, setCategories] = useState<Record<string, string[]>>({});
  const [selectedModel, setSelectedModel] = useState<ModelInfo | null>(null);
  const [datasetFile, setDatasetFile] = useState<File | null>(null);
  const [datasetInfo, setDatasetInfo] = useState<DatasetInfo | null>(null);
  const [targetColumn, setTargetColumn] = useState<string>('');
  const [hyperparameters, setHyperparameters] = useState<Record<string, unknown>>({});
  const [trainingResult, setTrainingResult] = useState<TrainingResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    mlApi.getModels().then(data => {
      setModels(data.models);
      setCategories(data.categories);
    }).catch(err => console.error('Error al cargar modelos:', err));
  }, []);

  const autoSelectTarget = (info: DatasetInfo) => {
    const cols = info.column_names;
    setTargetColumn(cols[cols.length - 1] || '');
  };

  const handleModelSelect = (model: ModelInfo | null) => {
    setSelectedModel(model);
    if (model && datasetInfo && model.category !== 'clustering') {
      autoSelectTarget(datasetInfo);
    }
  };

  const handleFileUpload = async (file: File) => {
    setDatasetFile(file);
    setError(null);
    try {
      const info = await mlApi.uploadDataset(file);
      setDatasetInfo(info);
      if (selectedModel && selectedModel.category !== 'clustering') {
        autoSelectTarget(info);
      }
    } catch (err) {
      const axiosErr = err as { response?: { data?: { detail?: string } } };
      setError(axiosErr.response?.data?.detail || 'Error al subir dataset');
    }
  };

  const handleTrain = async () => {
    if (!selectedModel || !datasetFile) return;
    if (selectedModel.category !== 'clustering' && !targetColumn) {
      setError('Selecciona una columna objetivo antes de entrenar.');
      return;
    }
    setLoading(true);
    setError(null);
    setTrainingResult(null);
    try {
      const target = selectedModel.category === 'clustering' ? undefined : targetColumn;
      const result = await mlApi.trainModel(selectedModel.model_id, datasetFile, target, hyperparameters);
      setTrainingResult(result);
    } catch (err) {
      const axiosErr = err as { response?: { data?: { detail?: string } } };
      setError(axiosErr.response?.data?.detail || 'Entrenamiento fallido');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="playground">
      <div className="playground-grid">
        <div className="panel">
          <h2>1. Seleccionar Modelo</h2>
          <ModelSelector
            models={models}
            categories={categories}
            selectedModel={selectedModel}
            onSelect={handleModelSelect}
          />
        </div>

        <div className="panel">
          <h2>2. Subir Dataset</h2>
          <DatasetUpload
            onFileSelect={handleFileUpload}
            datasetInfo={datasetInfo}
            targetColumn={targetColumn}
            onTargetChange={setTargetColumn}
            showTarget={selectedModel?.category !== 'clustering'}
          />
          <AlgorithmExplanations model={selectedModel} />
        </div>

        <div className="panel">
          <h2>3. Configurar y Entrenar</h2>
          {selectedModel && (
            <TrainingPanel
              key={selectedModel.model_id}
              model={selectedModel}
              onHyperparameterChange={setHyperparameters}
              onTrain={handleTrain}
              loading={loading}
            />
          )}
          {!selectedModel && <p className="hint">Selecciona un modelo para configurar</p>}
        </div>
      </div>

      {error && <div className="error-banner">{error}</div>}

      {trainingResult && (
        <div className="panel results-panel">
          <h2>Resultados del Entrenamiento</h2>
          <MetricsDisplay result={trainingResult} />
        </div>
      )}

      {trainingResult && selectedModel?.category !== 'clustering' && (
        <div className="panel">
          <h2>4. Hacer Predicciones</h2>
          <PredictPanel modelId={selectedModel!.model_id} />
        </div>
      )}
    </div>
  );
}

const categoryLabels: Record<string, string> = {
  classification: 'Clasificación',
  regression: 'Regresión',
  clustering: 'Agrupamiento',
  neural_network: 'Red Neuronal',
};

const categoryIcons: Record<string, string> = {
  classification: '🎯',
  regression: '📈',
  clustering: '🔵',
  neural_network: '🧠',
};

const taskLabels: Record<string, string> = {
  binary: 'Binaria',
  multiclass: 'Multiclase',
  continuous: 'Continua',
  unsupervised: 'No supervisado',
  regression: 'Regresión',
};

function ModelsPage() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [categories, setCategories] = useState<Record<string, string[]>>({});

  useEffect(() => {
    mlApi.getModels().then(data => {
      setModels(data.models);
      setCategories(data.categories);
    });
  }, []);

  return (
    <div className="models-page">
      <h1>Modelos Disponibles</h1>
      {Object.entries(categories).map(([category, modelIds]) => (
        <div key={category} className="category-section">
          <h2 className="category-title">
            {categoryIcons[category] || '🤖'} {categoryLabels[category] || category}
          </h2>
          <div className="models-grid">
            {modelIds.map(id => {
              const model = models.find(m => m.model_id === id);
              if (!model) return null;
              return (
                <div key={id} className="model-card">
                  <h3>{model.name}</h3>
                  <span className={`badge ${model.model_type}`}>{model.model_type}</span>
                  <p>{model.description}</p>
                  <div className="model-meta">
                    <span>Tareas: {model.supported_tasks.map(t => taskLabels[t] || t).join(', ')}</span>
                    <span>Params: {Object.keys(model.hyperparameters).length}</span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      ))}
    </div>
  );
}

function App() {
  return (
    <BrowserRouter>
      <div className="app">
        <Navigation />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<PlaygroundPage />} />
            <Route path="/models" element={<ModelsPage />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}

export default App;
