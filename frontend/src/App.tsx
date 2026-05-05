import { useState, useEffect } from 'react';
import { BrowserRouter, Routes, Route, Link, useLocation } from 'react-router-dom';
import { mlApi, type ModelInfo } from './api/ml';
import './App.css';
import ModelSelector from './components/ModelSelector';
import DatasetUpload from './components/DatasetUpload';
import TrainingPanel from './components/TrainingPanel';
import MetricsDisplay from './components/MetricsDisplay';
import PredictPanel from './components/PredictPanel';

function Navigation() {
  const location = useLocation();
  return (
    <nav className="navbar">
      <div className="nav-brand">ML Portfolio</div>
      <div className="nav-links">
        <Link to="/" className={location.pathname === '/' ? 'active' : ''}>Playground</Link>
        <Link to="/models" className={location.pathname === '/models' ? 'active' : ''}>Models</Link>
      </div>
    </nav>
  );
}

function PlaygroundPage() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [categories, setCategories] = useState<Record<string, string[]>>({});
  const [selectedModel, setSelectedModel] = useState<ModelInfo | null>(null);
  const [datasetFile, setDatasetFile] = useState<File | null>(null);
  const [datasetInfo, setDatasetInfo] = useState<any>(null);
  const [targetColumn, setTargetColumn] = useState<string>('');
  const [hyperparameters, setHyperparameters] = useState<Record<string, unknown>>({});
  const [trainingResult, setTrainingResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    mlApi.getModels().then(data => {
      setModels(data.models);
      setCategories(data.categories);
    }).catch(err => console.error('Error loading models:', err));
  }, []);

  const handleFileUpload = async (file: File) => {
    setDatasetFile(file);
    setError(null);
    try {
      const info = await mlApi.uploadDataset(file);
      setDatasetInfo(info);
      if (selectedModel && selectedModel.category !== 'clustering') {
        const numericCols = Object.entries(info.dtypes)
          .filter(([, dtype]) => dtype.includes('int') || dtype.includes('float'))
          .map(([col]) => col);
        setTargetColumn(numericCols[numericCols.length - 1] || '');
      }
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Error uploading dataset');
    }
  };

  const handleTrain = async () => {
    if (!selectedModel || !datasetFile) return;
    setLoading(true);
    setError(null);
    setTrainingResult(null);
    try {
      const target = selectedModel.category === 'clustering' ? undefined : targetColumn;
      const result = await mlApi.trainModel(selectedModel.model_id, datasetFile, target, hyperparameters);
      setTrainingResult(result);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Training failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="playground">
      <div className="playground-grid">
        <div className="panel">
          <h2>1. Select Model</h2>
          <ModelSelector
            models={models}
            categories={categories}
            selectedModel={selectedModel}
            onSelect={setSelectedModel}
          />
        </div>

        <div className="panel">
          <h2>2. Upload Dataset</h2>
          <DatasetUpload
            onFileSelect={handleFileUpload}
            datasetInfo={datasetInfo}
            targetColumn={targetColumn}
            onTargetChange={setTargetColumn}
            showTarget={selectedModel?.category !== 'clustering'}
          />
        </div>

        <div className="panel">
          <h2>3. Configure & Train</h2>
          {selectedModel && (
            <TrainingPanel
              model={selectedModel}
              hyperparameters={hyperparameters}
              onHyperparameterChange={setHyperparameters}
              onTrain={handleTrain}
              loading={loading}
            />
          )}
          {!selectedModel && <p className="hint">Select a model to configure</p>}
        </div>
      </div>

      {error && <div className="error-banner">{error}</div>}

      {trainingResult && (
        <div className="panel results-panel">
          <h2>Training Results</h2>
          <MetricsDisplay result={trainingResult} />
        </div>
      )}

      {trainingResult && selectedModel?.category !== 'clustering' && (
        <div className="panel">
          <h2>4. Make Predictions</h2>
          <PredictPanel modelId={selectedModel.model_id} />
        </div>
      )}
    </div>
  );
}

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
      <h1>Available Models</h1>
      {Object.entries(categories).map(([category, modelIds]) => (
        <div key={category} className="category-section">
          <h2 className="category-title">{category.charAt(0).toUpperCase() + category.slice(1)}</h2>
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
                    <span>Tasks: {model.supported_tasks.join(', ')}</span>
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
