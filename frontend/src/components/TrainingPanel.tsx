import { useState, useEffect } from 'react';
import { type ModelInfo, type Hyperparameter } from '../api/ml';

interface Props {
  model: ModelInfo;
  onHyperparameterChange: (params: Record<string, unknown>) => void;
  onTrain: () => void;
  loading: boolean;
}

function getDefaultParams(model: ModelInfo): Record<string, unknown> {
  const defaults: Record<string, unknown> = {};
  Object.entries(model.hyperparameters).forEach(([key, hp]) => {
    defaults[key] = hp.default;
  });
  return defaults;
}

export default function TrainingPanel({ model, onHyperparameterChange, onTrain, loading }: Props) {
  const [params, setParams] = useState<Record<string, unknown>>(() => getDefaultParams(model));

  useEffect(() => {
    onHyperparameterChange(params);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const updateParam = (key: string, value: unknown) => {
    const updated = { ...params, [key]: value };
    setParams(updated);
    onHyperparameterChange(updated);
  };

  const renderInput = (key: string, hp: Hyperparameter) => {
    switch (hp.type) {
      case 'int':
        return (
          <input
            type="number"
            value={params[key] as number}
            min={hp.min}
            max={hp.max}
            onChange={e => updateParam(key, parseInt(e.target.value))}
          />
        );
      case 'float':
        return (
          <input
            type="number"
            step="0.01"
            value={params[key] as number}
            min={hp.min}
            max={hp.max}
            onChange={e => updateParam(key, parseFloat(e.target.value))}
          />
        );
      case 'choice':
        return (
          <select value={params[key] as string} onChange={e => updateParam(key, e.target.value)}>
            {hp.options?.map(opt => (
              <option key={opt} value={opt}>{opt}</option>
            ))}
          </select>
        );
      case 'string':
        return (
          <input
            type="text"
            value={params[key] as string}
            onChange={e => updateParam(key, e.target.value)}
          />
        );
      case 'bool':
        return (
          <input
            type="checkbox"
            checked={params[key] as boolean}
            onChange={e => updateParam(key, e.target.checked)}
          />
        );
      default:
        return (
          <input
            type="text"
            value={String(params[key] ?? '')}
            onChange={e => updateParam(key, e.target.value)}
          />
        );
    }
  };

  return (
    <div className="training-panel">
      <div className="model-description">
        <p>{model.description}</p>
        <div className="tags">
          <span className="tag">{model.model_type}</span>
          <span className="tag">{model.category}</span>
        </div>
      </div>

      {Object.keys(model.hyperparameters).length > 0 && (
        <div className="hyperparams">
          <h4>Hiperparámetros</h4>
          {Object.entries(model.hyperparameters).map(([key, hp]) => (
            <div key={key} className="param-row">
              <label>
                {hp.description}
              </label>
              {renderInput(key, hp)}
            </div>
          ))}
        </div>
      )}

      <button
        className="train-btn"
        onClick={onTrain}
        disabled={loading}
      >
        {loading ? 'Entrenando...' : 'Entrenar Modelo'}
      </button>
    </div>
  );
}
