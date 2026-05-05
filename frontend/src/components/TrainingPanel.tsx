import { useState, useEffect } from 'react';
import { type ModelInfo, type Hyperparameter } from '../api/ml';

interface Props {
  model: ModelInfo;
  hyperparameters: Record<string, unknown>;
  onHyperparameterChange: (params: Record<string, unknown>) => void;
  onTrain: () => void;
  loading: boolean;
}

export default function TrainingPanel({ model, hyperparameters, onHyperparameterChange, onTrain, loading }: Props) {
  const [params, setParams] = useState<Record<string, unknown>>({});

  useEffect(() => {
    const defaults: Record<string, unknown> = {};
    Object.entries(model.hyperparameters).forEach(([key, hp]) => {
      defaults[key] = hp.default;
    });
    setParams(defaults);
    onHyperparameterChange(defaults);
  }, [model]);

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
          <h4>Hyperparameters</h4>
          {Object.entries(model.hyperparameters).map(([key, hp]) => (
            <div key={key} className="param-row">
              <label>
                {key}
                <span className="param-desc">{hp.description}</span>
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
        {loading ? 'Training...' : 'Train Model'}
      </button>
    </div>
  );
}
