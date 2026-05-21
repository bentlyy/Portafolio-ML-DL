import { type ModelInfo } from '../api/ml';

interface Props {
  models: ModelInfo[];
  categories: Record<string, string[]>;
  selectedModel: ModelInfo | null;
  onSelect: (model: ModelInfo) => void;
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

export default function ModelSelector({ models, categories, selectedModel, onSelect }: Props) {
  return (
    <div className="model-selector">
      {Object.entries(categories).map(([category, modelIds]) => (
        <div key={category} className="category-group">
          <h3>
            {categoryIcons[category] || '🤖'} {categoryLabels[category] || category}
          </h3>
          <div className="model-list">
            {modelIds.map(id => {
              const model = models.find(m => m.model_id === id);
              if (!model) return null;
              return (
                <button
                  key={id}
                  className={`model-btn ${selectedModel?.model_id === id ? 'selected' : ''}`}
                  onClick={() => onSelect(model)}
                >
                  <span className="model-name">{model.name}</span>
                  <span className="model-type">{model.model_type}</span>
                </button>
              );
            })}
          </div>
        </div>
      ))}
    </div>
  );
}
