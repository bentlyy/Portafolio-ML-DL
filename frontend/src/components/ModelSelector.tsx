import { type ModelInfo } from '../api/ml';

interface Props {
  models: ModelInfo[];
  categories: Record<string, string[]>;
  selectedModel: ModelInfo | null;
  onSelect: (model: ModelInfo) => void;
}

export default function ModelSelector({ models, categories, selectedModel, onSelect }: Props) {
  const getCategoryIcon = (cat: string) => {
    switch (cat) {
      case 'classification': return '🎯';
      case 'regression': return '📈';
      case 'clustering': return '🔵';
      case 'neural_network': return '🧠';
      default: return '🤖';
    }
  };

  return (
    <div className="model-selector">
      {Object.entries(categories).map(([category, modelIds]) => (
        <div key={category} className="category-group">
          <h3>
            {getCategoryIcon(category)} {category.charAt(0).toUpperCase() + category.slice(1)}
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
