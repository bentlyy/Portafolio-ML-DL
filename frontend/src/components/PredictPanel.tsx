import { useState } from 'react';
import { mlApi, type PredictionResult } from '../api/ml';

interface Props {
  modelId: string;
}

export default function PredictPanel({ modelId }: Props) {
  const [inputData, setInputData] = useState('');
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handlePredict = async () => {
    if (!inputData.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const rows = inputData.trim().split('\n').map(line =>
        line.split(',').map(v => parseFloat(v.trim()))
      );
      const prediction = await mlApi.predict(modelId, rows);
      setResult(prediction);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Prediction failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="predict-panel">
      <div className="input-section">
        <label>
          Input Data (comma-separated, one sample per line):
        </label>
        <textarea
          value={inputData}
          onChange={e => setInputData(e.target.value)}
          placeholder="1.2, 3.4, 5.6, 7.8&#10;2.3, 4.5, 6.7, 8.9"
          rows={5}
        />
        <button onClick={handlePredict} disabled={loading || !inputData.trim()}>
          {loading ? 'Predicting...' : 'Predict'}
        </button>
      </div>

      {error && <div className="error-text">{error}</div>}

      {result && (
        <div className="prediction-results">
          <h4>Results</h4>
          <div className="predictions-list">
            {result.predictions.map((pred, i) => (
              <div key={i} className="prediction-item">
                <span className="sample-label">Sample {i + 1}</span>
                <span className="prediction-value">{pred}</span>
                {result.probabilities?.[i] && (
                  <div className="probabilities">
                    {result.probabilities[i].map((prob, j) => (
                      <div key={j} className="prob-bar">
                        <span className="prob-class">{result.classes?.[j] ?? j}</span>
                        <div className="prob-track">
                          <div
                            className="prob-fill"
                            style={{ width: `${prob * 100}%` }}
                          />
                        </div>
                        <span className="prob-value">{(prob * 100).toFixed(1)}%</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
