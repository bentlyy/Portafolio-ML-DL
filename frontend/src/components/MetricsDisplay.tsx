import { type TrainingResult } from '../api/ml';

interface ClassificationMetrics {
  precision: number;
  recall: number;
  'f1-score': number;
  support: number;
}

interface Props {
  result: TrainingResult;
}

export default function MetricsDisplay({ result }: Props) {
  const getMetricColor = (key: string, value: number) => {
    if (['accuracy', 'precision', 'recall', 'f1', 'r2'].includes(key)) {
      if (value >= 0.9) return 'metric-excellent';
      if (value >= 0.7) return 'metric-good';
      if (value >= 0.5) return 'metric-ok';
      return 'metric-poor';
    }
    if (['mse', 'rmse', 'mae'].includes(key)) {
      if (value < 0.1) return 'metric-excellent';
      if (value < 1) return 'metric-good';
      return 'metric-ok';
    }
    return '';
  };

  const getMetricLabel = (key: string) => {
    const labels: Record<string, string> = {
      accuracy: 'Precisión',
      precision: 'Precisión',
      recall: 'Exhaustividad',
      f1: 'Puntaje F1',
      mse: 'ECM',
      rmse: 'RECM',
      mae: 'EAM',
      r2: 'R²',
      silhouette_score: 'Silueta',
      calinski_harabasz: 'Calinski-Harabasz',
      n_clusters: 'Grupos',
      n_noise: 'Puntos de Ruido',
    };
    return labels[key] || key;
  };

  return (
    <div className="metrics-display">
      <div className="metrics-header">
        <div className="metric-badge">
          <span className="label">Modelo</span>
          <span className="value">{result.model_id}</span>
        </div>
        <div className="metric-badge">
          <span className="label">Tiempo</span>
          <span className="value">{result.training_time_seconds}s</span>
        </div>
      </div>

      <div className="metrics-grid">
        {Object.entries(result.metrics).map(([key, value]) => (
          <div key={key} className={`metric-card ${getMetricColor(key, value)}`}>
            <span className="metric-label">{getMetricLabel(key)}</span>
            <span className="metric-value">{typeof value === 'number' ? value.toFixed(4) : value}</span>
          </div>
        ))}
      </div>

      {result.feature_importance && Object.keys(result.feature_importance).length > 0 && (
        <div className="feature-importance">
          <h4>Importancia de Características</h4>
          <div className="feature-bars">
            {Object.entries(result.feature_importance)
              .sort(([, a], [, b]) => Math.abs(b) - Math.abs(a))
              .slice(0, 10)
              .map(([feature, importance]) => {
                const maxVal = Math.max(...Object.values(result.feature_importance!).map(Math.abs));
                const width = maxVal > 0 ? (Math.abs(importance) / maxVal) * 100 : 0;
                return (
                  <div key={feature} className="feature-bar">
                    <span className="feature-name">{feature}</span>
                    <div className="bar-container">
                      <div
                        className={`bar ${importance >= 0 ? 'positive' : 'negative'}`}
                        style={{ width: `${width}%` }}
                      />
                    </div>
                    <span className="feature-value">{importance.toFixed(4)}</span>
                  </div>
                );
              })}
          </div>
        </div>
      )}

      {result.confusion_matrix && (
        <div className="confusion-matrix">
          <h4>Matriz de Confusión</h4>
          <div className="cm-grid">
            {result.confusion_matrix.map((row: number[], i: number) => (
              <div key={i} className="cm-row">
                {row.map((val: number, j: number) => {
                  const maxVal = Math.max(...result.confusion_matrix!.flat());
                  const intensity = maxVal > 0 ? val / maxVal : 0;
                  return (
                    <div
                      key={j}
                      className="cm-cell"
                      style={{
                        backgroundColor: `rgba(59, 130, 246, ${intensity})`,
                        border: i === j ? '2px solid #22c55e' : '1px solid #374151',
                      }}
                    >
                      {val}
                    </div>
                  );
                })}
              </div>
            ))}
          </div>
        </div>
      )}

      {result.classification_report && (
        <div className="classification-report">
          <h4>Reporte de Clasificación</h4>
          <table>
            <thead>
              <tr>
                <th>Clase</th>
                <th>Precisión</th>
                <th>Exhaustividad</th>
                <th>F1-Score</th>
                <th>Soporte</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(result.classification_report)
                .filter(([key]) => !['accuracy', 'macro avg', 'weighted avg'].includes(key))
                .map(([cls, metrics]) => (
                  <tr key={cls}>
                    <td>{cls}</td>
                    <td>{(metrics as unknown as ClassificationMetrics).precision?.toFixed(4)}</td>
                    <td>{(metrics as unknown as ClassificationMetrics).recall?.toFixed(4)}</td>
                    <td>{(metrics as unknown as ClassificationMetrics)['f1-score']?.toFixed(4)}</td>
                    <td>{(metrics as unknown as ClassificationMetrics).support}</td>
                  </tr>
                ))}
              {result.classification_report['weighted avg'] && (
                <tr className="avg-row">
                  <td>Promedio Ponderado</td>
                  <td>{result.classification_report['weighted avg'].precision?.toFixed(4)}</td>
                  <td>{result.classification_report['weighted avg'].recall?.toFixed(4)}</td>
                  <td>{result.classification_report['weighted avg']['f1-score']?.toFixed(4)}</td>
                  <td>{result.classification_report['weighted avg'].support}</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
