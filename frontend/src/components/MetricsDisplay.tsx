import { type TrainingResult, type PerClassMetrics } from '../api/ml';

interface Props {
  result: TrainingResult;
}

function getMetricColor(key: string, value: number) {
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
}

function getMetricLabel(key: string) {
  const labels: Record<string, string> = {
    accuracy: 'Precisión Global',
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
}

function ClassMetricsChart({ report }: { report: Record<string, unknown> }) {
  const classes = Object.entries(report)
    .filter(([key]) => !['accuracy', 'macro avg', 'weighted avg'].includes(key))
    .map(([cls, metrics]) => ({
      class: cls,
      precision: (metrics as PerClassMetrics).precision || 0,
      recall: (metrics as PerClassMetrics).recall || 0,
      f1: (metrics as PerClassMetrics)['f1-score'] || 0,
    }));

  return (
    <div className="class-chart">
      <h4>Métricas por Clase</h4>
      <div className="class-chart-grid">
        {classes.map(({ class: cls, precision, recall, f1 }) => {
          const maxVal = 1;
          return (
            <div key={cls} className="class-chart-card">
              <div className="class-chart-title">{cls}</div>
              <div className="class-metrics-bars">
                <div className="class-bar-row">
                  <span className="class-bar-label">Precisión</span>
                  <div className="class-bar-track">
                    <div className="class-bar-fill precision" style={{ width: `${(precision / maxVal) * 100}%` }} />
                  </div>
                  <span className="class-bar-value">{(precision * 100).toFixed(1)}%</span>
                </div>
                <div className="class-bar-row">
                  <span className="class-bar-label">Recall</span>
                  <div className="class-bar-track">
                    <div className="class-bar-fill recall" style={{ width: `${(recall / maxVal) * 100}%` }} />
                  </div>
                  <span className="class-bar-value">{(recall * 100).toFixed(1)}%</span>
                </div>
                <div className="class-bar-row">
                  <span className="class-bar-label">F1</span>
                  <div className="class-bar-track">
                    <div className="class-bar-fill f1" style={{ width: `${(f1 / maxVal) * 100}%` }} />
                  </div>
                  <span className="class-bar-value">{(f1 * 100).toFixed(1)}%</span>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function ConfusionMatrixDisplay({ matrix, report }: { matrix: number[][]; report?: Record<string, unknown> }) {
  const classLabels = report
    ? Object.entries(report)
        .filter(([key]) => !['accuracy', 'macro avg', 'weighted avg'].includes(key))
        .map(([cls]) => cls)
    : matrix.map((_, i) => `Clase ${i}`);

  const maxVal = Math.max(...matrix.flat());

  return (
    <div className="confusion-matrix">
      <h4>Matriz de Confusión (Confusion Matrix)</h4>
      <div className="cm-wrapper">
        <div className="cm-corner" />
        {classLabels.map((cls) => (
          <div key={cls} className="cm-col-label">{cls}</div>
        ))}
        {matrix.map((row, i) => (
          <div key={i} className="cm-row-group">
            <div className="cm-row-label">{classLabels[i]}</div>
            {row.map((val, j) => (
              <div
                key={j}
                className="cm-cell"
                style={{
                  backgroundColor: `rgba(59, 130, 246, ${maxVal > 0 ? val / maxVal : 0})`,
                  border: i === j ? '2px solid var(--success)' : '1px solid var(--border)',
                }}
              >
                <span className="cm-cell-value">{val}</span>
              </div>
            ))}
          </div>
        ))}
      </div>
      <div className="cm-legend">
        <span><span className="cm-legend-dot correct" /> Correctos (diagonal)</span>
        <span><span className="cm-legend-dot incorrect" /> Incorrectos</span>
      </div>
    </div>
  );
}

function ClassificationReport({ report }: { report: Record<string, PerClassMetrics | number> }) {
  return (
    <div className="classification-report">
      <h4>Reporte de Clasificación (Classification Report)</h4>
      <table>
        <thead>
          <tr>
            <th>Clase (Class)</th>
            <th>Precisión (Precision)</th>
            <th>Exhaustividad (Recall)</th>
            <th>F1-Score</th>
            <th>Soporte (Support)</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(report)
            .filter(([key]) => !['accuracy', 'macro avg', 'weighted avg'].includes(key))
            .map(([cls, m]) => {
              const metrics = m as PerClassMetrics;
              return (
                <tr key={cls}>
                  <td className="class-name">{cls}</td>
                  <td>{metrics.precision?.toFixed(4)}</td>
                  <td>{metrics.recall?.toFixed(4)}</td>
                  <td>{metrics['f1-score']?.toFixed(4)}</td>
                  <td>{metrics.support}</td>
                </tr>
              );
            })}
          {(report['weighted avg'] as PerClassMetrics | undefined) && (
            <tr className="avg-row">
              <td>Promedio Ponderado (Weighted Avg)</td>
              <td>{(report['weighted avg'] as PerClassMetrics).precision?.toFixed(4)}</td>
              <td>{(report['weighted avg'] as PerClassMetrics).recall?.toFixed(4)}</td>
              <td>{(report['weighted avg'] as PerClassMetrics)['f1-score']?.toFixed(4)}</td>
              <td>{(report['weighted avg'] as PerClassMetrics).support}</td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}

function AlgorithmDetails({ result }: { result: TrainingResult }) {
  const details = result.algorithm_details as Record<string, unknown> | undefined;
  if (!details) return null;

  if (details.vectores_soporte_por_clase) {
    const support = details.vectores_soporte_por_clase as Record<string, number>;
    return (
      <div className="algorithm-details">
        <h4>Detalles del Algoritmo SVM</h4>
        <div className="algo-detail-grid">
          <div className="algo-detail-card">
            <span className="algo-detail-label">Total Vectores Soporte</span>
            <span className="algo-detail-value">{String(details.total_vectores_soporte)}</span>
          </div>
          {Object.entries(support).map(([cls, count]) => (
            <div key={cls} className="algo-detail-card">
              <span className="algo-detail-label">Vectores - {cls}</span>
              <span className="algo-detail-value">{count}</span>
            </div>
          ))}
        </div>
        <p className="algo-detail-hint">
          Los vectores de soporte son las muestras más cercanas al hiperplano de separación.
          Más vectores = frontera de decisión más compleja.
        </p>
      </div>
    );
  }

  if (details.coeficientes_por_caracteristica) {
    const coefs = details.coeficientes_por_caracteristica as Record<string, number>;
    const sorted = Object.entries(coefs).sort(([, a], [, b]) => Math.abs(b) - Math.abs(a));
    const maxAbs = Math.max(...Object.values(coefs).map(Math.abs), 0.001);

    return (
      <div className="algorithm-details">
        <h4>Coeficientes del Modelo (Logistic Regression)</h4>
        <p className="algo-detail-hint">
          Coeficientes positivos = mayor probabilidad de clase. Negativos = menor probabilidad.
          Magnitud indica intensidad del efecto.
        </p>
        <div className="coef-bars">
          {sorted.slice(0, 8).map(([feature, coef]) => (
            <div key={feature} className="coef-bar-row">
              <span className="coef-name">{feature}</span>
              <div className="coef-bar-track">
                <div
                  className={`coef-bar ${coef >= 0 ? 'positive' : 'negative'}`}
                  style={{
                    width: `${(Math.abs(coef) / maxAbs) * 100}%`,
                    marginLeft: coef >= 0 ? '50%' : undefined,
                    right: coef < 0 ? '50%' : undefined,
                  }}
                />
              </div>
              <span className="coef-value">{coef.toFixed(4)}</span>
            </div>
          ))}
        </div>
      </div>
    );
  }

  return null;
}

export default function MetricsDisplay({ result }: Props) {
  return (
    <div className="metrics-display">
      <div className="metrics-header">
        <div className="metric-badge">
          <span className="label">Modelo (Model)</span>
          <span className="value">{result.model_id}</span>
        </div>
        <div className="metric-badge">
          <span className="label">Tiempo de Entrenamiento</span>
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

      <AlgorithmDetails result={result} />

      {result.feature_importance && Object.keys(result.feature_importance).length > 0 && (
        <div className="feature-importance">
          <h4>Importancia de Características (Feature Importance)</h4>
          <p className="algo-detail-hint">
            {result.model_id === 'logistic_regression'
              ? 'Magnitud promedio del coeficiente para cada característica (absoluto).'
              : 'Mide cuánto contribuye cada característica a reducir la impureza en los árboles.'}
          </p>
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

      {result.classification_report && (
        <>
          <ClassMetricsChart report={result.classification_report} />
          <ConfusionMatrixDisplay
            matrix={result.confusion_matrix!}
            report={result.classification_report}
          />
          <ClassificationReport report={result.classification_report} />
        </>
      )}
    </div>
  );
}
