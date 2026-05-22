import { useMemo } from 'react';
import type { TrainingResult } from '../api/ml';

interface Props {
  result: TrainingResult;
}

const CHART_W = 500;
const CHART_H = 300;
const PAD = { top: 20, right: 20, bottom: 45, left: 55 };
const PLOT_W = CHART_W - PAD.left - PAD.right;
const PLOT_H = CHART_H - PAD.top - PAD.bottom;

function ScatterPlot({ data, xLabel, yLabel, diagonal = false, color = '#3b82f6', title }: {
  data: { x: number; y: number }[];
  xLabel: string;
  yLabel: string;
  diagonal?: boolean;
  color?: string;
  title?: string;
}) {
    const { xMin, xMax, yMin, yMax } = useMemo(() => {
    const xs = data.map(d => d.x);
    const ys = data.map(d => d.y);
    return {
      xMin: Math.min(...xs),
      xMax: Math.max(...xs),
      yMin: Math.min(...ys),
      yMax: Math.max(...ys),
    };
  }, [data]);

  const range = Math.max(xMax - xMin, yMax - yMin, 0.001);
    const pad = range * 0.1;
    const xLo = xMin - pad;
    const xHi = xMax + pad;
    const yLo = yMin - pad;
    const yHi = yMax + pad;
    const xRng = xHi - xLo || 1;
    const yRng = yHi - yLo || 1;

  const sx = (v: number) => PAD.left + ((v - xLo) / xRng) * PLOT_W;
  const sy = (v: number) => PAD.top + PLOT_H - ((v - yLo) / yRng) * PLOT_H;

    const xTicks = useMemo(() => {
      const n = 5;
      const step = xRng / n;
      return Array.from({ length: n + 1 }, (_, i) => xLo + step * i);
    }, [xLo, xRng]);

    const yTicks = useMemo(() => {
      const n = 5;
      const step = yRng / n;
      return Array.from({ length: n + 1 }, (_, i) => yLo + step * i);
    }, [yLo, yRng]);

  return (
    <div className="regression-chart">
      {title && <h4>{title}</h4>}
      <svg viewBox={`0 0 ${CHART_W} ${CHART_H}`} className="chart-svg">
        <defs>
          <clipPath id="plot-clip">
            <rect x={PAD.left} y={PAD.top} width={PLOT_W} height={PLOT_H} />
          </clipPath>
        </defs>
        <g clipPath="url(#plot-clip)">
          {data.map((d, i) => (
            <circle key={i} cx={sx(d.x)} cy={sy(d.y)} r={2.5} fill={color} opacity={0.6} />
          ))}
          {diagonal && (
            <line
              x1={sx(Math.max(xLo, yLo))} y1={sy(Math.max(xLo, yLo))}
              x2={sx(Math.min(xHi, yHi))} y2={sy(Math.min(xHi, yHi))}
              stroke="#22c55e" strokeWidth={1.5} strokeDasharray="4 2"
            />
          )}
        </g>
        {xTicks.map((v, i) => (
          <g key={i}>
            <line x1={sx(v)} y1={PAD.top + PLOT_H} x2={sx(v)} y2={PAD.top + PLOT_H + 5} stroke="#475569" />
            <text x={sx(v)} y={PAD.top + PLOT_H + 18} textAnchor="middle" fill="#94a3b8" fontSize={10}>
              {v.toExponential(2)}
            </text>
          </g>
        ))}
        {yTicks.map((v, i) => (
          <g key={i}>
            <line x1={PAD.left - 5} y1={sy(v)} x2={PAD.left} y2={sy(v)} stroke="#475569" />
            <text x={PAD.left - 8} y={sy(v) + 3} textAnchor="end" fill="#94a3b8" fontSize={10}>
              {v.toExponential(2)}
            </text>
            <line x1={PAD.left} y1={sy(v)} x2={CHART_W - PAD.right} y2={sy(v)} stroke="#1e293b" strokeWidth={0.5} />
          </g>
        ))}
        <text x={PAD.left + PLOT_W / 2} y={CHART_H - 4} textAnchor="middle" fill="#94a3b8" fontSize={11}>
          {xLabel}
        </text>
        <text x={14} y={PAD.top + PLOT_H / 2} textAnchor="middle" fill="#94a3b8" fontSize={11} transform={`rotate(-90, 14, ${PAD.top + PLOT_H / 2})`}>
          {yLabel}
        </text>
      </svg>
    </div>
  );
}

function Histogram({ data, bins = 20, color = '#3b82f6', title, xLabel }: {
  data: number[];
  bins?: number;
  color?: string;
  title?: string;
  xLabel?: string;
}) {
  const bars = useMemo(() => {
    const min = Math.min(...data);
    const max = Math.max(...data);
    const rng = max - min || 1;
    const w = rng / bins;
    const counts = Array(bins).fill(0);
    for (const v of data) {
      const idx = Math.min(Math.floor((v - min) / w), bins - 1);
      counts[idx]++;
    }
    const maxCount = Math.max(...counts, 1);
    return Array.from({ length: bins }, (_, i) => ({
      x: min + w * i,
      w: w * 0.85,
      h: counts[i] / maxCount,
      count: counts[i],
    }));
  }, [data, bins]);

  const maxHeight = Math.max(...bars.map(b => b.h), 0.01);
  const barW = PLOT_W / bins * 0.85;
  const rng = bars.length > 0 ? bars[bars.length - 1].x + bars[bars.length - 1].w / 0.85 - bars[0].x : 1;
  const minX = bars[0]?.x ?? 0;

  const sx = (v: number) => PAD.left + ((v - minX) / rng) * PLOT_W * 0.9 + PLOT_W * 0.05;
  const sy = (v: number) => PAD.top + PLOT_H * (1 - v / maxHeight);

  const yTicks = [0, 0.25, 0.5, 0.75, 1].filter(v => v <= maxHeight);

  return (
    <div className="regression-chart">
      {title && <h4>{title}</h4>}
      <svg viewBox={`0 0 ${CHART_W} ${CHART_H}`} className="chart-svg">
        {bars.map((b, i) => (
          <rect
            key={i}
            x={sx(b.x) - barW / 2}
            y={sy(b.h)}
            width={barW}
            height={PLOT_H - sy(b.h) + PAD.top}
            fill={color}
            opacity={0.7}
            rx={1}
          />
        ))}
        {yTicks.map((v, i) => (
          <g key={i}>
            <line x1={PAD.left - 5} y1={sy(v)} x2={PAD.left} y2={sy(v)} stroke="#475569" />
            <text x={PAD.left - 8} y={sy(v) + 3} textAnchor="end" fill="#94a3b8" fontSize={10}>
              {Math.round(v * (bars[0]?.count ?? 0) / maxHeight)}
            </text>
          </g>
        ))}
        {xLabel && (
          <text x={PAD.left + PLOT_W / 2} y={CHART_H - 4} textAnchor="middle" fill="#94a3b8" fontSize={11}>
            {xLabel}
          </text>
        )}
        <text x={14} y={PAD.top + PLOT_H / 2} textAnchor="middle" fill="#94a3b8" fontSize={11} transform={`rotate(-90, 14, ${PAD.top + PLOT_H / 2})`}>
          Frecuencia
        </text>
      </svg>
    </div>
  );
}

function LineChart({ data, color = '#3b82f6', title, xLabel, yLabel }: {
  data: { x: number; y: number }[];
  color?: string;
  title?: string;
  xLabel?: string;
  yLabel?: string;
}) {
  const { xMin, xMax, yMin, yMax } = useMemo(() => {
    const xs = data.map(d => d.x);
    const ys = data.map(d => d.y);
    return {
      xMin: Math.min(...xs),
      xMax: Math.max(...xs),
      yMin: Math.min(...ys),
      yMax: Math.max(...ys),
    };
  }, [data]);

  const xRng = xMax - xMin || 1;
  const yRng = yMax - yMin || 1;
  const yPad = yRng * 0.1;

  const sx = (v: number) => PAD.left + ((v - xMin) / xRng) * PLOT_W;
  const sy = (v: number) => PAD.top + PLOT_H - ((v - (yMin - yPad)) / (yRng + 2 * yPad)) * PLOT_H;

  const points = data.map(d => `${sx(d.x)},${sy(d.y)}`).join(' ');

  return (
    <div className="regression-chart">
      {title && <h4>{title}</h4>}
      <svg viewBox={`0 0 ${CHART_W} ${CHART_H}`} className="chart-svg">
        <polyline points={points} fill="none" stroke={color} strokeWidth={2} />
        {data.map((d, i) => (
          <circle key={i} cx={sx(d.x)} cy={sy(d.y)} r={2} fill={color} />
        ))}
        {xLabel && (
          <text x={PAD.left + PLOT_W / 2} y={CHART_H - 4} textAnchor="middle" fill="#94a3b8" fontSize={11}>
            {xLabel}
          </text>
        )}
        {yLabel && (
          <text x={14} y={PAD.top + PLOT_H / 2} textAnchor="middle" fill="#94a3b8" fontSize={11} transform={`rotate(-90, 14, ${PAD.top + PLOT_H / 2})`}>
            {yLabel}
          </text>
        )}
      </svg>
    </div>
  );
}

function SignedBarChart({ data, title }: {
  data: Record<string, number>;
  title?: string;
}) {
  const sorted = useMemo(() => {
    return Object.entries(data)
      .sort(([, a], [, b]) => Math.abs(b) - Math.abs(a))
      .slice(0, 10);
  }, [data]);

  const maxAbs = useMemo(() => Math.max(...sorted.map(([, v]) => Math.abs(v)), 0.001), [sorted]);
  const barH = 22;
  const gap = 4;
  const totalH = sorted.length * (barH + gap) + 30;
  const svgH = Math.max(totalH, 100);
  const svgW = 450;
  const midX = svgW / 2;
  const barMaxW = svgW / 2 - 60;

  return (
    <div className="regression-chart">
      {title && <h4>{title}</h4>}
      <svg viewBox={`0 0 ${svgW} ${svgH}`} className="chart-svg">
        <line x1={midX} y1={10} x2={midX} y2={svgH - 20} stroke="#475569" strokeWidth={1} />
        {sorted.map(([name, val], i) => {
          const y = 25 + i * (barH + gap);
          const w = (Math.abs(val) / maxAbs) * barMaxW;
          const isPos = val >= 0;
          const x = isPos ? midX : midX - w;
          return (
            <g key={i}>
              <rect x={x} y={y} width={Math.max(w, 2)} height={barH} rx={3}
                fill={isPos ? '#22c55e' : '#ef4444'} opacity={0.85} />
              <text x={isPos ? midX + 6 : midX - 6} y={y + barH / 2 + 4}
                textAnchor={isPos ? 'start' : 'end'} fill="#f1f5f9" fontSize={11}>
                {name}
              </text>
              <text x={isPos ? midX + w + 6 : midX - w - 6} y={y + barH / 2 + 4}
                textAnchor={isPos ? 'start' : 'end'} fill="#94a3b8" fontSize={10}>
                {val.toFixed(4)}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

const modelLabels: Record<string, string> = {
  linear_regression: 'Regresión Lineal',
  ridge: 'Regresión Ridge',
  random_forest_regressor: 'Random Forest Regresor',
  gradient_boosting_regressor: 'Gradient Boosting Regresor',
};

export default function RegressionVisualizations({ result }: Props) {
  const hasData = result.predictions && result.actual_values && result.residuals;

  const scatterData = useMemo(() => {
    if (!hasData) return [];
    return result.predictions!.map((p, i) => ({ x: result.actual_values![i], y: p }));
  }, [result.predictions, result.actual_values, hasData]);

  const residualData = useMemo(() => {
    if (!hasData) return [];
    return result.predictions!.map((p, i) => ({ x: p, y: result.residuals![i] }));
  }, [result.predictions, result.residuals, hasData]);

  const modelName = modelLabels[result.model_id] || result.model_id;

  if (!hasData) {
    return (
      <div className="regression-visualizations">
        <p className="hint">Datos de predicción no disponibles para visualización.</p>
      </div>
    );
  }

  return (
    <div className="regression-visualizations">
      <h3>Visualizaciones - {modelName}</h3>
      <div className="algo-detail-hint">
        <p>
          {result.model_id === 'linear_regression' &&
            'La Regresión Lineal busca la línea recta que minimiza el ECM. Los coeficientes indican el cambio esperado en la variable objetivo por cada unidad de cambio en la característica.'}
          {result.model_id === 'ridge' &&
            'Ridge aplica regularización L2, contrayendo coeficientes para prevenir sobreajuste. El parámetro alpha controla la fuerza de la contracción.'}
          {result.model_id === 'random_forest_regressor' &&
            'Random Forest promedia múltiples árboles de decisión entrenados con bootstrap. Cada árbol captura relaciones no lineales en diferentes subconjuntos de datos.'}
          {result.model_id === 'gradient_boosting_regressor' &&
            'Gradient Boosting construye árboles secuencialmente, donde cada nuevo árbol corrige los residuos del anterior. La curva de pérdida muestra la convergencia.'}
        </p>
      </div>

      <div className="regression-charts-grid">
        <ScatterPlot
          data={scatterData}
          xLabel="Valores Reales"
          yLabel="Valores Predichos"
          diagonal
          color="#3b82f6"
          title="Valores Reales vs Predichos"
        />

        <ScatterPlot
          data={residualData}
          xLabel="Valores Predichos"
          yLabel="Residuos"
          color="#f59e0b"
          title="Residuos vs Valores Predichos"
        />
      </div>

      <div className="regression-charts-grid">
        <Histogram
          data={result.residuals!}
          bins={25}
          color="#8b5cf6"
          title="Distribución de Residuos"
          xLabel="Residuo"
        />

        {result.model_id === 'linear_regression' && result.feature_importance && (
          <SignedBarChart
            data={result.feature_importance}
            title="Coeficientes del Modelo Lineal"
          />
        )}

        {result.model_id === 'ridge' && result.feature_importance && (
          <SignedBarChart
            data={result.feature_importance}
            title="Coeficientes Ridge (Regularizados)"
          />
        )}

        {result.model_id === 'random_forest_regressor' && result.feature_importance && (
          <div className="regression-chart">
            <h4>Importancia de Características (Gini)</h4>
            <svg viewBox="0 0 450 250" className="chart-svg">
              {Object.entries(result.feature_importance)
                .sort(([, a], [, b]) => b - a)
                .slice(0, 8)
                .map(([name, val], i) => {
                  const maxVal = Math.max(...Object.values(result.feature_importance!), 0.001);
                  const y = 25 + i * 26;
                  const w = (val / maxVal) * 280;
                  return (
                    <g key={i}>
                      <text x={10} y={y + 14} fill="#f1f5f9" fontSize={11}>{name}</text>
                      <rect x={140} y={y + 2} width={Math.max(w, 2)} height={18} rx={3}
                        fill="#3b82f6" opacity={0.85} />
                      <text x={140 + w + 6} y={y + 15} fill="#94a3b8" fontSize={10}>
                        {val.toFixed(4)}
                      </text>
                    </g>
                  );
                })}
            </svg>
          </div>
        )}

        {result.model_id === 'gradient_boosting_regressor' && result.algorithm_details && (
          (() => {
            const details = result.algorithm_details as Record<string, unknown>;
            const trainLoss = details.train_loss as number[] | undefined;
            if (!trainLoss) return null;
            const lossData = trainLoss.map((v, i) => ({ x: i + 1, y: v }));
            return (
              <LineChart
                data={lossData}
                color="#ef4444"
                title="Pérdida de Entrenamiento por Iteración"
                xLabel="Iteración"
                yLabel="Pérdida (Deviance)"
              />
            );
          })()
        )}
      </div>

      {result.model_id === 'gradient_boosting_regressor' && result.feature_importance && (
        <div className="regression-charts-grid">
          <div className="regression-chart">
            <h4>Importancia de Características (Gradient Boosting)</h4>
            <svg viewBox="0 0 450 250" className="chart-svg">
              {Object.entries(result.feature_importance)
                .sort(([, a], [, b]) => b - a)
                .slice(0, 8)
                .map(([name, val], i) => {
                  const maxVal = Math.max(...Object.values(result.feature_importance!), 0.001);
                  const y = 25 + i * 26;
                  const w = (val / maxVal) * 280;
                  return (
                    <g key={i}>
                      <text x={10} y={y + 14} fill="#f1f5f9" fontSize={11}>{name}</text>
                      <rect x={140} y={y + 2} width={Math.max(w, 2)} height={18} rx={3}
                        fill="#ef4444" opacity={0.85} />
                      <text x={140 + w + 6} y={y + 15} fill="#94a3b8" fontSize={10}>
                        {val.toFixed(4)}
                      </text>
                    </g>
                  );
                })}
            </svg>
          </div>
        </div>
      )}
    </div>
  );
}
