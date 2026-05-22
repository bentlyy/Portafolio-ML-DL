import { useMemo } from 'react';
import type { TrainingResult } from '../api/ml';

interface Props {
  result: TrainingResult;
}

const CHART_W = 500;
const CHART_H = 280;
const PAD = { top: 20, right: 20, bottom: 45, left: 60 };
const PLOT_W = CHART_W - PAD.left - PAD.right;
const PLOT_H = CHART_H - PAD.top - PAD.bottom;

function LossCurve({ data, color, title, yLabel }: {
  data: { x: number; y: number }[];
  color: string;
  title: string;
  yLabel: string;
}) {
  const { yMin, yMax } = useMemo(() => {
    const ys = data.map(d => d.y);
    return { yMin: Math.min(...ys), yMax: Math.max(...ys) };
  }, [data]);

  const xRng = data.length - 1 || 1;
  const yRng = yMax - yMin || 1;
  const yPad = yRng * 0.15;
  const yLo = yMin - yPad;
  const yHi = yMax + yPad * 2;

  const sx = (i: number) => PAD.left + (i / xRng) * PLOT_W;
  const sy = (v: number) => PAD.top + PLOT_H - ((v - yLo) / (yHi - yLo)) * PLOT_H;

  const points = data.map((d, i) => `${sx(i)},${sy(d.y)}`).join(' ');

  const yTicks = useMemo(() => {
    const n = 5;
    const step = (yHi - yLo) / n;
    return Array.from({ length: n + 1 }, (_, i) => yLo + step * i);
  }, [yLo, yHi]);

  return (
    <div className="nn-chart">
      <h4>{title}</h4>
      <svg viewBox={`0 0 ${CHART_W} ${CHART_H}`} className="chart-svg">
        <defs>
          <clipPath id="nn-loss-clip">
            <rect x={PAD.left} y={PAD.top} width={PLOT_W} height={PLOT_H} />
          </clipPath>
        </defs>
        <g clipPath="url(#nn-loss-clip)">
          <polyline points={points} fill="none" stroke={color} strokeWidth={2} opacity={0.8} />
          {data.filter((_, i) => i % Math.max(1, Math.floor(data.length / 20)) === 0).map((d, i) => (
            <circle key={i} cx={sx(d.x - 1)} cy={sy(d.y)} r={2.5} fill={color} opacity={0.6} />
          ))}
        </g>
        {yTicks.map((v, i) => (
          <g key={i}>
            <line x1={PAD.left - 5} y1={sy(v)} x2={PAD.left} y2={sy(v)} stroke="#475569" />
            <text x={PAD.left - 8} y={sy(v) + 3} textAnchor="end" fill="#94a3b8" fontSize={10}>
              {v < 0.01 ? v.toExponential(1) : v.toFixed(4)}
            </text>
            <line x1={PAD.left} y1={sy(v)} x2={CHART_W - PAD.right} y2={sy(v)} stroke="#1e293b" strokeWidth={0.5} />
          </g>
        ))}
        <text x={PAD.left + PLOT_W / 2} y={CHART_H - 4} textAnchor="middle" fill="#94a3b8" fontSize={11}>
          Iteración
        </text>
        <text x={14} y={PAD.top + PLOT_H / 2} textAnchor="middle" fill="#94a3b8" fontSize={11} transform={`rotate(-90, 14, ${PAD.top + PLOT_H / 2})`}>
          {yLabel}
        </text>
      </svg>
    </div>
  );
}

function ArchitectureDiagram({ architecture }: { architecture: number[] }) {
  const svgW = 500;
  const svgH = 120;
  const layerGap = svgW / (architecture.length + 1);

  return (
    <div className="nn-chart">
      <h4>Arquitectura de la Red</h4>
      <svg viewBox={`0 0 ${svgW} ${svgH}`} className="chart-svg">
        {architecture.map((neurons, i) => {
          const cx = layerGap * (i + 1);
          const dotR = Math.min(
            12,
            Math.max(4, (svgH * 0.6) / Math.max(neurons, 1) / 2)
          );
          const spacing = (svgH * 0.7) / Math.max(neurons, 1);
          const startY = (svgH - spacing * (neurons - 1)) / 2;

          return (
            <g key={i}>
              {neurons > 0 && Array.from({ length: neurons }).map((_, j) => {
                const ny = startY + j * spacing;
                return (
                  <g key={j}>
                    {i < architecture.length - 1 && architecture[i + 1] > 0 && (
                      Array.from({ length: Math.min(architecture[i + 1], 5) }).map((_, k) => {
                        const nextStart = (svgH - spacing * (architecture[i + 1] - 1)) / 2;
                        const nextY = nextStart + k * spacing;
                        return (
                          <line
                            key={`conn-${j}-${k}`}
                            x1={cx}
                            y1={ny}
                            x2={cx + layerGap}
                            y2={nextY}
                            stroke="#334155"
                            strokeWidth={0.5}
                            opacity={0.4}
                          />
                        );
                      })
                    )}
                    <circle cx={cx} cy={ny} r={dotR} fill="#3b82f6" opacity={0.8} />
                  </g>
                );
              })}
              <text x={cx} y={svgH - 8} textAnchor="middle" fill="#94a3b8" fontSize={10}>
                {neurons}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

function NetworkStats({ details }: { details: Record<string, unknown> }) {
  const stats = [
    { label: 'Iteraciones Reales', value: String(details.n_iter) },
    { label: 'Capas', value: String(details.n_layers) },
    { label: 'Parámetros Totales', value: Number(details.total_parameters).toLocaleString() },
    { label: 'Mejor Pérdida', value: typeof details.best_loss === 'number' ? details.best_loss.toFixed(6) : '-' },
    { label: 'Activación', value: String(details.activation) },
    { label: 'Optimizador', value: String(details.solver) },
    { label: 'Convergió', value: details.converged ? 'Sí' : 'No (límite alcanzado)' },
  ];

  return (
    <div className="nn-stats">
      <h4>Estadísticas de Entrenamiento</h4>
      <div className="algo-detail-grid">
        {stats.map(({ label, value }) => (
          <div key={label} className="algo-detail-card">
            <span className="algo-detail-label">{label}</span>
            <span className="algo-detail-value">{value}</span>
          </div>
        ))}
      </div>
      <p className="algo-detail-hint">
        {details.converged
          ? 'El modelo convergió antes de alcanzar el límite de iteraciones. La pérdida dejó de mejorar significativamente.'
          : 'El modelo alcanzó el máximo de iteraciones sin converger completamente. Aumenta max_iter o ajusta la tasa de aprendizaje.'}
      </p>
    </div>
  );
}

export default function NeuralNetworkVisualizations({ result }: Props) {
  const details = result.algorithm_details as Record<string, unknown> | undefined;

  const lossData = useMemo(() => {
    const curve = (details?.loss_curve as number[]) || [];
    return curve.map((v, i) => ({ x: i + 1, y: v }));
  }, [details?.loss_curve]);

  const valData = useMemo(() => {
    const scores = (details?.validation_scores as number[]) || [];
    return scores.map((v, i) => ({ x: i + 1, y: v }));
  }, [details?.validation_scores]);

  if (!details) return null;

  const architecture = (details.architecture as number[]) || [];

  return (
    <div className="nn-visualizations">
      <div className="regression-charts-grid">
        {lossData.length > 0 && (
          <LossCurve
            data={lossData}
            color="#3b82f6"
            title="Curva de Pérdida (Loss) por Iteración"
            yLabel="Pérdida"
          />
        )}
        {valData.length > 0 && (
          <LossCurve
            data={valData}
            color="#22c55e"
            title="Puntuación de Validación por Iteración"
            yLabel="Precisión"
          />
        )}
      </div>

      {architecture.length > 1 && <ArchitectureDiagram architecture={architecture} />}

      <NetworkStats details={details} />
    </div>
  );
}
