import { useMemo } from 'react';
import type { TrainingResult } from '../api/ml';

interface Props {
  result: TrainingResult;
}

interface ProjectionPoint {
  points: number[][];
  labels: number[];
}

interface ClusterDetails {
  type: string;
  projection: ProjectionPoint;
  pca_explained_variance?: number[] | null;
  cluster_sizes: Record<string, number>;
  cluster_centers_2d?: number[][] | null;
  cluster_centers?: number[][] | null;
  elbow?: { k: number; inertia: number }[];
  is_core?: number[];
  dbscan_eps?: number;
  dbscan_min_samples?: number;
  linkage_matrix?: number[][];
  linkage?: string;
}

const CHART_W = 500;
const CHART_H = 350;
const PAD = { top: 25, right: 25, bottom: 45, left: 55 };
const PLOT_W = CHART_W - PAD.left - PAD.right;
const PLOT_H = CHART_H - PAD.top - PAD.bottom;

const CLUSTER_COLORS = [
  '#3b82f6', '#22c55e', '#f59e0b', '#ef4444', '#a855f7',
  '#ec4899', '#14b8a6', '#f97316', '#6366f1', '#84cc16',
  '#06b6d4', '#d946ef', '#10b981', '#e11d48', '#0ea5e9',
  '#facc15', '#8b5cf6', '#34d399', '#fb923c', '#2dd4bf',
];

function getClusterColor(label: number): string {
  if (label === -1) return '#64748b';
  return CLUSTER_COLORS[label % CLUSTER_COLORS.length];
}

function ScatterPlotProjection({ projection, centers, title }: {
  projection: ProjectionPoint;
  centers?: number[][] | null;
  title?: string;
}) {
  const { points, labels } = projection;

  const { xMin, xMax, yMin, yMax } = useMemo(() => {
    const xs = points.map(p => p[0]);
    const ys = points.map(p => p[1]);
    const allX = centers ? [...xs, ...centers.map(c => c[0])] : xs;
    const allY = centers ? [...ys, ...centers.map(c => c[1])] : ys;
    return {
      xMin: Math.min(...allX),
      xMax: Math.max(...allX),
      yMin: Math.min(...allY),
      yMax: Math.max(...allY),
    };
  }, [points, centers]);

  const xRng = xMax - xMin || 1;
  const yRng = yMax - yMin || 1;
  const pad = 0.1;
  const xLo = xMin - xRng * pad;
  const xHi = xMax + xRng * pad;
  const yLo = yMin - yRng * pad;
  const yHi = yMax + yRng * pad;
  const xScale = xHi - xLo || 1;
  const yScale = yHi - yLo || 1;

  const sx = (v: number) => PAD.left + ((v - xLo) / xScale) * PLOT_W;
  const sy = (v: number) => PAD.top + PLOT_H - ((v - yLo) / yScale) * PLOT_H;

  const xTicks = useMemo(() => {
    const n = 5; const step = xScale / n;
    return Array.from({ length: n + 1 }, (_, i) => xLo + step * i);
  }, [xLo, xScale]);

  const yTicks = useMemo(() => {
    const n = 5; const step = yScale / n;
    return Array.from({ length: n + 1 }, (_, i) => yLo + step * i);
  }, [yLo, yScale]);

  const uniqueLabels = useMemo(() => [...new Set(labels)].sort((a, b) => a - b), [labels]);

  return (
    <div className="clustering-chart">
      {title && <h4>{title}</h4>}
      <svg viewBox={`0 0 ${CHART_W} ${CHART_H}`} className="chart-svg">
        <defs>
          <clipPath id="proj-clip">
            <rect x={PAD.left} y={PAD.top} width={PLOT_W} height={PLOT_H} />
          </clipPath>
        </defs>
        <g clipPath="url(#proj-clip)">
          {points.map((p, i) => (
            <circle key={i} cx={sx(p[0])} cy={sy(p[1])} r={3}
              fill={getClusterColor(labels[i])} opacity={0.7}
              stroke={labels[i] === -1 ? '#94a3b8' : 'none'} strokeWidth={labels[i] === -1 ? 1 : 0}
            />
          ))}
          {centers && centers.map((c, i) => (
            <g key={`c${i}`}>
              <circle cx={sx(c[0])} cy={sy(c[1])} r={8}
                fill={getClusterColor(i)} stroke="#f1f5f9" strokeWidth={2} opacity={0.95}
              />
              <text x={sx(c[0])} y={sy(c[1]) + 1} textAnchor="middle" fill="#0f172a"
                fontSize={9} fontWeight={700}>
                {i + 1}
              </text>
            </g>
          ))}
        </g>
        {xTicks.map((v, i) => (
          <g key={`xt${i}`}>
            <line x1={sx(v)} y1={PAD.top + PLOT_H} x2={sx(v)} y2={PAD.top + PLOT_H + 5} stroke="#475569" />
            <text x={sx(v)} y={PAD.top + PLOT_H + 18} textAnchor="middle" fill="#94a3b8" fontSize={10}>
              {v.toFixed(2)}
            </text>
          </g>
        ))}
        {yTicks.map((v, i) => (
          <g key={`yt${i}`}>
            <line x1={PAD.left - 5} y1={sy(v)} x2={PAD.left} y2={sy(v)} stroke="#475569" />
            <text x={PAD.left - 8} y={sy(v) + 3} textAnchor="end" fill="#94a3b8" fontSize={10}>
              {v.toFixed(2)}
            </text>
            <line x1={PAD.left} y1={sy(v)} x2={CHART_W - PAD.right} y2={sy(v)} stroke="#1e293b" strokeWidth={0.5} />
          </g>
        ))}
        <text x={PAD.left + PLOT_W / 2} y={CHART_H - 4} textAnchor="middle" fill="#94a3b8" fontSize={11}>
          Componente Principal 1
        </text>
        <text x={14} y={PAD.top + PLOT_H / 2} textAnchor="middle" fill="#94a3b8" fontSize={11}
          transform={`rotate(-90, 14, ${PAD.top + PLOT_H / 2})`}>
          Componente Principal 2
        </text>
      </svg>
      <div className="cluster-legend">
        {uniqueLabels.map(l => (
          <span key={l} className="legend-item">
            <span className="legend-dot" style={{ backgroundColor: getClusterColor(l) }} />
            {l === -1 ? 'Ruido' : `Grupo ${l + 1}`}
          </span>
        ))}
      </div>
    </div>
  );
}

function ClusterSizesChart({ sizes }: { sizes: Record<string, number> }) {
  const entries = useMemo(() => {
    return Object.entries(sizes)
      .map(([k, v]) => ({ label: k === '-1' ? 'Ruido' : `Grupo ${parseInt(k) + 1}`, value: v, key: k }))
      .sort((a, b) => parseInt(a.key) - parseInt(b.key));
  }, [sizes]);

  const maxVal = Math.max(...entries.map(e => e.value), 1);
  const barH = 24;
  const gap = 6;
  const totalH = entries.length * (barH + gap) + 30;
  const svgH = Math.max(totalH, 80);

  return (
    <div className="clustering-chart">
      <h4>Tamaño de Grupos</h4>
      <svg viewBox={`0 0 400 ${svgH}`} className="chart-svg">
        {entries.map((e, i) => {
          const y = 20 + i * (barH + gap);
          const w = (e.value / maxVal) * 250;
          return (
            <g key={i}>
              <text x={10} y={y + 16} fill="#f1f5f9" fontSize={11}>{e.label}</text>
              <rect x={120} y={y + 2} width={Math.max(w, 2)} height={barH} rx={4}
                fill={getClusterColor(parseInt(e.key))} opacity={0.85} />
              <text x={120 + w + 6} y={y + 18} fill="#94a3b8" fontSize={11}>{e.value}</text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

function ElbowChart({ data }: { data: { k: number; inertia: number }[] }) {
  const { xMin, xMax, yMin, yMax } = useMemo(() => {
    const xs = data.map(d => d.k);
    const ys = data.map(d => d.inertia);
    return {
      xMin: Math.min(...xs), xMax: Math.max(...xs),
      yMin: Math.min(...ys), yMax: Math.max(...ys),
    };
  }, [data]);

  const xRng = xMax - xMin || 1;
  const yRng = yMax - yMin || 1;
  const yPad = yRng * 0.1;
  const yLo = yMin - yPad;
  const yHi = yMax + yPad;

  const sx = (v: number) => PAD.left + ((v - xMin) / xRng) * PLOT_W;
  const sy = (v: number) => PAD.top + PLOT_H - ((v - yLo) / (yHi - yLo || 1)) * PLOT_H;

  const points = data.map(d => `${sx(d.k)},${sy(d.inertia)}`).join(' ');

  const xTicks = useMemo(() => {
    const n = Math.min(data.length - 1, 8);
    const step = (xMax - xMin) / n || 1;
    return Array.from({ length: n + 1 }, (_, i) => xMin + step * i);
  }, [xMin, xMax, data]);

  const yTicks = useMemo(() => {
    return [0, 0.25, 0.5, 0.75, 1].map(v => yLo + v * (yHi - yLo));
  }, [yLo, yHi]);

  return (
    <div className="clustering-chart">
      <h4>Método del Codo (Elbow) - Inercia vs K</h4>
      <svg viewBox={`0 0 ${CHART_W} ${CHART_H}`} className="chart-svg">
        <polyline points={points} fill="none" stroke="#3b82f6" strokeWidth={2} />
        {data.map((d, i) => (
          <circle key={i} cx={sx(d.k)} cy={sy(d.inertia)} r={4} fill="#3b82f6" stroke="#f1f5f9" strokeWidth={1} />
        ))}
        {xTicks.map((v, i) => (
          <g key={`xt${i}`}>
            <line x1={sx(v)} y1={PAD.top + PLOT_H} x2={sx(v)} y2={PAD.top + PLOT_H + 5} stroke="#475569" />
            <text x={sx(v)} y={PAD.top + PLOT_H + 18} textAnchor="middle" fill="#94a3b8" fontSize={10}>
              {Math.round(v)}
            </text>
          </g>
        ))}
        {yTicks.map((v, i) => (
          <g key={`yt${i}`}>
            <line x1={PAD.left - 5} y1={sy(v)} x2={PAD.left} y2={sy(v)} stroke="#475569" />
            <text x={PAD.left - 8} y={sy(v) + 3} textAnchor="end" fill="#94a3b8" fontSize={10}>
              {v.toExponential(1)}
            </text>
            <line x1={PAD.left} y1={sy(v)} x2={CHART_W - PAD.right} y2={sy(v)} stroke="#1e293b" strokeWidth={0.5} />
          </g>
        ))}
        <text x={PAD.left + PLOT_W / 2} y={CHART_H - 4} textAnchor="middle" fill="#94a3b8" fontSize={11}>
          Número de Grupos (K)
        </text>
        <text x={14} y={PAD.top + PLOT_H / 2} textAnchor="middle" fill="#94a3b8" fontSize={11}
          transform={`rotate(-90, 14, ${PAD.top + PLOT_H / 2})`}>
          Inercia
        </text>
      </svg>
      <p className="algo-detail-hint">
        El "codo" en la curva sugiere el número óptimo de grupos. Busca el punto donde la inercia deja de disminuir drásticamente.
      </p>
    </div>
  );
}

function DendrogramChart({ linkageMatrix }: { linkageMatrix: number[][] }) {
  const chartH = 300;
  const chartW = 500;
  const dgPad = { top: 20, right: 20, bottom: 40, left: 60 };
  const dgW = chartW - dgPad.left - dgPad.right;
  const dgH = chartH - dgPad.top - dgPad.bottom;

  const { treeData, maxDist } = useMemo(() => {
    if (!linkageMatrix || linkageMatrix.length === 0) {
      return { treeData: null, maxDist: 1 };
    }
    const Z = linkageMatrix;
    const n = Z.length + 1;
    const maxD = Math.max(...Z.map(row => row[2]), 1);

    interface Node {
      id: number;
      left: Node | null;
      right: Node | null;
      dist: number;
      leafIdx: number | null;
      size: number;
      x: number;
      y: number;
    }

    const nodes: (Node | null)[] = [];
    for (let i = 0; i < n; i++) {
      nodes.push({ id: i, left: null, right: null, dist: 0, leafIdx: i, size: 1, x: 0, y: 0 });
    }

    for (let i = 0; i < Z.length; i++) {
      const idxA = Math.round(Z[i][0]);
      const idxB = Math.round(Z[i][1]);
      const dist = Z[i][2];
      const a = nodes[idxA];
      const b = nodes[idxB];
      if (!a || !b) continue;
      const newNode: Node = {
        id: n + i,
        left: a,
        right: b,
        dist: dist,
        leafIdx: null,
        size: a.size + b.size,
        x: 0, y: 0,
      };
      nodes.push(newNode);
    }

    const root = nodes[nodes.length - 1];
    if (!root) return { treeData: null, maxDist: maxD };

    function assignX(node: Node, offset: number): number {
      if (!node.left && !node.right) {
        node.x = offset;
        return offset + 1;
      }
      let off = offset;
      if (node.left) off = assignX(node.left, off);
      if (node.right) off = assignX(node.right, off);
      const lx = node.left ? node.left.x : off;
      const rx = node.right ? node.right.x : off;
      node.x = (lx + rx) / 2;
      return off;
    }
    assignX(root, 0);

    function assignY(node: Node) {
      if (!node.left && !node.right) {
        node.y = 0;
        return;
      }
      if (node.left) assignY(node.left);
      if (node.right) assignY(node.right);
      node.y = node.dist;
    }
    assignY(root);

    const dataLines: { x1: number; y1: number; x2: number; y2: number }[] = [];

    function collectLines(node: Node) {
      const nx = dgPad.left + (node.x / (n - 1 || 1)) * dgW;
      const ny = dgPad.top + dgH - (node.y / maxD) * dgH;
      if (node.left) {
        collectLines(node.left);
        const lx = dgPad.left + (node.left.x / (n - 1 || 1)) * dgW;
        const ly = dgPad.top + dgH - (node.left.y / maxD) * dgH;
        dataLines.push({ x1: lx, y1: ly, x2: lx, y2: ny });
        dataLines.push({ x1: lx, y1: ny, x2: nx, y2: ny });
      }
      if (node.right) {
        collectLines(node.right);
        const rx = dgPad.left + (node.right.x / (n - 1 || 1)) * dgW;
        const ry = dgPad.top + dgH - (node.right.y / maxD) * dgH;
        dataLines.push({ x1: rx, y1: ry, x2: rx, y2: ny });
        dataLines.push({ x1: rx, y1: ny, x2: nx, y2: ny });
      }
    }
    collectLines(root);

    const leafPositions: { x: number; idx: number }[] = [];
    function collectLeaves(node: Node) {
      if (!node.left && !node.right && node.leafIdx !== null) {
        const lx = dgPad.left + (node.x / (n - 1 || 1)) * dgW;
        leafPositions.push({ x: lx, idx: node.leafIdx });
      }
      if (node.left) collectLeaves(node.left);
      if (node.right) collectLeaves(node.right);
    }
    collectLeaves(root);
    leafPositions.sort((a, b) => a.x - b.x);

    return { treeData: dataLines, maxDist: maxD };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [linkageMatrix]);

  if (!treeData || !linkageMatrix || linkageMatrix.length === 0) {
    return (
      <div className="clustering-chart">
        <h4>Dendrograma</h4>
        <p className="hint">Dendrograma no disponible para este conjunto de datos.</p>
      </div>
    );
  }

  const yTicks = [0, 0.25, 0.5, 0.75, 1].map(v => v * maxDist);
  const totalSvgH = chartH;

  return (
    <div className="clustering-chart dendrogram-chart">
      <h4>Dendrograma (Jerarquía de Grupos)</h4>
      <svg viewBox={`0 0 ${chartW} ${totalSvgH}`} className="chart-svg">
        {treeData.map((line, i) => (
          <line key={i} x1={line.x1} y1={line.y1} x2={line.x2} y2={line.y2}
            stroke="#3b82f6" strokeWidth={1.5} opacity={0.7} />
        ))}
        {yTicks.map((v, i) => {
          const y = dgPad.top + dgH - (v / maxDist) * dgH;
          return (
            <g key={`yt${i}`}>
              <line x1={dgPad.left - 5} y1={y} x2={dgPad.left} y2={y} stroke="#475569" />
              <text x={dgPad.left - 8} y={y + 3} textAnchor="end" fill="#94a3b8" fontSize={9}>
                {v.toFixed(2)}
              </text>
              <line x1={dgPad.left} y1={y} x2={chartW - dgPad.right} y2={y} stroke="#1e293b" strokeWidth={0.5} />
            </g>
          );
        })}
        <text x={14} y={dgPad.top + dgH / 2} textAnchor="middle" fill="#94a3b8" fontSize={11}
          transform={`rotate(-90, 14, ${dgPad.top + dgH / 2})`}>
          Distancia
        </text>
      </svg>
      <p className="algo-detail-hint">
        El dendrograma muestra la jerarquía de agrupaciones. Cortando a diferentes alturas se obtienen diferentes números de grupos.
      </p>
    </div>
  );
}

function DBSCANInfo({ details }: { details: ClusterDetails }) {
  return (
    <div className="clustering-chart">
      <h4>Información DBSCAN</h4>
      <div className="algo-detail-grid">
        <div className="algo-detail-card">
          <span className="algo-detail-label">Épsilon (eps)</span>
          <span className="algo-detail-value">{details.dbscan_eps}</span>
        </div>
        <div className="algo-detail-card">
          <span className="algo-detail-label">Min. Muestras</span>
          <span className="algo-detail-value">{details.dbscan_min_samples}</span>
        </div>
        <div className="algo-detail-card">
          <span className="algo-detail-label">Puntos Centrales</span>
          <span className="algo-detail-value">
            {details.is_core ? details.is_core.filter(v => v === 1).length : '-'}
          </span>
        </div>
        <div className="algo-detail-card">
          <span className="algo-detail-label">Puntos Frontera/Ruido</span>
          <span className="algo-detail-value">
            {details.is_core ? details.is_core.filter(v => v === 0).length : '-'}
          </span>
        </div>
      </div>
      <p className="algo-detail-hint">
        DBSCAN agrupa puntos basándose en la densidad. Los círculos rellenos son puntos centrales,
        los vacíos son puntos frontera o ruido (gris).
      </p>
    </div>
  );
}

function HierarchicalInfo({ details }: { details: ClusterDetails }) {
  return (
    <div className="clustering-chart">
      <h4>Información Jerárquica</h4>
      <div className="algo-detail-grid">
        <div className="algo-detail-card">
          <span className="algo-detail-label">Método de Enlace</span>
          <span className="algo-detail-value">{details.linkage || 'ward'}</span>
        </div>
      </div>
      <p className="algo-detail-hint">
        El agrupamiento jerárquico ascendente combina puntos/grupos vecinos iterativamente.
        El método de enlace determina cómo se mide la distancia entre grupos.
      </p>
    </div>
  );
}

const modelLabels: Record<string, string> = {
  kmeans: 'K-Means',
  dbscan: 'DBSCAN',
  hierarchical: 'Agrupamiento Jerárquico',
};

const modelDescriptions: Record<string, string> = {
  kmeans: 'K-Means particiona los datos en K grupos minimizando la varianza dentro de cada grupo. '
    + 'Los círculos numerados son los centroides (centros de cada grupo). '
    + 'La proyección PCA permite visualizar la separación en 2 dimensiones.',
  dbscan: 'DBSCAN agrupa puntos basándose en densidad, identificando automáticamente el número de grupos '
    + 'y detectando valores atípicos (ruido, en gris). No necesita especificar K.',
  hierarchical: 'El agrupamiento jerárquico construye un árbol de agrupaciones sin asumir formas específicas. '
    + 'El dendrograma muestra la jerarquía completa de uniones entre grupos.',
};

export default function ClusteringVisualizations({ result }: Props) {
  const details = result.algorithm_details as ClusterDetails | undefined;

  if (!details || !details.projection) {
    return (
      <div className="clustering-visualizations">
        <p className="hint">Datos de visualización no disponibles.</p>
      </div>
    );
  }

  const modelName = modelLabels[details.type] || details.type;

  return (
    <div className="clustering-visualizations">
      <h3>Visualizaciones de Agrupamiento - {modelName}</h3>
      <div className="algo-detail-hint">
        <p>{modelDescriptions[details.type] || ''}</p>
      </div>

      <div className="clustering-charts-grid">
        <ScatterPlotProjection
          projection={details.projection}
          centers={details.cluster_centers_2d}
          title="Proyección PCA - Grupos"
        />

        {details.type === 'kmeans' && details.elbow && details.elbow.length > 0 && (
          <ElbowChart data={details.elbow} />
        )}

        {details.type === 'kmeans' && !details.elbow && (
          <ClusterSizesChart sizes={details.cluster_sizes} />
        )}
      </div>

      {details.type === 'kmeans' && details.elbow && details.elbow.length > 0 && (
        <div className="clustering-charts-grid">
          <ClusterSizesChart sizes={details.cluster_sizes} />
        </div>
      )}

      <div className="clustering-charts-grid">
        {details.type === 'dbscan' && (
          <>
            <DBSCANInfo details={details} />
            <ClusterSizesChart sizes={details.cluster_sizes} />
          </>
        )}

        {details.type === 'hierarchical' && (
          <>
            <HierarchicalInfo details={details} />
            <ClusterSizesChart sizes={details.cluster_sizes} />
          </>
        )}
      </div>

      {details.type === 'hierarchical' && details.linkage_matrix && (
        <div className="clustering-charts-grid dendrogram-grid">
          <DendrogramChart linkageMatrix={details.linkage_matrix} />
        </div>
      )}
    </div>
  );
}
