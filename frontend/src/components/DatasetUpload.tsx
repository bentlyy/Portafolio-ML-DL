import { useRef, useState } from 'react';

interface Props {
  onFileSelect: (file: File) => void;
  datasetInfo: any | null;
  targetColumn: string;
  onTargetChange: (col: string) => void;
  showTarget: boolean;
}

export default function DatasetUpload({ onFileSelect, datasetInfo, targetColumn, onTargetChange, showTarget }: Props) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [dragActive, setDragActive] = useState(false);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(false);
    const file = e.dataTransfer.files[0];
    if (file?.name.endsWith('.csv')) {
      onFileSelect(file);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) onFileSelect(file);
  };

  return (
    <div className="dataset-upload">
      <div
        className={`drop-zone ${dragActive ? 'active' : ''}`}
        onDragOver={e => { e.preventDefault(); setDragActive(true); }}
        onDragLeave={() => setDragActive(false)}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".csv"
          onChange={handleFileChange}
          hidden
        />
        <div className="drop-content">
          <span className="drop-icon">📁</span>
          <p>Drop CSV file here or click to browse</p>
        </div>
      </div>

      {datasetInfo && (
        <div className="dataset-info">
          <div className="info-header">
            <h4>{datasetInfo.filename}</h4>
            <span className="info-badge">{datasetInfo.rows} rows × {datasetInfo.columns} cols</span>
          </div>

          {showTarget && (
            <div className="target-selector">
              <label>Target Column:</label>
              <select value={targetColumn} onChange={e => onTargetChange(e.target.value)}>
                <option value="">-- Select target --</option>
                {datasetInfo.column_names.map((col: string) => (
                  <option key={col} value={col}>{col}</option>
                ))}
              </select>
            </div>
          )}

          <div className="dataset-preview">
            <h5>Preview (first 5 rows)</h5>
            <div className="table-container">
              <table>
                <thead>
                  <tr>
                    {datasetInfo.column_names.map((col: string) => (
                      <th key={col}>{col}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {datasetInfo.head.map((row: Record<string, unknown>, i: number) => (
                    <tr key={i}>
                      {datasetInfo.column_names.map((col: string) => (
                        <td key={col}>{String(row[col] ?? '')}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {Object.entries(datasetInfo.missing_values as Record<string, number>).some(([, v]) => v > 0) && (
            <div className="missing-info">
              <h5>Missing Values</h5>
              {Object.entries(datasetInfo.missing_values)
                .filter(([, v]) => v > 0)
                .map(([col, count]) => (
                  <span key={col} className="missing-badge">{col}: {count}</span>
                ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
