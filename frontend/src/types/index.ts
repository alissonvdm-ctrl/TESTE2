// ── Uploads ──────────────────────────────────────────────────────────────────

export interface ColumnInfo {
  name: string
  dtype: string
  is_numeric: boolean
  null_count: number
  sample_values: unknown[]
}

export interface UploadResponse {
  upload_id: string
  filename: string
  row_count: number
  column_count: number
  columns: ColumnInfo[]
  preview: Record<string, unknown>[]
  uploaded_at: string
  file_size_bytes: number
}

export interface ValidationIssue {
  level: 'error' | 'warning'
  code: string
  message: string
}

export interface ValidationResponse {
  upload_id: string
  valid: boolean
  row_count: number
  usable_rows: number
  issues: ValidationIssue[]
}

// ── Experiments ───────────────────────────────────────────────────────────────

export interface ExperimentCreate {
  name: string
  description?: string
  upload_id: string
  sequence_column: string
  n_value?: number
  k_value?: number
  config?: Record<string, unknown>
}

export interface Experiment {
  id: string
  name: string
  description: string | null
  upload_id: string
  sequence_column: string
  n_value: number | null
  k_value: number | null
  config: Record<string, unknown> | null
  status: 'pending' | 'running' | 'completed' | 'failed'
  created_at: string
  updated_at: string | null
}

export interface ExperimentList {
  total: number
  page: number
  page_size: number
  items: Experiment[]
}

export interface ExperimentStatus {
  experiment_id: string
  status: string
  stage: string | null
  progress_pct: number
  message: string | null
  started_at: string | null
  completed_at: string | null
  error: string | null
}

// ── Analysis ──────────────────────────────────────────────────────────────────

export interface AnalysisRunResponse {
  experiment_id: string
  task_id: string | null
  status: string
  message: string
  queued_at: string
}

export interface Changepoint {
  index: number
  timestamp: string | null
  confidence: number
  regime_before: string | null
  regime_after: string | null
  description: string | null
}

export interface RegimeStats {
  regime_id: number
  start_index: number
  end_index: number
  length: number
  mean: number
  std: number
  trend: string
}

export interface ChangepointsResponse {
  experiment_id: string
  algorithm: string
  n_changepoints: number
  changepoints: Changepoint[]
  regimes: RegimeStats[]
  computed_at: string | null
}

export interface EnsembleWeights {
  model_name: string
  weight: number
  contribution_pct: number
}

export interface EnsembleResponse {
  experiment_id: string
  strategy: string
  metrics: Record<string, number> | null
  weights: EnsembleWeights[]
  improvement_over_best_single: number | null
  computed_at: string | null
}

// ── Reports ───────────────────────────────────────────────────────────────────

export interface ReportInfo {
  id: string
  experiment_id: string
  report_type: 'pdf' | 'html' | 'json' | 'excel'
  file_path: string
  created_at: string
}

// ── WebSocket messages ────────────────────────────────────────────────────────

export interface ProgressMessage {
  experiment_id: string
  stage: string
  status: string
  progress_pct: number
  message: string
  timestamp: string
}
