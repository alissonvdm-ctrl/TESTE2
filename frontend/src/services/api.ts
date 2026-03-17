import axios from 'axios'
import type {
  UploadResponse,
  ValidationResponse,
  ExperimentCreate,
  Experiment,
  ExperimentList,
  ExperimentStatus,
  AnalysisRunResponse,
  ChangepointsResponse,
  EnsembleResponse,
  ReportInfo,
} from '../types'

const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? ''

const http = axios.create({
  baseURL: BASE_URL,
  timeout: 30_000,
})

// ── Uploads ───────────────────────────────────────────────────────────────────

export const uploadExcel = (file: File) => {
  const form = new FormData()
  form.append('file', file)
  return http.post<UploadResponse>('/uploads/excel', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
}

export const validateUpload = (
  uploadId: string,
  sequenceColumn: string,
  nValue: number,
  kValue: number,
) =>
  http.post<ValidationResponse>(`/uploads/${uploadId}/validate`, {
    sequence_column: sequenceColumn,
    n_value: nValue,
    k_value: kValue,
  })

export const getUploadColumns = (uploadId: string) =>
  http.get(`/uploads/${uploadId}/columns`)

export const getUploadPreview = (uploadId: string, rows = 10) =>
  http.get(`/uploads/${uploadId}/preview`, { params: { rows } })

// ── Experiments ───────────────────────────────────────────────────────────────

export const createExperiment = (payload: ExperimentCreate) =>
  http.post<Experiment>('/experiments', payload)

export const listExperiments = (page = 1, pageSize = 20, status?: string) =>
  http.get<ExperimentList>('/experiments', { params: { page, page_size: pageSize, status } })

export const getExperiment = (id: string) =>
  http.get<Experiment>(`/experiments/${id}`)

export const deleteExperiment = (id: string) =>
  http.delete(`/experiments/${id}`)

export const getExperimentStatus = (id: string) =>
  http.get<ExperimentStatus>(`/experiments/${id}/status`)

export const startExperiment = (id: string) =>
  http.post<ExperimentStatus>(`/experiments/${id}/start`)

// ── Analysis ──────────────────────────────────────────────────────────────────

export const runAnalysis = (experimentId: string, forceRerun = false) =>
  http.post<AnalysisRunResponse>(`/analysis/${experimentId}/run`, {
    force_rerun: forceRerun,
  })

export const getChangepoints = (experimentId: string, algorithm = 'PELT') =>
  http.get<ChangepointsResponse>(`/analysis/${experimentId}/changepoints`, {
    params: { algorithm },
  })

export const getEnsemble = (experimentId: string) =>
  http.get<EnsembleResponse>(`/analysis/${experimentId}/ensemble`)

export const getFeatures = (experimentId: string, topN = 20) =>
  http.get(`/analysis/${experimentId}/features`, { params: { top_n: topN } })

// ── Reports ───────────────────────────────────────────────────────────────────

export const listReports = (experimentId: string) =>
  http.get<ReportInfo[]>(`/reports/${experimentId}`)

export const generateReport = (experimentId: string, format: string) =>
  http.post(`/reports/${experimentId}/generate`, { report_format: format })

export const downloadReport = (reportId: string) =>
  http.get(`/reports/download/${reportId}`, { responseType: 'blob' })
