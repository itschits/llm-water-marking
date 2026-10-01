// API client — proxied through Vite to http://localhost:8000
import axios from 'axios'

const api = axios.create({ baseURL: '' }) // Vite proxy forwards /api → :8000

// ── Types ─────────────────────────────────────────────────────────────

export interface AlgorithmResult {
  z_score: number
  detected: boolean
  threshold: number
}

export interface DetectResponse {
  token_count: number
  sta1: AlgorithmResult
  synthid: AlgorithmResult
}

export interface GeneratedTextResult {
  text: string
  token_count: number
  z_score: number
  p_value: number
  detected: boolean
  signal_label: string
  signal_value: number
  signal_detail: string
}

export interface RunExperimentResponse {
  model: string
  model_id: string
  algorithm: string
  prompt: string
  threshold: number
  elapsed_seconds: number
  plain: GeneratedTextResult
  watermarked: GeneratedTextResult
}

export interface RunExperimentRequest {
  model: 'gpt2' | 'qwen-2.5-1.5b' | 'qwen-2.5-3b'
  algorithm: 'sta1' | 'synthid'
  prompt: string
  max_new_tokens: number
  temperature: number
}

export interface Metrics {
  tpr: number
  fpr: number
  accuracy: number
  f1: number
  auc_roc: number
  n_watermarked: number
  n_unwatermarked: number
  threshold: number
  tp: number
  fp: number
  tn: number
  fn: number
}

export interface AlgoData {
  watermarked_scores: number[]
  unwatermarked_scores: number[]
  metrics: Metrics
}

export type ResultsData = Record<string, Record<string, AlgoData>>

// ── Calls ─────────────────────────────────────────────────────────────

export const getHealth = (): Promise<{ status: string; version: string }> =>
  api.get('/api/health').then(r => r.data)

export const detectWatermark = (text: string): Promise<DetectResponse> =>
  api.post('/api/detect', { text }).then(r => r.data)

export const runExperiment = (body: RunExperimentRequest): Promise<RunExperimentResponse> =>
  api.post('/api/experiments/run', body).then(r => r.data)

export const getResults = (): Promise<{
  available: boolean
  data: ResultsData | null
  filename: string | null
}> => api.get('/api/results').then(r => r.data)
