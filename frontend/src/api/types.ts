export type JobStatusLiteral = "pending" | "running" | "succeeded" | "failed";

export interface RunPipelineRequest {
  use_lstm?: boolean;
  lstm_sequence_length?: number;
  seed?: number | null;
}

export interface JobAccepted {
  job_id: string;
  status: JobStatusLiteral;
}

export interface JobStatus {
  job_id: string;
  status: JobStatusLiteral;
  step: string | null;
  progress: number;
  logs: string[];
  error: string | null;
}

export interface DatasetSummary {
  n_samples: number;
  date_start: string;
  date_end: string;
  duration_days: number;
}

export interface HealthIndicator {
  timestamps: string[];
  values: number[];
  threshold: number | null;
}

export interface ModelPrediction {
  predictions: number[];
  targets: number[];
  sequence_offset: number;
}

export interface PredictionsResponse {
  models: Record<string, ModelPrediction>;
}

export interface ModelMetrics {
  mae_days: number;
  rmse_days: number;
}

export interface MetricsResponse {
  metrics: Record<string, ModelMetrics>;
  best_model: string | null;
}
