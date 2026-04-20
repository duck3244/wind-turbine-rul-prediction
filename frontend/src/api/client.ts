import axios from "axios";
import type {
  DatasetSummary,
  HealthIndicator,
  JobAccepted,
  JobStatus,
  MetricsResponse,
  PredictionsResponse,
  RunPipelineRequest,
} from "./types";

const client = axios.create({ baseURL: "/api/v1", timeout: 15000 });

export async function runPipeline(payload: RunPipelineRequest): Promise<JobAccepted> {
  const { data } = await client.post<JobAccepted>("/pipelines/run", payload);
  return data;
}

export async function getJobStatus(jobId: string): Promise<JobStatus> {
  const { data } = await client.get<JobStatus>(`/jobs/${jobId}`);
  return data;
}

export async function getDatasetSummary(jobId: string): Promise<DatasetSummary> {
  const { data } = await client.get<DatasetSummary>(
    `/pipelines/${jobId}/dataset-summary`,
  );
  return data;
}

export async function getHealthIndicator(jobId: string): Promise<HealthIndicator> {
  const { data } = await client.get<HealthIndicator>(
    `/pipelines/${jobId}/health-indicator`,
  );
  return data;
}

export async function getPredictions(jobId: string): Promise<PredictionsResponse> {
  const { data } = await client.get<PredictionsResponse>(
    `/pipelines/${jobId}/predictions`,
  );
  return data;
}

export async function getMetrics(jobId: string): Promise<MetricsResponse> {
  const { data } = await client.get<MetricsResponse>(`/pipelines/${jobId}/metrics`);
  return data;
}
