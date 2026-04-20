"""Pipeline run + result endpoints."""

from __future__ import annotations

from typing import Dict

from fastapi import APIRouter, BackgroundTasks, HTTPException

from app.schemas.pipeline import (
    DatasetSummary,
    HealthIndicatorResponse,
    JobAccepted,
    MetricsResponse,
    ModelMetrics,
    ModelPrediction,
    PredictionsResponse,
    RunPipelineRequest,
)
from app.services.job_registry import Job, get_registry
from app.services.pipeline_service import run_pipeline_job

router = APIRouter(prefix="/pipelines", tags=["pipelines"])


def _require_succeeded(job_id: str) -> Job:
    job = get_registry().get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if job.status != "succeeded" or job.result is None:
        raise HTTPException(
            status_code=409,
            detail=f"job not ready (status={job.status})",
        )
    return job


@router.post("/run", response_model=JobAccepted, status_code=202)
def run_pipeline(request: RunPipelineRequest, background_tasks: BackgroundTasks) -> JobAccepted:
    registry = get_registry()
    job = registry.create()
    background_tasks.add_task(
        run_pipeline_job,
        job_id=job.id,
        registry=registry,
        use_lstm=request.use_lstm,
        lstm_sequence_length=request.lstm_sequence_length,
        seed=request.seed,
    )
    return JobAccepted(job_id=job.id, status="pending")


@router.get("/{job_id}/dataset-summary", response_model=DatasetSummary)
def dataset_summary(job_id: str) -> DatasetSummary:
    job = _require_succeeded(job_id)
    summary = job.result.get("dataset_summary") if job.result else None
    if summary is None:
        raise HTTPException(status_code=404, detail="dataset summary unavailable")
    return DatasetSummary(**summary)


@router.get("/{job_id}/health-indicator", response_model=HealthIndicatorResponse)
def health_indicator(job_id: str) -> HealthIndicatorResponse:
    job = _require_succeeded(job_id)
    hi = job.result["health_indicator"]  # type: ignore[index]
    return HealthIndicatorResponse(**hi)


@router.get("/{job_id}/predictions", response_model=PredictionsResponse)
def predictions(job_id: str) -> PredictionsResponse:
    job = _require_succeeded(job_id)
    raw_models: Dict[str, dict] = job.result["models"]  # type: ignore[index]
    models = {name: ModelPrediction(**payload) for name, payload in raw_models.items()}
    return PredictionsResponse(models=models)


@router.get("/{job_id}/metrics", response_model=MetricsResponse)
def metrics(job_id: str) -> MetricsResponse:
    job = _require_succeeded(job_id)
    raw_metrics: Dict[str, dict] = job.result["metrics"]  # type: ignore[index]
    best = job.result.get("best_model")  # type: ignore[union-attr]
    metrics_out = {name: ModelMetrics(**payload) for name, payload in raw_metrics.items()}
    return MetricsResponse(metrics=metrics_out, best_model=best)
