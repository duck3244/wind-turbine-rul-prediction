"""Pydantic DTOs for the pipeline API."""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

JobStatusLiteral = Literal["pending", "running", "succeeded", "failed"]


class RunPipelineRequest(BaseModel):
    use_lstm: bool = True
    lstm_sequence_length: int = Field(8, ge=2, le=64)
    seed: Optional[int] = None


class JobAccepted(BaseModel):
    job_id: str
    status: JobStatusLiteral = "pending"


class JobStatus(BaseModel):
    job_id: str
    status: JobStatusLiteral
    step: Optional[str] = None
    progress: float = Field(0.0, ge=0.0, le=1.0)
    logs: List[str] = []
    error: Optional[str] = None


class DatasetSummary(BaseModel):
    n_samples: int
    date_start: str
    date_end: str
    duration_days: int


class HealthIndicatorResponse(BaseModel):
    timestamps: List[str]
    values: List[float]
    threshold: Optional[float] = None


class ModelPrediction(BaseModel):
    predictions: List[float]
    targets: List[float]
    sequence_offset: int = 0


class PredictionsResponse(BaseModel):
    models: Dict[str, ModelPrediction]


class ModelMetrics(BaseModel):
    mae_days: float
    rmse_days: float


class MetricsResponse(BaseModel):
    metrics: Dict[str, ModelMetrics]
    best_model: Optional[str] = None
