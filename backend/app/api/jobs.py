"""Job status endpoint."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.schemas.pipeline import JobStatus
from app.services.job_registry import get_registry

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("/{job_id}", response_model=JobStatus)
def get_job(job_id: str) -> JobStatus:
    job = get_registry().get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return JobStatus(
        job_id=job.id,
        status=job.status,  # type: ignore[arg-type]
        step=job.step,
        progress=job.progress,
        logs=list(job.logs),
        error=job.error,
    )
