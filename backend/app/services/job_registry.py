"""In-memory job registry for long-running pipeline tasks.

For the MVP a single-process dict with a lock is enough. When we move to
Celery/RQ, this module is the natural swap-in point.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Job:
    id: str
    status: str = "pending"  # pending | running | succeeded | failed
    step: Optional[str] = None
    progress: float = 0.0
    logs: List[str] = field(default_factory=list)
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None


class JobRegistry:
    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self) -> Job:
        job = Job(id=uuid.uuid4().hex)
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, **fields: Any) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            for key, value in fields.items():
                setattr(job, key, value)

    def append_log(self, job_id: str, message: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.logs.append(message)


_registry = JobRegistry()


def get_registry() -> JobRegistry:
    return _registry
