"""FastAPI-specific settings.

Reads environment variables lazily so tests and production can override
without re-importing modules.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List


def _split_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    api_v1_prefix: str = "/api/v1"
    project_name: str = "Wind Turbine RUL API"
    version: str = "0.2.0"
    cors_origins: List[str] = field(
        default_factory=lambda: _split_csv(
            os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
        )
    )


def get_settings() -> Settings:
    return Settings()
