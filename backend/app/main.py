"""FastAPI entry point for the Wind Turbine RUL service."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import jobs, pipelines
from app.core.settings import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.project_name, version=settings.version)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/healthz", tags=["health"])
    def healthz() -> dict:
        return {"status": "ok", "version": settings.version}

    app.include_router(pipelines.router, prefix=settings.api_v1_prefix)
    app.include_router(jobs.router, prefix=settings.api_v1_prefix)

    return app


app = create_app()
