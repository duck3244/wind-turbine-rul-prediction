"""Smoke test: /healthz responds and routes are registered."""

from fastapi.testclient import TestClient

from app.main import app


def test_healthz_ok():
    client = TestClient(app)
    resp = client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_openapi_lists_pipeline_routes():
    client = TestClient(app)
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    paths = resp.json()["paths"]
    assert "/api/v1/pipelines/run" in paths
    assert "/api/v1/jobs/{job_id}" in paths
