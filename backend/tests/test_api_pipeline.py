"""End-to-end API flow using a stubbed pipeline runner.

The real pipeline loads/synthesises the full dataset and trains LSTM,
which is too slow for a unit test. Here we swap the background-task
callable for a fake that writes a canned result into the registry, then
exercise every downstream endpoint.
"""

from fastapi.testclient import TestClient

from app.main import app
from app.services import job_registry, pipeline_service
from app.api import pipelines as pipelines_api


FAKE_RESULT = {
    "dataset_summary": {
        "n_samples": 3,
        "date_start": "2018-01-01 00:00:00",
        "date_end": "2018-01-03 00:00:00",
        "duration_days": 2,
    },
    "health_indicator": {
        "timestamps": ["2018-01-01", "2018-01-02", "2018-01-03"],
        "values": [0.1, 0.5, 0.9],
        "threshold": 0.9,
    },
    "true_rul": [2.0, 1.0, 0.0],
    "models": {
        "Exponential": {"predictions": [2.1, 1.1], "targets": [2.0, 1.0], "sequence_offset": 0},
        "LSTM": {"predictions": [1.0], "targets": [1.0], "sequence_offset": 1},
    },
    "metrics": {
        "Exponential": {"mae_days": 0.1, "rmse_days": 0.1},
        "LSTM": {"mae_days": 0.0, "rmse_days": 0.0},
    },
    "best_model": "LSTM",
}


def _fake_run(job_id, registry, **_kwargs):
    registry.update(job_id, status="succeeded", step="done", progress=1.0, result=FAKE_RESULT)


def test_pipeline_flow(monkeypatch):
    # Fresh registry per test avoids cross-test bleed.
    fresh = job_registry.JobRegistry()
    monkeypatch.setattr(job_registry, "_registry", fresh)
    monkeypatch.setattr(pipelines_api, "get_registry", lambda: fresh)
    monkeypatch.setattr(pipeline_service, "run_pipeline_job", _fake_run)
    monkeypatch.setattr(pipelines_api, "run_pipeline_job", _fake_run)

    client = TestClient(app)

    # Start a job.
    start = client.post("/api/v1/pipelines/run", json={"use_lstm": True})
    assert start.status_code == 202
    job_id = start.json()["job_id"]

    # Job status reflects the stubbed completion.
    status = client.get(f"/api/v1/jobs/{job_id}")
    assert status.status_code == 200
    assert status.json()["status"] == "succeeded"

    # Result-backed endpoints.
    summary = client.get(f"/api/v1/pipelines/{job_id}/dataset-summary")
    assert summary.status_code == 200
    assert summary.json()["n_samples"] == 3

    hi = client.get(f"/api/v1/pipelines/{job_id}/health-indicator")
    assert hi.status_code == 200
    assert hi.json()["values"] == [0.1, 0.5, 0.9]

    preds = client.get(f"/api/v1/pipelines/{job_id}/predictions")
    assert preds.status_code == 200
    assert set(preds.json()["models"]) == {"Exponential", "LSTM"}

    metrics = client.get(f"/api/v1/pipelines/{job_id}/metrics")
    assert metrics.status_code == 200
    body = metrics.json()
    assert body["best_model"] == "LSTM"
    assert body["metrics"]["LSTM"]["mae_days"] == 0.0


def test_missing_job_returns_404(monkeypatch):
    fresh = job_registry.JobRegistry()
    monkeypatch.setattr(job_registry, "_registry", fresh)
    monkeypatch.setattr(pipelines_api, "get_registry", lambda: fresh)

    client = TestClient(app)
    assert client.get("/api/v1/jobs/does-not-exist").status_code == 404
    assert client.get("/api/v1/pipelines/does-not-exist/metrics").status_code == 404


def test_pending_job_returns_409(monkeypatch):
    fresh = job_registry.JobRegistry()
    monkeypatch.setattr(job_registry, "_registry", fresh)
    monkeypatch.setattr(pipelines_api, "get_registry", lambda: fresh)

    job = fresh.create()  # status stays "pending"
    client = TestClient(app)
    resp = client.get(f"/api/v1/pipelines/{job.id}/metrics")
    assert resp.status_code == 409
