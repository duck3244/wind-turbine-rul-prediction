"""Wraps SimpleRULPipeline for execution as a FastAPI BackgroundTask.

The service keeps pipeline artifacts (health indicator, predictions,
metrics, dataset summary) in the job registry so REST handlers can serve
them without re-running the pipeline.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from app.services.job_registry import JobRegistry


def _to_list(array: Any) -> list:
    if array is None:
        return []
    if isinstance(array, np.ndarray):
        return array.astype(float).tolist()
    return list(array)


def run_pipeline_job(
    job_id: str,
    registry: JobRegistry,
    use_lstm: bool,
    lstm_sequence_length: int,
    seed: Optional[int],
) -> None:
    """Entry point invoked by FastAPI's BackgroundTasks.

    Runs each pipeline step while publishing progress into the registry
    so clients polling GET /jobs/{id} see live updates.
    """

    from config import RANDOM_SEED, ensure_output_dirs
    from utils import set_global_seed
    from main import SimpleRULPipeline, check_imports

    effective_seed = seed if seed is not None else RANDOM_SEED
    set_global_seed(effective_seed)
    ensure_output_dirs()

    ok, components = check_imports()
    if not ok or components is None:
        registry.update(job_id, status="failed", error="module import failed")
        return

    if not components["tensorflow_available"]:
        use_lstm = False

    pipeline = SimpleRULPipeline(
        components=components,
        use_lstm=use_lstm,
        use_visualization=False,
        lstm_sequence_length=lstm_sequence_length,
    )

    steps = [
        ("load_data", pipeline.step1_load_data),
        ("extract_features", pipeline.step2_extract_features),
        ("prepare_targets", pipeline.step3_prepare_targets),
        ("exponential_model", pipeline.step4_train_exponential_model),
        ("lstm_model", pipeline.step5_train_lstm_model),
    ]

    registry.update(job_id, status="running")

    for idx, (name, fn) in enumerate(steps, start=1):
        registry.update(job_id, step=name, progress=(idx - 1) / len(steps))
        registry.append_log(job_id, f"start: {name}")
        try:
            success = fn()
        except Exception as exc:
            registry.update(job_id, status="failed", error=f"{name}: {exc}")
            registry.append_log(job_id, f"error in {name}: {exc}")
            return
        if not success:
            registry.update(job_id, status="failed", error=f"step {name} returned False")
            return
        registry.append_log(job_id, f"done:  {name}")

    result = _collect_result(pipeline)
    registry.update(job_id, status="succeeded", step="done", progress=1.0, result=result)


def _collect_result(pipeline) -> Dict[str, Any]:
    """Snapshot the pipeline into a JSON-serializable dict."""
    dataset_summary = None
    if pipeline.data is not None:
        date_col = pipeline.data["Date"]
        dataset_summary = {
            "n_samples": int(len(pipeline.data)),
            "date_start": str(date_col.min()),
            "date_end": str(date_col.max()),
            "duration_days": int((date_col.max() - date_col.min()).days),
        }

    hi_values = _to_list(pipeline.health_indicator)
    timestamps: list = []
    if pipeline.data is not None and "Date" in pipeline.data.columns:
        timestamps = [str(ts) for ts in pipeline.data["Date"].tolist()]

    true_rul = _to_list(pipeline.true_rul)

    models: Dict[str, Dict[str, Any]] = {}
    metrics: Dict[str, Dict[str, float]] = {}

    if "Exponential" in pipeline.results:
        res = pipeline.results["Exponential"]
        preds = _to_list(res["predictions"])
        targets = true_rul[: len(preds)]
        models["Exponential"] = {
            "predictions": preds,
            "targets": targets,
            "sequence_offset": 0,
        }
        metrics["Exponential"] = {
            "mae_days": float(res["mae"]),
            "rmse_days": float(res["rmse"]),
        }

    if "LSTM" in pipeline.results:
        res = pipeline.results["LSTM"]
        preds = _to_list(res["predictions"])
        targets = _to_list(res.get("targets", []))
        models["LSTM"] = {
            "predictions": preds,
            "targets": targets,
            "sequence_offset": int(res.get("sequence_offset", 0)),
        }
        metrics["LSTM"] = {
            "mae_days": float(res["mae"]),
            "rmse_days": float(res["rmse"]),
        }

    best_model: Optional[str] = None
    if metrics:
        best_model = min(metrics.keys(), key=lambda k: metrics[k]["mae_days"])

    return {
        "dataset_summary": dataset_summary,
        "health_indicator": {
            "timestamps": timestamps,
            "values": hi_values,
            "threshold": hi_values[-1] if hi_values else None,
        },
        "true_rul": true_rul,
        "models": models,
        "metrics": metrics,
        "best_model": best_model,
    }
