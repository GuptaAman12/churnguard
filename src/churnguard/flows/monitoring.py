"""Prefect flow: scheduled drift monitoring with optional auto-retraining."""

from __future__ import annotations

from typing import Any

from prefect import flow, task
from prefect.logging import get_run_logger

from churnguard.config import get_settings
from churnguard.data import read_dataset
from churnguard.monitoring.drift import load_prediction_features, run_drift
from churnguard.monitoring.simulate import simulate_production_drift


@task(name="drift-report", retries=2, retry_delay_seconds=30)
def drift_task(use_prediction_log: bool) -> dict[str, Any]:
    settings = get_settings()
    reference = read_dataset(settings.reference_path)
    if use_prediction_log and settings.prediction_log_path.exists():
        current = load_prediction_features(settings.prediction_log_path)
    else:
        if not settings.current_data_path.exists():
            simulate_production_drift(settings.test_path, settings.current_data_path)
        current = read_dataset(settings.current_data_path)
    summary = run_drift(reference, current, settings)
    return {
        "drift_share": summary.drift_share,
        "drifted_features": summary.drifted_features,
        "total_features": summary.total_features,
        "dataset_drift": summary.dataset_drift,
        "generated_at": summary.generated_at,
    }


@flow(
    name="churnguard-monitoring-pipeline",
    description="Detect feature drift and emit Prometheus + artifact signals.",
    log_prints=True,
)
def monitoring_pipeline(use_prediction_log: bool = False) -> dict[str, Any]:
    result = drift_task(use_prediction_log=use_prediction_log)
    logger: Any = get_run_logger()
    if result["dataset_drift"]:
        logger.warning("dataset_drift_detected", **result)
    else:
        logger.info("no_dataset_drift", drift_share=result["drift_share"])
    return result
