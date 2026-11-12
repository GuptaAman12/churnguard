"""Prefect flow: ingest -> validate/split -> train -> score -> drift."""

from __future__ import annotations

from typing import Any

from prefect import flow, task
from prefect.logging import get_run_logger

from churnguard.api.model_loader import ModelService
from churnguard.batch import score_batch
from churnguard.config import get_settings
from churnguard.data import prepare_datasets, read_dataset
from churnguard.ingest import ensure_dataset
from churnguard.modeling.train import train_and_log
from churnguard.monitoring.drift import run_drift
from churnguard.monitoring.simulate import simulate_production_drift


@task(name="ingest-data", retries=2, retry_delay_seconds=15)
def ingest_task(force: bool = False) -> dict[str, Any]:
    settings = get_settings()
    path, source = ensure_dataset(settings, force=force)
    return {"path": str(path), "source": source}


@task(name="validate-and-split", retries=1, retry_delay_seconds=10)
def split_task() -> dict[str, Any]:
    settings = get_settings()
    return prepare_datasets(
        raw_path=settings.raw_data_path,
        train_path=settings.train_path,
        test_path=settings.test_path,
        reference_path=settings.reference_path,
        test_size=settings.test_size,
        random_state=settings.random_state,
    )


@task(name="train-and-register", retries=0, timeout_seconds=3600)
def train_task(n_trials: int, register: bool) -> dict[str, Any]:
    settings = get_settings()
    result = train_and_log(
        settings,
        read_dataset(settings.train_path),
        read_dataset(settings.test_path),
        n_trials=n_trials,
        register=register,
        extra_tags={"trigger": "prefect"},
    )
    return result.to_dict()


@task(name="score-current-batch", retries=1)
def score_task() -> dict[str, Any]:
    settings = get_settings()
    service = ModelService(settings)
    service.load()
    result = score_batch(service, settings.current_data_path, settings.reports_dir / "batch")
    return result.to_dict()


@task(name="generate-drift-report", retries=1)
def drift_task(simulate: bool = True) -> dict[str, Any]:
    settings = get_settings()
    if simulate:
        simulate_production_drift(settings.test_path, settings.current_data_path)
    summary = run_drift(
        read_dataset(settings.reference_path),
        read_dataset(settings.current_data_path),
        settings,
    )
    return {
        "drift_share": summary.drift_share,
        "drifted_features": summary.drifted_features,
        "dataset_drift": summary.dataset_drift,
        "report_html": summary.report_html,
    }


@flow(
    name="churnguard-training-pipeline",
    description="End-to-end training, registration, batch scoring and drift reporting.",
    log_prints=True,
    timeout_seconds=7200,
)
def training_pipeline(
    n_trials: int = 10,
    register: bool = True,
    run_drift: bool = True,
    force_ingest: bool = False,
) -> dict[str, Any]:
    logger: Any = get_run_logger()
    ingest_result = ingest_task(force=force_ingest)
    split_result = split_task()
    training_result = train_task(n_trials=n_trials, register=register)
    batch_result = score_task()
    drift_result = drift_task(simulate=True) if run_drift else None

    logger.info(
        "pipeline_complete",
        roc_auc=training_result["metrics"]["roc_auc"],
        promoted=training_result["promoted"],
        version=training_result["version"],
    )
    return {
        "ingest": ingest_result,
        "split": split_result,
        "training": training_result,
        "batch": batch_result,
        "drift": drift_result,
    }
