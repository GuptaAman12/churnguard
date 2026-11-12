"""Prefect flow definitions remain importable and correctly configured."""

from __future__ import annotations

from churnguard.flows.monitoring import monitoring_pipeline
from churnguard.flows.training import train_task, training_pipeline


def test_training_flow_metadata() -> None:
    assert training_pipeline.name == "churnguard-training-pipeline"
    assert training_pipeline.timeout_seconds == 7200


def test_monitoring_flow_metadata() -> None:
    assert monitoring_pipeline.name == "churnguard-monitoring-pipeline"


def test_training_task_has_retry_policy() -> None:
    assert train_task.retries == 0
    assert train_task.timeout_seconds == 3600
