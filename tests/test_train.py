"""End-to-end training integration tests against a local SQLite MLflow store."""

from __future__ import annotations

import json

import mlflow
import pytest

from churnguard.modeling.train import TrainResult, train_and_log


@pytest.fixture
def tracking_settings(settings):
    settings.ensure_directories()
    mlflow.set_tracking_uri(settings.tracking_uri)
    mlflow.create_experiment(
        settings.experiment_name,
        artifact_location=(settings.data_dir / "artifacts").as_uri(),
    )
    return settings


def test_train_and_log_without_registration(tracking_settings, split_frames) -> None:
    train, test = split_frames
    result = train_and_log(
        tracking_settings,
        train,
        test,
        n_trials=0,
        register=False,
        save_fallback=True,
    )
    assert isinstance(result, TrainResult)
    assert result.run_id
    assert result.metrics["roc_auc"] > 0.6
    assert result.version is None
    assert tracking_settings.fallback_model_path.exists()

    metadata = json.loads(
        tracking_settings.fallback_model_path.with_suffix(".json").read_text(encoding="utf-8")
    )
    assert metadata["threshold"] == pytest.approx(result.threshold)
    assert metadata["metrics"]["roc_auc"] == pytest.approx(result.metrics["roc_auc"])

    run = mlflow.get_run(result.run_id)
    assert run.data.metrics["test_roc_auc"] == pytest.approx(result.metrics["roc_auc"])
    assert run.data.tags["git_sha"]
    assert run.data.params["model_type"] == "xgboost"

    artifacts = mlflow.artifacts.list_artifacts(run_id=result.run_id)
    artifact_names = {artifact.path for artifact in artifacts}
    assert "model_card.md" in artifact_names
    assert "evaluation" in artifact_names

    # MLflow 3 tracks models as first-class logged models rather than run artifacts.
    logged_model = mlflow.models.get_model_info(result.model_uri)
    assert logged_model.signature is not None
    assert logged_model.model_uri == result.model_uri


def test_train_registers_and_promotes_first_model(tracking_settings, split_frames) -> None:
    train, test = split_frames
    settings = tracking_settings.model_copy(update={"min_roc_auc": 0.5})
    result = train_and_log(
        settings,
        train,
        test,
        n_trials=0,
        register=True,
        save_fallback=False,
    )
    assert result.version is not None
    assert result.promoted is True
    assert "no champion" in result.promotion_reason

    client = mlflow.MlflowClient()
    champion = client.get_model_version_by_alias(
        settings.registry_model_name, settings.champion_alias
    )
    assert champion.version == result.version
    assert champion.tags["decision_threshold"]
