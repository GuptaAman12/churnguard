"""Model loading tests: local fallback, metadata and the MLflow registry."""

from __future__ import annotations

import json
from types import SimpleNamespace

import joblib
import mlflow
import pytest
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from churnguard.api.model_loader import ModelNotAvailableError, ModelService


def _write_fallback(settings, pipeline, threshold: float = 0.42) -> None:
    settings.ensure_directories()
    joblib.dump(pipeline, settings.fallback_model_path)
    settings.fallback_model_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "model_name": "fallback-test",
                "version": "9",
                "threshold": threshold,
                "model_uri": "models:/fallback-test/9",
                "metrics": {"roc_auc": 0.83, "f1": 0.6},
            }
        ),
        encoding="utf-8",
    )


def test_fallback_model_loads_with_metadata(settings, trained_pipeline, split_frames) -> None:
    _write_fallback(settings, trained_pipeline)
    service = ModelService(settings)
    bundle = service.load()
    assert bundle.source == "fallback"
    assert bundle.version == "9"
    assert bundle.threshold == 0.42
    assert bundle.metrics["roc_auc"] == 0.83

    features = split_frames[1].drop(columns=["Churn"]).head(10)
    probabilities = service.predict_proba(features)
    assert probabilities.shape == (10,)
    assert service.descriptor()["source"] == "fallback"
    assert service.model_metadata()["metrics"]["f1"] == 0.6


def test_missing_model_raises(settings) -> None:
    service = ModelService(settings)
    with pytest.raises(ModelNotAvailableError):
        service.load()


def test_refresh_is_noop_without_registry(settings, trained_pipeline) -> None:
    _write_fallback(settings, trained_pipeline)
    service = ModelService(settings)
    service.load()
    assert service.refresh_if_changed() is False


def _fake_champion(version: str) -> SimpleNamespace:
    return SimpleNamespace(version=version)


def test_refresh_reloads_when_champion_version_changes(settings, monkeypatch) -> None:
    service = ModelService(settings)
    service._bundle = SimpleNamespace(version="1")  # type: ignore[assignment]
    monkeypatch.setattr(
        service,
        "_client",
        SimpleNamespace(get_model_version_by_alias=lambda *a, **k: _fake_champion("2")),
    )
    calls: list[bool] = []
    monkeypatch.setattr(service, "load", lambda: calls.append(True))

    assert service.refresh_if_changed() is True
    assert calls == [True]


def test_refresh_skips_reload_when_version_is_unchanged(settings, monkeypatch) -> None:
    service = ModelService(settings)
    service._bundle = SimpleNamespace(version="2")  # type: ignore[assignment]
    monkeypatch.setattr(
        service,
        "_client",
        SimpleNamespace(get_model_version_by_alias=lambda *a, **k: _fake_champion("2")),
    )
    calls: list[bool] = []
    monkeypatch.setattr(service, "load", lambda: calls.append(True))

    assert service.refresh_if_changed() is False
    assert calls == []


def test_refresh_returns_false_when_registry_errors(settings, monkeypatch) -> None:
    def boom(*args, **kwargs):
        raise MlflowException("registry unavailable")

    service = ModelService(settings)
    monkeypatch.setattr(service, "_client", SimpleNamespace(get_model_version_by_alias=boom))
    assert service.refresh_if_changed() is False


def test_refresh_is_skipped_for_pinned_model_uri(settings, monkeypatch) -> None:
    pinned = settings.model_copy(update={"model_uri": "models:/pinned/1"})
    service = ModelService(pinned)
    calls: list[bool] = []
    monkeypatch.setattr(service, "load", lambda: calls.append(True))

    assert service.refresh_if_changed() is False
    assert calls == []


def test_unreadable_metadata_is_ignored(settings, trained_pipeline) -> None:
    settings.ensure_directories()
    joblib.dump(trained_pipeline, settings.fallback_model_path)
    settings.fallback_model_path.with_suffix(".json").write_text("{not json", encoding="utf-8")
    bundle = ModelService(settings).load()
    assert bundle.threshold == settings.decision_threshold


def test_registry_model_loads_with_alias_threshold(
    settings, trained_pipeline, split_frames
) -> None:
    mlflow.set_tracking_uri(settings.tracking_uri)
    artifact_dir = settings.data_dir / "mlflow-artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    experiment_id = mlflow.create_experiment(
        "registry-test", artifact_location=artifact_dir.as_uri()
    )
    mlflow.set_experiment(experiment_id=experiment_id)

    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(
            trained_pipeline,
            name="model",
            registered_model_name=settings.registry_model_name,
            serialization_format="pickle",
        )
        run_id = run.info.run_id

    client = MlflowClient()
    versions = client.search_model_versions(
        f"name='{settings.registry_model_name}' and run_id='{run_id}'"
    )
    assert versions
    version = max(versions, key=lambda item: int(item.version)).version
    client.set_registered_model_alias(
        settings.registry_model_name, settings.champion_alias, version
    )
    client.set_model_version_tag(
        settings.registry_model_name, version, "decision_threshold", "0.37"
    )

    service = ModelService(settings)
    bundle = service.load()
    assert bundle.source == "registry"
    assert bundle.version == version
    assert bundle.threshold == pytest.approx(0.37)
    assert service.refresh_if_changed() is False

    features = split_frames[1].drop(columns=["Churn"]).head(5)
    assert service.predict_proba(features).shape == (5,)
