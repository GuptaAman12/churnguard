"""Model loading from the MLflow registry with an offline joblib fallback."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from churnguard.api import metrics
from churnguard.config import Settings
from churnguard.logging import get_logger

logger = get_logger(__name__)


class ModelNotAvailableError(RuntimeError):
    """Raised when neither the registry nor the local fallback has a model."""


@dataclass
class ModelBundle:
    pipeline: Any
    name: str
    version: str | None
    alias: str | None
    source: str
    model_uri: str | None
    threshold: float
    loaded_at: str
    loaded_at_epoch: float
    metrics: dict[str, float] = field(default_factory=dict)


class ModelService:
    """Hot-reloadable model holder; safe for concurrent prediction requests."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._bundle: ModelBundle | None = None
        self._lock = threading.RLock()
        settings.ensure_directories()
        mlflow.set_tracking_uri(settings.tracking_uri)
        try:
            self._client = MlflowClient(tracking_uri=settings.tracking_uri)
        except Exception:  # pragma: no cover - client construction is lazy
            self._client = MlflowClient()

    # -- state --------------------------------------------------------------
    @property
    def is_loaded(self) -> bool:
        return self._bundle is not None

    def load(self) -> ModelBundle:
        """Load the champion model, falling back to the local joblib artifact."""
        with self._lock:
            bundle = self._from_registry() or self._from_fallback()
            if bundle is None:
                metrics.record_model_unloaded()
                raise ModelNotAvailableError(
                    "No model available in the MLflow registry or local fallback path"
                )
            self._bundle = bundle
            metrics.record_model_loaded(
                name=bundle.name,
                version=bundle.version,
                source=bundle.source,
                alias=bundle.alias,
                loaded_at_epoch=bundle.loaded_at_epoch,
            )
            logger.info(
                "model_loaded",
                source=bundle.source,
                version=bundle.version,
                model_uri=bundle.model_uri,
                threshold=bundle.threshold,
            )
            return bundle

    def refresh_if_changed(self) -> bool:
        """Poll the registry; reload only when the champion version changes."""
        # A pinned model URI is immutable by design; polling would only churn.
        if self.settings.model_uri is not None:
            return False
        try:
            champion = self._client.get_model_version_by_alias(
                self.settings.registry_model_name, self.settings.champion_alias
            )
        except Exception:
            return False
        current = self._bundle
        if current is not None and current.version == champion.version:
            return False
        self.load()
        return True

    # -- loading ------------------------------------------------------------
    def _from_registry(self) -> ModelBundle | None:
        try:
            version_tag: str | None = None
            threshold: float | None = None
            model_uri = self.settings.model_uri

            if model_uri is None:
                model_uri = (
                    f"models:/{self.settings.registry_model_name}@{self.settings.champion_alias}"
                )
                champion = self._client.get_model_version_by_alias(
                    self.settings.registry_model_name, self.settings.champion_alias
                )
                version_tag = champion.version
                threshold = _to_float(champion.tags.get("decision_threshold"))

            pipeline = mlflow.sklearn.load_model(model_uri)
            now = time.time()
            return ModelBundle(
                pipeline=pipeline,
                name=self.settings.registry_model_name,
                version=version_tag,
                alias=self.settings.champion_alias,
                source="registry",
                model_uri=model_uri,
                threshold=threshold or self.settings.decision_threshold,
                loaded_at=datetime.fromtimestamp(now, tz=UTC).isoformat(),
                loaded_at_epoch=now,
            )
        except (MlflowException, OSError, ValueError, RuntimeError) as exc:
            logger.warning("registry_load_failed", error=str(exc))
            return None

    def _from_fallback(self) -> ModelBundle | None:
        path = Path(self.settings.fallback_model_path)
        if not path.exists():
            logger.error("fallback_model_missing", path=str(path))
            return None
        try:
            pipeline = joblib.load(path)
        except Exception as exc:  # pragma: no cover - corrupt artifact
            logger.error("fallback_load_failed", path=str(path), error=str(exc))
            return None

        metadata = _read_metadata(path.with_suffix(".json"))
        now = time.time()
        return ModelBundle(
            pipeline=pipeline,
            name=str(metadata.get("model_name", self.settings.registry_model_name)),
            version=_str_or_none(metadata.get("version")),
            alias=None,
            source="fallback",
            model_uri=_str_or_none(metadata.get("model_uri")),
            threshold=_to_float(metadata.get("threshold")) or self.settings.decision_threshold,
            loaded_at=datetime.fromtimestamp(now, tz=UTC).isoformat(),
            loaded_at_epoch=now,
            metrics={
                key: float(value)
                for key, value in dict(metadata.get("metrics") or {}).items()
                if isinstance(value, int | float)
            },
        )

    # -- inference ----------------------------------------------------------
    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        with self._lock:
            bundle = self._bundle
        if bundle is None:
            raise ModelNotAvailableError("Model is not loaded")
        probabilities = bundle.pipeline.predict_proba(frame)
        if probabilities.ndim == 2:
            return np.asarray(probabilities[:, 1], dtype=float)
        return np.asarray(probabilities, dtype=float).ravel()

    def descriptor(self) -> dict[str, Any]:
        with self._lock:
            bundle = self._bundle
        if bundle is None:
            raise ModelNotAvailableError("Model is not loaded")
        return {
            "name": bundle.name,
            "version": bundle.version,
            "alias": bundle.alias,
            "source": bundle.source,
            "model_uri": bundle.model_uri,
            "threshold": bundle.threshold,
            "loaded_at": bundle.loaded_at,
        }

    def model_metadata(self) -> dict[str, Any]:
        with self._lock:
            bundle = self._bundle
        if bundle is None:
            raise ModelNotAvailableError("Model is not loaded")
        return {
            **self.descriptor(),
            "metrics": bundle.metrics,
        }

    def explain(self, frame: pd.DataFrame, *, top_k: int = 8) -> dict[str, Any]:
        """Return TreeSHAP contributions for the first row of ``frame``."""
        from churnguard.explain import explain_prediction

        with self._lock:
            bundle = self._bundle
        if bundle is None:
            raise ModelNotAvailableError("Model is not loaded")
        return explain_prediction(bundle.pipeline, frame, top_k=top_k)


def _read_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _to_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _str_or_none(value: Any) -> str | None:
    return str(value) if value is not None else None
