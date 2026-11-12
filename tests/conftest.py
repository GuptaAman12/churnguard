"""Shared pytest fixtures."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from churnguard.api.model_loader import ModelNotAvailableError
from churnguard.config import Settings
from churnguard.constants import RAW_FEATURE_COLUMNS
from churnguard.data import clean_data, split_data
from churnguard.features import build_pipeline
from churnguard.modeling.train import make_classifier
from churnguard.synthetic import generate_churn_dataset


@pytest.fixture(scope="session")
def raw_frame() -> pd.DataFrame:
    return generate_churn_dataset(rows=600, seed=11)


@pytest.fixture(scope="session")
def cleaned_frame(raw_frame: pd.DataFrame) -> pd.DataFrame:
    return clean_data(raw_frame)


@pytest.fixture(scope="session")
def split_frames(cleaned_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return split_data(cleaned_frame, test_size=0.25, random_state=1)


@pytest.fixture(scope="session")
def trained_pipeline(split_frames: tuple[pd.DataFrame, pd.DataFrame]) -> Any:
    train, _ = split_frames
    pipeline = build_pipeline(
        make_classifier(
            {
                "n_estimators": 40,
                "max_depth": 3,
                "learning_rate": 0.1,
            },
            random_state=1,
            scale_pos_weight=2.0,
        )
    )
    pipeline.fit(train.drop(columns=["Churn"]), train["Churn"])
    return pipeline


@pytest.fixture
def settings(tmp_path: Any) -> Settings:
    base = tmp_path
    return Settings(
        env="test",
        data_dir=base / "data",
        raw_data_path=base / "data" / "raw.csv",
        train_path=base / "data" / "processed" / "train.parquet",
        test_path=base / "data" / "processed" / "test.parquet",
        reference_path=base / "data" / "reference" / "reference.parquet",
        current_data_path=base / "data" / "current" / "batch.parquet",
        models_dir=base / "models",
        fallback_model_path=base / "models" / "model.joblib",
        reports_dir=base / "reports",
        drift_summary_path=base / "reports" / "drift" / "latest.json",
        prediction_log_path=base / "logs" / "predictions.jsonl",
        tracking_uri=f"sqlite:///{(base / 'mlflow.db').as_posix()}",
        api_key=None,
        prediction_log_enabled=True,
        log_json=False,
    )


class StubModelService:
    """In-process stand-in for ModelService used by API tests."""

    def __init__(
        self,
        probability: float = 0.73,
        threshold: float = 0.5,
        loaded: bool = True,
        source: str = "injected",
    ) -> None:
        self._probability = probability
        self._threshold = threshold
        self.is_loaded = loaded
        self.load_calls = 0
        self.source = source
        self.version = "7" if loaded else None

    def load(self) -> SimpleNamespace:
        self.load_calls += 1
        if not self.is_loaded:
            raise ModelNotAvailableError("stub model is unavailable")
        return SimpleNamespace(source=self.source, version=self.version, loaded_at="2026-01-01")

    def refresh_if_changed(self) -> bool:
        return False

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return np.full(len(frame), self._probability, dtype=float)

    def descriptor(self) -> dict[str, Any]:
        return {
            "name": "churnguard-test",
            "version": self.version,
            "alias": None,
            "source": self.source,
            "model_uri": "models:/test/1",
            "threshold": self._threshold,
            "loaded_at": "2026-01-01T00:00:00+00:00",
        }

    def model_metadata(self) -> dict[str, Any]:
        return {**self.descriptor(), "metrics": {"roc_auc": 0.84}}


@pytest.fixture
def stub_service() -> StubModelService:
    return StubModelService()


def build_customer_payload(
    row: pd.Series | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    if row is None:
        row = generate_churn_dataset(rows=1, seed=3).iloc[0]
    payload: dict[str, Any] = {}
    for column in RAW_FEATURE_COLUMNS:
        value = row[column]
        if column in ("SeniorCitizen", "tenure"):
            payload[column] = int(value)
        elif column in ("MonthlyCharges", "TotalCharges"):
            payload[column] = float(value) if str(value).strip() not in {"", "nan"} else 0.0
        else:
            payload[column] = str(value)
    payload.update(overrides)
    return payload


@pytest.fixture
def customer_payload() -> dict[str, Any]:
    return build_customer_payload()
