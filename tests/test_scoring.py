"""Scoring helpers and batch-inference tests."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from churnguard.batch import score_batch
from churnguard.data import write_parquet
from churnguard.scoring import risk_tier, score_frame, score_probabilities
from tests.conftest import StubModelService


def test_risk_tier_boundaries() -> None:
    assert risk_tier(0.1, 0.5) == "low"
    assert risk_tier(0.4, 0.5) == "medium"
    assert risk_tier(0.5, 0.5) == "high"
    assert risk_tier(0.9, 0.5) == "high"


def test_score_probabilities_adds_contract_columns() -> None:
    frame = pd.DataFrame({"tenure": [1, 2, 3]})
    scored = score_probabilities(frame, np.array([0.1, 0.5, 0.9]), threshold=0.5)
    assert scored["churn_prediction"].tolist() == [0, 1, 1]
    assert scored["risk_tier"].tolist() == ["low", "high", "high"]
    assert scored["churn_probability"].tolist() == [0.1, 0.5, 0.9]


def test_score_frame_uses_pipeline_probabilities(trained_pipeline, split_frames) -> None:
    _, test = split_frames
    features = test.drop(columns=["Churn"]).head(10)
    scored = score_frame(trained_pipeline, features, threshold=0.5)
    assert len(scored) == 10
    assert scored["churn_probability"].between(0, 1).all()


def test_score_batch_writes_parquet_and_summary(settings, stub_service, split_frames) -> None:
    _, test = split_frames
    features = test.drop(columns=["Churn"]).head(25)
    input_path = settings.data_dir / "batch_input.parquet"
    write_parquet(features, input_path)

    result = score_batch(stub_service, input_path, settings.reports_dir / "batch")
    assert result.rows == 25
    assert result.mean_probability == pytest.approx(0.73)
    assert result.predicted_churn_rate == 1.0
    assert (settings.reports_dir / "batch" / "predictions.parquet").exists()
    summary = json.loads(
        (settings.reports_dir / "batch" / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["rows"] == 25
    assert summary["risk_tier_counts"]["high"] == 25
    assert summary["model"]["source"] == "injected"


def test_score_batch_rejects_missing_columns(settings) -> None:
    input_path = settings.data_dir / "broken.parquet"
    write_parquet(pd.DataFrame({"tenure": [1, 2]}), input_path)
    with pytest.raises(ValueError, match="missing required columns"):
        score_batch(StubModelService(), input_path, settings.reports_dir / "batch")
