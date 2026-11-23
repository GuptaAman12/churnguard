"""Drift detection and artifact tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from churnguard.monitoring.drift import (
    load_prediction_features,
    run_drift,
    simple_drift,
)
from churnguard.monitoring.summary import load_summary


def test_simple_drift_detects_shifted_numeric() -> None:
    rng = np.random.default_rng(0)
    reference = pd.DataFrame({"x": rng.normal(0, 1, 500)})
    current = pd.DataFrame({"x": rng.normal(5, 1, 500)})
    features, drifted, total = simple_drift(reference, current, ["x"])
    assert drifted == 1
    assert total == 1
    assert features[0].drift_detected is True
    assert features[0].method == "ks_2samp"


def test_simple_drift_does_not_flag_identical_data() -> None:
    rng = np.random.default_rng(1)
    values = rng.normal(0, 1, 500)
    reference = pd.DataFrame({"x": values, "category": ["A", "B"] * 250})
    current = reference.copy()
    features, drifted, _ = simple_drift(reference, current, ["x", "category"])
    assert drifted == 0
    assert all(not item.drift_detected for item in features)


def test_run_drift_writes_summary_and_html(settings, raw_frame) -> None:
    reference = raw_frame.head(300)
    current = raw_frame.tail(300).copy()
    current["MonthlyCharges"] = current["MonthlyCharges"] * 3

    summary = run_drift(reference, current, settings, push=False)
    assert summary.total_features > 0
    assert 0.0 <= summary.drift_share <= 1.0
    assert summary.report_html is not None
    report = Path(summary.report_html)
    assert report.exists()
    assert "ChurnGuard data-drift report" in report.read_text(encoding="utf-8")
    assert settings.drift_summary_path.exists()

    persisted = load_summary(settings.drift_summary_path)
    assert persisted is not None
    assert persisted.drifted_features == summary.drifted_features
    assert list((settings.drift_summary_path.parent / "history").glob("drift-*.json"))


def test_run_drift_marks_dataset_drift(settings, raw_frame) -> None:
    reference = raw_frame.head(300)
    current = raw_frame.head(300).copy()
    current["tenure"] = current["tenure"] + 50
    current["MonthlyCharges"] = current["MonthlyCharges"] * 4
    current["Contract"] = "Month-to-month"
    current["TechSupport"] = "No"
    current["InternetService"] = "Fiber optic"
    current["PaymentMethod"] = "Electronic check"
    current["PaperlessBilling"] = "Yes"
    current["OnlineSecurity"] = "No"
    current["StreamingTV"] = "Yes"
    summary = run_drift(reference, current, settings, push=False)
    assert summary.dataset_drift is True


def test_load_prediction_features(tmp_path) -> None:
    path = tmp_path / "predictions.jsonl"
    rows = [
        {"features": {"tenure": 1, "gender": "Female"}},
        {"features": {"tenure": 2, "gender": "Male"}},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    frame = load_prediction_features(path)
    assert len(frame) == 2
    assert list(frame.columns) == ["tenure", "gender"]
