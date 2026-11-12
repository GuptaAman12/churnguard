"""Drift detection, parsing and artifact tests."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from churnguard.monitoring import drift as drift_module
from churnguard.monitoring.drift import (
    _parse_evidently,
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


def test_parse_evidently_v2_payload() -> None:
    payload = {
        "metrics": [
            {
                "metric_name": "DriftedColumnsCount(drift_share=0.5)",
                "config": {"drift_share": 0.5},
                "value": {"count": 2.0, "share": 0.666},
            },
            {
                "metric_name": "ValueDrift(column=age,method=K-S p_value,threshold=0.05)",
                "config": {"column": "age", "method": "K-S p_value", "threshold": 0.05},
                "value": 0.001,
            },
            {
                "metric_name": "ValueDrift(column=income,method=Wasserstein distance (normed),threshold=0.1)",
                "config": {
                    "column": "income",
                    "method": "Wasserstein distance (normed)",
                    "threshold": 0.1,
                },
                "value": 0.2,
            },
            {
                "metric_name": "ValueDrift(column=score,method=Jensen-Shannon distance,threshold=0.1)",
                "config": {
                    "column": "score",
                    "method": "Jensen-Shannon distance",
                    "threshold": 0.1,
                },
                "value": 0.05,
            },
        ]
    }
    features, drifted, total = _parse_evidently(payload, ["age", "income", "score"])
    by_name = {item.feature: item for item in features}
    assert by_name["age"].drift_detected is True  # p-value below threshold
    assert by_name["income"].drift_detected is True  # distance above threshold
    assert by_name["score"].drift_detected is False  # distance below threshold
    assert drifted == 2
    assert total == 3


def test_run_drift_writes_summary_and_html(settings, monkeypatch, raw_frame) -> None:
    monkeypatch.setattr(drift_module, "_run_evidently", lambda *args, **kwargs: None)
    reference = raw_frame.head(300)
    current = raw_frame.tail(300).copy()
    current["MonthlyCharges"] = current["MonthlyCharges"] * 3

    summary = run_drift(reference, current, settings, push=False)
    assert summary.total_features > 0
    assert 0.0 <= summary.drift_share <= 1.0
    assert summary.report_html is not None
    assert settings.drift_summary_path.exists()

    persisted = load_summary(settings.drift_summary_path)
    assert persisted is not None
    assert persisted.drifted_features == summary.drifted_features
    assert list((settings.drift_summary_path.parent / "history").glob("drift-*.json"))


def test_run_drift_marks_dataset_drift(settings, monkeypatch, raw_frame) -> None:
    monkeypatch.setattr(drift_module, "_run_evidently", lambda *args, **kwargs: None)
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
