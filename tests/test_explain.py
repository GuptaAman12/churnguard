"""TreeSHAP explanation tests."""

from __future__ import annotations

import pytest
from sklearn.linear_model import LogisticRegression

from churnguard.explain import ExplanationUnavailableError, explain_prediction
from churnguard.features import build_pipeline


def test_explain_returns_ranked_contributions(trained_pipeline, split_frames) -> None:
    _, test = split_frames
    frame = test.drop(columns=["Churn"]).head(1)
    result = explain_prediction(trained_pipeline, frame, top_k=5)

    assert len(result["top_features"]) == 5
    magnitudes = [abs(item["contribution"]) for item in result["top_features"]]
    assert magnitudes == sorted(magnitudes, reverse=True)
    assert all(item["direction"] in {"increases", "decreases"} for item in result["top_features"])
    expected = float(trained_pipeline.predict_proba(frame)[:, 1][0])
    assert result["raw_probability"] == pytest.approx(expected, abs=1e-6)
    assert result["calibrated_probability"] == pytest.approx(expected, abs=1e-6)
    assert result["caveat"]


def test_contributions_sum_to_margin(trained_pipeline, split_frames) -> None:
    _, test = split_frames
    frame = test.drop(columns=["Churn"]).head(1)
    result = explain_prediction(trained_pipeline, frame, top_k=10_000)
    total = result["base_value"] + sum(item["contribution"] for item in result["top_features"])
    assert result["margin"] == pytest.approx(total, abs=1e-4)


def test_explain_accepts_calibrated_artifact(trained_pipeline, split_frames) -> None:
    import numpy as np

    from churnguard.modeling.calibration import CalibratedClassifier, fit_calibrator

    _, test = split_frames
    frame = test.drop(columns=["Churn"]).head(1)
    scores = np.linspace(0.01, 0.99, 100)
    calibrator = fit_calibrator(scores, (scores > 0.5).astype(int))
    artifact = CalibratedClassifier(
        estimator=trained_pipeline, calibrator=calibrator, method="isotonic"
    )
    artifact.estimator_ = trained_pipeline
    artifact.calibrator_ = calibrator
    artifact.classes_ = trained_pipeline.classes_

    result = explain_prediction(artifact, frame, top_k=3)
    assert len(result["top_features"]) == 3


def test_non_tree_models_are_rejected(split_frames) -> None:
    train, _ = split_frames
    pipeline = build_pipeline(LogisticRegression(max_iter=200))
    pipeline.fit(train.drop(columns=["Churn"]), train["Churn"])
    with pytest.raises(ExplanationUnavailableError):
        explain_prediction(pipeline, train.drop(columns=["Churn"]).head(1))
