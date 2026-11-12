"""Calibration utilities and artifact tests."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from churnguard.modeling.calibration import (
    CalibratedClassifier,
    build_calibrated_artifact,
    expected_calibration_error,
    fit_calibrator,
    reliability_curve,
)


def test_ece_is_low_for_calibrated_bins() -> None:
    probabilities = np.array([0.1] * 10 + [0.9] * 10)
    y_true = np.array([1] + [0] * 9 + [1] * 9 + [0])
    assert expected_calibration_error(y_true, probabilities) < 0.02


def test_ece_penalises_overconfidence() -> None:
    probabilities = np.full(100, 0.9)
    y_true = np.zeros(100)
    assert expected_calibration_error(y_true, probabilities) == pytest.approx(0.9, abs=0.01)


def test_reliability_curve_covers_every_row() -> None:
    rng = np.random.default_rng(0)
    probabilities = rng.random(500)
    y_true = rng.integers(0, 2, 500)
    curve = reliability_curve(y_true, probabilities, n_bins=10)
    assert sum(row["count"] for row in curve) == 500
    assert all(0.0 <= row["mean_predicted"] <= 1.0 for row in curve)
    assert all(0.0 <= row["observed_frequency"] <= 1.0 for row in curve)


def test_calibrated_classifier_is_picklable(trained_pipeline, split_frames) -> None:
    _, test = split_frames
    features = test.drop(columns=["Churn"]).head(20)
    scores = np.linspace(0.01, 0.99, 100)
    calibrator = fit_calibrator(scores, (scores > 0.5).astype(int))

    artifact = CalibratedClassifier(
        estimator=trained_pipeline, calibrator=calibrator, method="isotonic"
    )
    artifact.estimator_ = trained_pipeline
    artifact.calibrator_ = calibrator
    artifact.classes_ = np.array([0, 1])

    probabilities = artifact.predict_proba(features)
    assert probabilities.shape == (20, 2)
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    assert np.allclose(probabilities.sum(axis=1), 1.0)

    restored = pickle.loads(pickle.dumps(artifact))
    np.testing.assert_allclose(restored.predict_proba(features), probabilities)


def test_build_calibrated_artifact_returns_valid_report(trained_pipeline, split_frames) -> None:
    train, test = split_frames
    features_train = train.drop(columns=["Churn"])
    target_train = train["Churn"].to_numpy()
    features_test = test.drop(columns=["Churn"])
    target_test = test["Churn"].to_numpy()

    raw_test = trained_pipeline.predict_proba(features_test)[:, 1]
    out_of_fold = trained_pipeline.predict_proba(features_train)[:, 1]

    artifact, report = build_calibrated_artifact(
        trained_pipeline,
        out_of_fold,
        target_train,
        target_test,
        raw_test,
    )
    assert report.method in {"isotonic", "identity"}
    assert report.reliability
    assert report.raw.log_loss > 0
    served = artifact.predict_proba(features_test)[:, 1]
    assert served.shape == raw_test.shape
    assert np.all((served >= 0) & (served <= 1))
