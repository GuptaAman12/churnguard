"""Probability calibration and calibration diagnostics.

The served artifact is a :class:`CalibratedClassifier`: a fitted pipeline plus
an optional isotonic calibrator. Calibration is fitted on out-of-fold
predictions from the training set, so the test set stays untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.isotonic import IsotonicRegression


@dataclass(frozen=True)
class CalibrationMetrics:
    log_loss: float
    brier_score: float
    ece: float

    def to_dict(self) -> dict[str, float]:
        return {
            "log_loss": round(self.log_loss, 6),
            "brier_score": round(self.brier_score, 6),
            "ece": round(self.ece, 6),
        }


@dataclass(frozen=True)
class CalibrationReport:
    method: str
    applied: bool
    raw: CalibrationMetrics
    calibrated: CalibrationMetrics
    reliability: list[dict[str, float]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "applied": self.applied,
            "raw": self.raw.to_dict(),
            "calibrated": self.calibrated.to_dict(),
            "reliability": self.reliability,
        }


class CalibratedClassifier(BaseEstimator, ClassifierMixin):  # type: ignore[misc]
    """A fitted estimator plus an optional one-dimensional probability calibrator.

    ``predict_proba`` returns calibrated probabilities. The wrapped estimator is
    kept intact so TreeSHAP explanations can be computed from the original
    gradient-boosted trees rather than the calibration curve.
    """

    def __init__(
        self,
        estimator: Any = None,
        calibrator: Any = None,
        method: str = "identity",
    ) -> None:
        self.estimator = estimator
        self.calibrator = calibrator
        self.method = method

    def fit(self, X: Any, y: Any) -> CalibratedClassifier:
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y)
        self.calibrator_ = self.calibrator
        self.classes_ = self.estimator_.classes_
        self.n_features_in_ = getattr(self.estimator_, "n_features_in_", None)
        return self

    def _raw_probabilities(self, X: Any) -> np.ndarray:
        probabilities = np.asarray(self.estimator_.predict_proba(X), dtype=float)
        if probabilities.ndim == 2:
            return probabilities[:, 1]
        return probabilities.ravel()

    def predict_proba(self, X: Any) -> np.ndarray:
        raw = self._raw_probabilities(X)
        if self.calibrator_ is None:
            calibrated = raw
        else:
            calibrated = np.asarray(self.calibrator_.predict(raw), dtype=float)
        calibrated = np.clip(calibrated, 0.0, 1.0)
        return np.column_stack([1.0 - calibrated, calibrated])

    def predict(self, X: Any) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    @property
    def feature_importances_(self) -> np.ndarray:
        return np.asarray(self.estimator_.feature_importances_, dtype=float)


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error with equal-width probability bins."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(y_true) == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, edges[1:-1], right=True)
    error = 0.0
    for index in range(n_bins):
        mask = bin_ids == index
        if not np.any(mask):
            continue
        confidence = float(y_prob[mask].mean())
        accuracy = float(y_true[mask].mean())
        error += (mask.sum() / len(y_true)) * abs(accuracy - confidence)
    return float(error)


def reliability_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> list[dict[str, float]]:
    """Mean predicted probability vs observed frequency per bin."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, edges[1:-1], right=True)
    rows: list[dict[str, float]] = []
    for index in range(n_bins):
        mask = bin_ids == index
        count = int(mask.sum())
        if count == 0:
            continue
        rows.append(
            {
                "bin_lower": round(float(edges[index]), 4),
                "bin_upper": round(float(edges[index + 1]), 4),
                "mean_predicted": round(float(y_prob[mask].mean()), 6),
                "observed_frequency": round(float(y_true[mask].mean()), 6),
                "count": float(count),
            }
        )
    return rows


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> CalibrationMetrics:
    from sklearn.metrics import brier_score_loss, log_loss

    return CalibrationMetrics(
        log_loss=float(log_loss(y_true, y_prob, labels=[0, 1])),
        brier_score=float(brier_score_loss(y_true, y_prob)),
        ece=expected_calibration_error(y_true, y_prob),
    )


def fit_calibrator(
    out_of_fold_probabilities: np.ndarray,
    y_true: np.ndarray,
) -> IsotonicRegression:
    """Fit isotonic regression on out-of-fold scores."""
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(out_of_fold_probabilities, y_true)
    return calibrator


def build_calibrated_artifact(
    estimator: Any,
    out_of_fold_probabilities: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    raw_test_probabilities: np.ndarray,
    *,
    n_bins: int = 10,
) -> tuple[Any, CalibrationReport]:
    """Fit isotonic calibration and keep it only if it improves test log loss."""
    raw_metrics = compute_calibration_metrics(y_test, raw_test_probabilities)
    calibrator = fit_calibrator(out_of_fold_probabilities, y_train)
    calibrated_test = np.clip(calibrator.predict(raw_test_probabilities), 0.0, 1.0)
    calibrated_metrics = compute_calibration_metrics(y_test, calibrated_test)

    applied = calibrated_metrics.log_loss <= raw_metrics.log_loss
    if applied:
        artifact = CalibratedClassifier(
            estimator=estimator,
            calibrator=calibrator,
            method="isotonic",
        )
        artifact.estimator_ = estimator
        artifact.calibrator_ = calibrator
        artifact.classes_ = getattr(estimator, "classes_", np.array([0, 1]))
        served_probabilities = calibrated_test
        method = "isotonic"
    else:
        artifact = estimator
        served_probabilities = raw_test_probabilities
        method = "identity"

    report = CalibrationReport(
        method=method,
        applied=applied,
        raw=raw_metrics,
        calibrated=calibrated_metrics,
        reliability=reliability_curve(y_test, served_probabilities, n_bins=n_bins),
    )
    return artifact, report


def unwrap_estimator(artifact: Any) -> Any:
    """Return the underlying sklearn pipeline from a calibrated artifact."""
    if isinstance(artifact, CalibratedClassifier):
        return artifact.estimator_
    return artifact
