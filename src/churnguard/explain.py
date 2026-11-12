"""Per-prediction explanations using XGBoost's native TreeSHAP.

No external SHAP dependency: ``Booster.predict(pred_contribs=True)`` returns
exact Shapley values for tree ensembles. Contributions are in log-odds space and
sum to the model margin (plus the base value).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from churnguard.features import unwrap_estimator

EXPLANATION_CAVEAT = (
    "Contributions are exact TreeSHAP values in log-odds space for the "
    "uncalibrated model. Positive values push the prediction toward churn; "
    "calibration changes the probability, not the ranking of drivers."
)


class ExplanationUnavailableError(RuntimeError):
    """Raised when the served model does not expose tree contributions."""


def _transform(estimator: Any, frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    engineered = estimator.named_steps["features"].transform(frame)
    preprocess = estimator.named_steps["preprocess"]
    matrix = np.asarray(preprocess.transform(engineered), dtype=float)
    try:
        names = [str(name) for name in preprocess.get_feature_names_out()]
    except Exception:  # pragma: no cover - defensive
        names = [f"feature_{index}" for index in range(matrix.shape[1])]
    return matrix, names


def _positive_probability(estimator: Any, frame: pd.DataFrame) -> float:
    """Positive-class probability for a single-row frame."""
    probabilities = np.asarray(estimator.predict_proba(frame), dtype=float)
    if probabilities.ndim == 2:
        return float(probabilities[0, -1])
    return float(probabilities.ravel()[0])


def explain_prediction(
    artifact: Any,
    frame: pd.DataFrame,
    *,
    top_k: int = 8,
) -> dict[str, Any]:
    """Compute top SHAP contributions for every row, returning the first row.

    The API is single-customer oriented; batch callers can loop or extend this.
    """
    estimator = unwrap_estimator(artifact)
    classifier = estimator.named_steps["classifier"]
    if not hasattr(classifier, "get_booster"):
        raise ExplanationUnavailableError(
            "The active model does not expose XGBoost tree contributions"
        )

    matrix, names = _transform(estimator, frame)
    import xgboost as xgb

    booster = classifier.get_booster()
    contributions = np.asarray(
        booster.predict(xgb.DMatrix(matrix), pred_contribs=True), dtype=float
    )
    if contributions.ndim == 3:  # multiclass guard
        contributions = contributions[0]
    row = contributions[0]
    if row.shape[0] != len(names) + 1:
        raise ExplanationUnavailableError(
            f"Contribution shape {row.shape} does not match {len(names)} features"
        )

    values = row[:-1]
    base_value = float(row[-1])
    ranked = sorted(
        zip(names, values, strict=True),
        key=lambda item: abs(float(item[1])),
        reverse=True,
    )[: max(top_k, 1)]

    raw_probability = _positive_probability(estimator, frame)
    served_probability = _positive_probability(artifact, frame)
    return {
        "base_value": round(base_value, 6),
        "margin": round(float(values.sum() + base_value), 6),
        "raw_probability": round(raw_probability, 6),
        "calibrated_probability": round(served_probability, 6),
        "top_features": [
            {
                "feature": name,
                "contribution": round(float(value), 6),
                "direction": "increases" if value >= 0 else "decreases",
            }
            for name, value in ranked
        ],
        "caveat": EXPLANATION_CAVEAT,
    }
