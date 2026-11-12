"""Shared scoring helpers used by the API, batch jobs and CLI."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

RISK_LOW_FACTOR = 0.6
RISK_MEDIUM_FACTOR = 1.0


def risk_tier(probability: float, threshold: float = 0.5) -> str:
    """Bucket a probability relative to the tuned decision threshold."""
    if probability < threshold * RISK_LOW_FACTOR:
        return "low"
    if probability < threshold * RISK_MEDIUM_FACTOR:
        return "medium"
    return "high"


def score_probabilities(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Attach predictions, labels and risk tiers to an existing feature frame."""
    scores = np.asarray(probabilities, dtype=float).ravel()
    scored = frame.copy()
    scored["churn_probability"] = scores.round(6)
    scored["churn_prediction"] = (scores >= threshold).astype(int)
    scored["risk_tier"] = [risk_tier(value, threshold) for value in scores]
    return scored


def score_frame(
    pipeline: Any,
    frame: pd.DataFrame,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Score a feature dataframe, returning probabilities, labels and risk tiers."""
    probabilities = np.asarray(pipeline.predict_proba(frame)[:, 1], dtype=float)
    return score_probabilities(frame, probabilities, threshold)
