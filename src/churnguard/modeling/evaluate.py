"""Model evaluation metrics and decision-threshold optimisation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


@dataclass(frozen=True)
class EvaluationResult:
    metrics: dict[str, float]
    threshold: float
    confusion_matrix: dict[str, int]
    roc_curve: list[dict[str, float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "metrics": self.metrics,
            "threshold": self.threshold,
            "confusion_matrix": self.confusion_matrix,
        }


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def compute_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute discrimination, calibration and classification metrics."""
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": _safe_divide(tn, tn + fp),
        "npv": _safe_divide(tn, tn + fn),
        "log_loss": float(log_loss(y_true, y_proba, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_true, y_proba)),
        "true_positives": float(tp),
        "true_negatives": float(tn),
        "false_positives": float(fp),
        "false_negatives": float(fn),
    }


def find_best_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    grid: np.ndarray | None = None,
) -> tuple[float, float]:
    """Return the probability cutoff that maximises F1 on the given data."""
    if grid is None:
        grid = np.linspace(0.05, 0.95, 91)
    scores = [f1_score(y_true, (y_proba >= t).astype(int), zero_division=0) for t in grid]
    best_index = int(np.argmax(scores))
    return float(grid[best_index]), float(scores[best_index])


def subgroup_metrics(
    features: pd.DataFrame,
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    columns: tuple[str, ...] = ("gender", "SeniorCitizen", "Contract"),
    threshold: float = 0.5,
    min_group_size: int = 30,
) -> dict[str, dict[str, float]]:
    """Disaggregated performance metrics for fairness / bias reviews."""
    results: dict[str, dict[str, float]] = {}
    for column in columns:
        if column not in features.columns:
            continue
        values = features[column].dropna().unique()
        for value in values:
            mask = (features[column] == value).to_numpy()
            if int(mask.sum()) < min_group_size:
                continue
            group_y = y_true[mask]
            group_proba = y_proba[mask]
            if len(np.unique(group_y)) < 2:
                continue
            predictions = (group_proba >= threshold).astype(int)
            results[f"{column}={value}"] = {
                "roc_auc": round(float(roc_auc_score(group_y, group_proba)), 6),
                "recall": round(float(recall_score(group_y, predictions, zero_division=0)), 6),
                "precision": round(
                    float(precision_score(group_y, predictions, zero_division=0)), 6
                ),
                "churn_rate": round(float(group_y.mean()), 6),
                "n": float(len(group_y)),
            }
    return results


def evaluate(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float | None = None,
) -> EvaluationResult:
    """Evaluate a model, optionally tuning the decision threshold."""
    tuned_threshold, _ = find_best_threshold(y_true, y_proba)
    chosen = tuned_threshold if threshold is None else float(threshold)
    metrics = compute_metrics(y_true, y_proba, chosen)
    y_pred = (y_proba >= chosen).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    # Downsample the curve to keep JSON artifacts small.
    stride = max(len(fpr) // 100, 1)
    curve = [
        {"fpr": round(float(f), 4), "tpr": round(float(t), 4)}
        for f, t in zip(fpr[::stride], tpr[::stride], strict=False)
    ]
    return EvaluationResult(
        metrics={key: round(value, 6) for key, value in metrics.items()},
        threshold=round(chosen, 4),
        confusion_matrix={
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        roc_curve=curve,
    )
