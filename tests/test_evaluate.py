"""Metric and threshold optimisation tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from churnguard.modeling.evaluate import (
    compute_metrics,
    evaluate,
    find_best_threshold,
    subgroup_metrics,
)


def test_compute_metrics_perfect_predictions() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = compute_metrics(y_true, y_proba, threshold=0.5)
    assert metrics["roc_auc"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0


def test_find_best_threshold_on_separable_data() -> None:
    y_true = np.array([0] * 50 + [1] * 50)
    y_proba = np.concatenate([np.linspace(0, 0.4, 50), np.linspace(0.6, 1.0, 50)])
    threshold, score = find_best_threshold(y_true, y_proba)
    assert 0.4 <= threshold <= 0.6
    assert score == 1.0


def test_evaluate_returns_complete_artifact() -> None:
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 200)
    y_proba = rng.random(200)
    result = evaluate(y_true, y_proba)
    assert 0.05 <= result.threshold <= 0.95
    assert sum(result.confusion_matrix.values()) == 200
    assert result.roc_curve
    assert set(result.metrics) >= {"roc_auc", "pr_auc", "f1", "log_loss"}


def test_subgroup_metrics_reports_groups() -> None:
    features = pd.DataFrame(
        {
            "gender": ["Female"] * 60 + ["Male"] * 60,
            "Contract": ["Month-to-month"] * 120,
        }
    )
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 2, 120)
    y_proba = np.clip(y_true * 0.6 + rng.random(120) * 0.4, 0, 1)
    result = subgroup_metrics(features, y_true, y_proba, min_group_size=10)
    assert "gender=Female" in result
    assert "gender=Male" in result
    assert result["gender=Female"]["n"] == 60
