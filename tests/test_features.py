"""Feature engineering and pipeline contract tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from churnguard.constants import SERVICE_COLUMNS
from churnguard.features import engineer_features


def test_engineer_features_adds_derived_columns(raw_frame: pd.DataFrame) -> None:
    engineered = engineer_features(raw_frame)
    for column in (
        "avg_monthly_spend",
        "service_count",
        "is_month_to_month",
        "has_support",
    ):
        assert column in engineered.columns
    assert engineered["service_count"].between(0, len(SERVICE_COLUMNS)).all()
    assert engineered["avg_monthly_spend"].ge(0).all()


def test_engineer_features_handles_zero_tenure() -> None:
    frame = pd.DataFrame(
        {
            "tenure": [0, 10],
            "TotalCharges": [0.0, 500.0],
            "Contract": ["Month-to-month", "Two year"],
            "TechSupport": ["No", "Yes"],
            "OnlineSecurity": ["No", "No"],
            "PhoneService": ["Yes", "No"],
            "MultipleLines": ["No", "No phone service"],
            "InternetService": ["DSL", "No"],
            "OnlineBackup": ["No", "No internet service"],
            "DeviceProtection": ["No", "No internet service"],
            "StreamingTV": ["No", "No internet service"],
            "StreamingMovies": ["No", "No internet service"],
        }
    )
    engineered = engineer_features(frame)
    assert engineered.loc[0, "avg_monthly_spend"] == 0.0
    assert engineered.loc[1, "avg_monthly_spend"] == 50.0
    assert engineered.loc[0, "service_count"] == 1
    assert engineered.loc[0, "is_month_to_month"] == 1


def test_pipeline_predicts_probabilities(trained_pipeline, split_frames) -> None:
    _, test = split_frames
    probabilities = trained_pipeline.predict_proba(test.drop(columns=["Churn"]))[:, 1]
    assert probabilities.shape == (len(test),)
    assert np.all((probabilities >= 0) & (probabilities <= 1))


def test_pipeline_is_robust_to_unknown_categories(trained_pipeline, split_frames) -> None:
    _, test = split_frames
    frame = test.drop(columns=["Churn"]).head(5).copy()
    frame["Contract"] = "Three year"
    probabilities = trained_pipeline.predict_proba(frame)
    assert probabilities.shape[0] == 5


def test_build_pipeline_composes_expected_steps(trained_pipeline) -> None:
    assert list(trained_pipeline.named_steps) == ["features", "preprocess", "classifier"]
