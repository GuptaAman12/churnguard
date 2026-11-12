"""Feature engineering and the reusable scikit-learn preprocessing pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from churnguard.constants import CATEGORICAL_FEATURES, NUMERIC_FEATURES, SERVICE_COLUMNS


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic feature engineering shared by training and serving.

    The function is embedded *inside* the sklearn pipeline so online and offline
    transformations can never diverge (no training/serving skew).
    """
    out = df.copy()
    tenure = out["tenure"].clip(lower=1)
    total_charges = pd.to_numeric(out["TotalCharges"], errors="coerce").fillna(0.0)
    out["avg_monthly_spend"] = (total_charges / tenure).astype(float)
    out["service_count"] = sum(
        (out[column].astype(str) == "Yes").astype(int) for column in SERVICE_COLUMNS
    )
    out["is_month_to_month"] = (out["Contract"] == "Month-to-month").astype(int)
    out["has_support"] = ((out["TechSupport"] == "Yes") | (out["OnlineSecurity"] == "Yes")).astype(
        int
    )
    return out


def build_preprocessor() -> ColumnTransformer:
    """Numeric + categorical preprocessing with unknown-category safety."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    min_frequency=20,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, list(NUMERIC_FEATURES)),
            ("cat", categorical_pipeline, list(CATEGORICAL_FEATURES)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(classifier: object) -> Pipeline:
    """Full inference artifact: feature engineering -> preprocessing -> model."""
    return Pipeline(
        steps=[
            ("features", FunctionTransformer(engineer_features, validate=False)),
            ("preprocess", build_preprocessor()),
            ("classifier", classifier),
        ]
    )


def unwrap_estimator(artifact: object) -> Any:
    """Return the sklearn pipeline inside an optional calibration wrapper."""
    inner = getattr(artifact, "estimator_", None)
    if inner is not None and hasattr(inner, "named_steps"):
        return inner
    return artifact


def get_feature_names(pipeline: Pipeline | object) -> list[str]:
    """Extract transformed feature names when the classifier exposes them."""
    estimator = unwrap_estimator(pipeline)
    preprocessor: ColumnTransformer = estimator.named_steps["preprocess"]
    try:
        return [str(name) for name in preprocessor.get_feature_names_out()]
    except Exception:  # pragma: no cover - defensive
        return []


def model_feature_importance(pipeline: Pipeline | object) -> dict[str, float]:
    """Map transformed feature importances back to human-readable names."""
    estimator = unwrap_estimator(pipeline)
    classifier = estimator.named_steps["classifier"]
    if not hasattr(classifier, "feature_importances_"):
        return {}
    importances = np.asarray(classifier.feature_importances_, dtype=float)
    names = get_feature_names(estimator)
    if len(names) != len(importances):
        return {f"feature_{i}": float(v) for i, v in enumerate(importances)}
    ranked = sorted(zip(names, importances, strict=True), key=lambda item: item[1], reverse=True)
    return {name: round(float(value), 6) for name, value in ranked[:30]}
