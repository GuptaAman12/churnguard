"""Pandera schemas for data-quality gates (raw + processed data)."""

from __future__ import annotations

from typing import Any

import pandas as pd

try:  # pandera >= 0.20 (new namespace)
    import pandera.pandas as pa
    from pandera.pandas import Check, Column, DataFrameSchema
except ImportError:  # pragma: no cover - older pandera
    import pandera as pa  # type: ignore[no-redef]
    from pandera import Check, Column, DataFrameSchema

from churnguard.constants import ALLOWED_VALUES, DATASET_COLUMNS, TARGET_COLUMN


class DataValidationError(ValueError):
    """Raised when a dataframe does not satisfy the data contract."""

    def __init__(self, message: str, failure_cases: pd.DataFrame | None = None) -> None:
        super().__init__(message)
        self.failure_cases = failure_cases


def _categorical_checks() -> dict[str, Any]:
    return {
        column: Column(
            str,
            checks=Check.isin(list(values)),
            nullable=False,
            required=True,
        )
        for column, values in ALLOWED_VALUES.items()
    }


RAW_SCHEMA = DataFrameSchema(
    {
        "customerID": Column(str, nullable=False, unique=True),
        "SeniorCitizen": Column(int, Check.isin([0, 1])),
        "tenure": Column(int, Check.in_range(0, 100)),
        "MonthlyCharges": Column(float, Check.greater_than(0)),
        # Raw dataset contains blanks here (" "), coerced later during cleaning.
        "TotalCharges": Column(object, nullable=True),
        "Churn": Column(str, Check.isin(["Yes", "No"]), nullable=True),
        **_categorical_checks(),
    },
    strict=False,
    coerce=True,
    name="telco_churn_raw",
)

PROCESSED_SCHEMA = DataFrameSchema(
    {
        "SeniorCitizen": Column(int, Check.isin([0, 1])),
        "tenure": Column(int, Check.in_range(0, 100)),
        "MonthlyCharges": Column(float, Check.greater_than(0)),
        "TotalCharges": Column(float, Check.greater_than_or_equal_to(0)),
        "Churn": Column(int, Check.isin([0, 1])),
        **_categorical_checks(),
    },
    strict=False,
    coerce=True,
    name="telco_churn_processed",
)

INFERENCE_SCHEMA = DataFrameSchema(
    {
        "SeniorCitizen": Column(int, Check.isin([0, 1])),
        "tenure": Column(int, Check.in_range(0, 100)),
        "MonthlyCharges": Column(float, Check.greater_than(0)),
        "TotalCharges": Column(float, Check.greater_than_or_equal_to(0)),
        **_categorical_checks(),
    },
    strict=False,
    coerce=True,
    name="telco_churn_inference",
)


def validate_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the raw CSV contract, returning the validated frame."""
    try:
        return RAW_SCHEMA.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        raise DataValidationError(
            f"Raw data failed validation with {len(exc.failure_cases)} issue(s)",
            failure_cases=exc.failure_cases,
        ) from exc


def validate_processed(df: pd.DataFrame) -> pd.DataFrame:
    try:
        return PROCESSED_SCHEMA.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        raise DataValidationError(
            f"Processed data failed validation with {len(exc.failure_cases)} issue(s)",
            failure_cases=exc.failure_cases,
        ) from exc


def validate_inference(df: pd.DataFrame) -> pd.DataFrame:
    try:
        return INFERENCE_SCHEMA.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        raise DataValidationError(
            f"Inference payload failed validation with {len(exc.failure_cases)} issue(s)",
            failure_cases=exc.failure_cases,
        ) from exc


__all__ = [
    "DATASET_COLUMNS",
    "INFERENCE_SCHEMA",
    "PROCESSED_SCHEMA",
    "RAW_SCHEMA",
    "TARGET_COLUMN",
    "DataValidationError",
    "validate_inference",
    "validate_processed",
    "validate_raw",
]
