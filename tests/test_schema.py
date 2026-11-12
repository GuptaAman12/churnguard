"""Tests for the data-quality contract and cleaning logic."""

from __future__ import annotations

import pandas as pd
import pytest

from churnguard.data import clean_data
from churnguard.schema import DataValidationError, validate_raw


def test_validate_raw_accepts_public_schema(raw_frame: pd.DataFrame) -> None:
    validated = validate_raw(raw_frame)
    assert len(validated) == len(raw_frame)


def test_validate_raw_rejects_unknown_category(raw_frame: pd.DataFrame) -> None:
    broken = raw_frame.copy()
    broken.loc[0, "gender"] = "Unknown"
    with pytest.raises(DataValidationError) as error:
        validate_raw(broken)
    assert error.value.failure_cases is not None
    assert not error.value.failure_cases.empty


def test_validate_raw_rejects_missing_column(raw_frame: pd.DataFrame) -> None:
    with pytest.raises(DataValidationError):
        validate_raw(raw_frame.drop(columns=["tenure"]))


def test_clean_data_transforms_target_and_charges(raw_frame: pd.DataFrame) -> None:
    cleaned = clean_data(raw_frame)
    assert "customerID" not in cleaned.columns
    assert cleaned["Churn"].dtype.kind in {"i", "u"}
    assert set(cleaned["Churn"].unique()).issubset({0, 1})
    assert pd.api.types.is_float_dtype(cleaned["TotalCharges"])
    assert cleaned["TotalCharges"].isna().sum() == 0


def test_clean_data_handles_blank_total_charges(raw_frame: pd.DataFrame) -> None:
    frame = raw_frame.copy()
    frame.loc[0, "tenure"] = 0
    frame.loc[0, "TotalCharges"] = " "
    cleaned = clean_data(frame)
    assert cleaned.loc[0, "TotalCharges"] == 0.0
