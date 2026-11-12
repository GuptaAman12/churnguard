"""Dataset splitting, lineage and persistence tests."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from churnguard.data import (
    build_manifest,
    clean_data,
    fingerprint,
    prepare_datasets,
    read_dataset,
    split_data,
    write_parquet,
)
from churnguard.synthetic import generate_churn_dataset


def test_split_is_stratified(cleaned_frame: pd.DataFrame) -> None:
    train, test = split_data(cleaned_frame, test_size=0.25, random_state=3)
    assert len(train) + len(test) == len(cleaned_frame)
    assert abs(train["Churn"].mean() - test["Churn"].mean()) < 0.05


def test_split_requires_target(cleaned_frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        split_data(cleaned_frame.drop(columns=["Churn"]))


def test_fingerprint_is_stable_and_content_sensitive(cleaned_frame: pd.DataFrame) -> None:
    first = fingerprint(cleaned_frame)
    second = fingerprint(cleaned_frame.copy())
    assert first == second

    modified = cleaned_frame.copy()
    modified.loc[0, "tenure"] = int(modified.loc[0, "tenure"]) + 1
    assert fingerprint(modified) != first


def test_manifest_round_trip(cleaned_frame: pd.DataFrame) -> None:
    manifest = build_manifest(cleaned_frame, "train", "unit-test")
    payload = manifest.to_dict()
    assert payload["rows"] == len(cleaned_frame)
    assert payload["name"] == "train"
    assert len(payload["fingerprint"]) == 16


def test_prepare_datasets_writes_artifacts(tmp_path) -> None:
    raw = generate_churn_dataset(rows=400, seed=5)
    raw_path = tmp_path / "raw.csv"
    raw.to_csv(raw_path, index=False)
    train_path = tmp_path / "processed" / "train.parquet"
    test_path = tmp_path / "processed" / "test.parquet"
    reference_path = tmp_path / "reference" / "reference.parquet"

    result = prepare_datasets(
        raw_path=raw_path,
        train_path=train_path,
        test_path=test_path,
        reference_path=reference_path,
        test_size=0.2,
        random_state=1,
    )
    assert result["train_rows"] > 0
    assert train_path.exists() and test_path.exists() and reference_path.exists()
    manifest_path = train_path.with_suffix(".manifest.json")
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["rows"] == result["train_rows"]


def test_parquet_round_trip(cleaned_frame: pd.DataFrame, tmp_path) -> None:
    path = tmp_path / "frame.parquet"
    write_parquet(cleaned_frame, path)
    loaded = read_dataset(path)
    pd.testing.assert_frame_equal(loaded, cleaned_frame)


def test_read_dataset_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        read_dataset(tmp_path / "nope.csv")


def test_clean_data_is_idempotent(raw_frame: pd.DataFrame) -> None:
    once = clean_data(raw_frame)
    twice = clean_data(raw_frame)
    pd.testing.assert_frame_equal(once, twice)
