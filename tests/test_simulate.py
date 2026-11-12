"""Synthetic drift-batch generation tests."""

from __future__ import annotations

import pandas as pd

from churnguard.constants import RAW_FEATURE_COLUMNS
from churnguard.monitoring.simulate import simulate_production_drift


def test_simulated_batch_is_persisted_and_shifted(settings, split_frames, tmp_path) -> None:
    _, test = split_frames
    source = tmp_path / "source.parquet"
    test.to_parquet(source, index=False)
    output = tmp_path / "current.parquet"

    sample = simulate_production_drift(source, output, rows=200, seed=4)
    assert output.exists()
    assert len(sample) == min(200, len(test))
    assert list(sample.columns) == list(test.columns)
    assert set(RAW_FEATURE_COLUMNS).issubset(sample.columns)

    month_to_month = (sample["Contract"] == "Month-to-month").mean()
    assert month_to_month > (test["Contract"] == "Month-to-month").mean()
    assert sample["tenure"].mean() < test["tenure"].mean()


def test_simulation_is_deterministic(settings, split_frames, tmp_path) -> None:
    _, test = split_frames
    source = tmp_path / "source.parquet"
    test.to_parquet(source, index=False)
    first = simulate_production_drift(source, tmp_path / "a.parquet", rows=100, seed=9)
    second = simulate_production_drift(source, tmp_path / "b.parquet", rows=100, seed=9)
    pd.testing.assert_frame_equal(first, second)
