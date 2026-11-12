"""Generate a drifted 'current' batch to exercise monitoring locally."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from churnguard.data import read_dataset, write_parquet
from churnguard.logging import get_logger

logger = get_logger(__name__)


def simulate_production_drift(
    source_path: Path | str,
    output_path: Path | str,
    *,
    rows: int = 2000,
    seed: int = 7,
) -> pd.DataFrame:
    """Create a statistically shifted batch (price rise, tenure drop, contract mix)."""
    source = read_dataset(source_path)
    sample = source.sample(n=min(rows, len(source)), random_state=seed).copy()
    rng = np.random.default_rng(seed)

    fiber = sample["InternetService"] == "Fiber optic"
    sample.loc[fiber, "MonthlyCharges"] = (sample.loc[fiber, "MonthlyCharges"] * 1.35).round(2)
    sample["tenure"] = (sample["tenure"] * 0.7).astype(int)

    non_monthly = sample["Contract"] != "Month-to-month"
    move = non_monthly & (rng.random(len(sample)) < 0.35)
    sample.loc[move, "Contract"] = "Month-to-month"
    sample.loc[move, "PaperlessBilling"] = "Yes"

    # Product-mix shift: a marketing campaign pushes fibre and streaming,
    # while support add-ons are de-prioritised.
    fiber_adoption = (rng.random(len(sample)) < 0.25) & ~fiber
    sample.loc[fiber_adoption, "InternetService"] = "Fiber optic"
    for column in ("TechSupport", "OnlineSecurity", "OnlineBackup"):
        churn_addon = rng.random(len(sample)) < 0.40
        sample.loc[churn_addon, column] = "No"
    for column in ("StreamingTV", "StreamingMovies"):
        uptake = rng.random(len(sample)) < 0.35
        sample.loc[uptake, column] = "Yes"

    payment_shift = rng.random(len(sample)) < 0.30
    sample.loc[payment_shift, "PaymentMethod"] = "Electronic check"

    write_parquet(sample.reset_index(drop=True), output_path)
    logger.info(
        "drift_batch_simulated",
        output=str(output_path),
        rows=len(sample),
        fiber_price_increase="35%",
        tenure_shift="0.7x",
        contract_shift="35% -> month-to-month",
        product_mix_shift="fibre/streaming up, support add-ons down",
    )
    return sample
