"""Deterministic synthetic data generator (offline fallback for the real dataset)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from churnguard.constants import ALLOWED_VALUES, DATASET_COLUMNS


def generate_churn_dataset(rows: int = 7043, seed: int = 42) -> pd.DataFrame:
    """Generate a schema-compatible churn dataset for offline development and CI."""
    rng = np.random.default_rng(seed)
    gender = rng.choice(ALLOWED_VALUES["gender"], size=rows)
    senior = rng.binomial(1, 0.16, size=rows)
    partner = rng.choice(["Yes", "No"], size=rows, p=[0.48, 0.52])
    dependents = rng.choice(["Yes", "No"], size=rows, p=[0.3, 0.7])
    tenure = rng.integers(0, 73, size=rows)
    phone = rng.choice(["Yes", "No"], size=rows, p=[0.9, 0.1])
    multiple_lines = np.where(
        phone == "No",
        "No phone service",
        rng.choice(["Yes", "No"], size=rows, p=[0.42, 0.58]),
    )
    internet = rng.choice(["DSL", "Fiber optic", "No"], size=rows, p=[0.34, 0.44, 0.22])
    addons = {}
    for column in (
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ):
        addons[column] = np.where(
            internet == "No",
            "No internet service",
            rng.choice(["Yes", "No"], size=rows, p=[0.4, 0.6]),
        )
    contract = rng.choice(ALLOWED_VALUES["Contract"], size=rows, p=[0.55, 0.21, 0.24])
    paperless = rng.choice(["Yes", "No"], size=rows, p=[0.59, 0.41])
    payment = rng.choice(ALLOWED_VALUES["PaymentMethod"], size=rows)
    monthly = np.clip(rng.normal(64, 30, size=rows) + (internet == "Fiber optic") * 20, 18, 140)
    total = np.clip(monthly * np.maximum(tenure, 1) * rng.uniform(0.85, 1.05, rows), 0, 9000)
    # Brand-new customers have blank TotalCharges, matching the public dataset.
    total_series = pd.Series(total.round(2)).mask(tenure == 0, " ")

    logit = (
        -1.1
        + 0.9 * (contract == "Month-to-month")
        + 0.6 * (internet == "Fiber optic")
        - 0.05 * tenure
        + 0.012 * (monthly - 64)
        + 0.45 * (addons["TechSupport"] == "No")
        + 0.3 * senior
        - 0.25 * (partner == "Yes")
    )
    probability = 1 / (1 + np.exp(-logit))
    churn = rng.binomial(1, probability)

    frame = pd.DataFrame(
        {
            "customerID": [f"SYN-{i:05d}-{rng.integers(1000, 9999)}" for i in range(rows)],
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple_lines,
            "InternetService": internet,
            **addons,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly.round(2),
            "TotalCharges": total_series,
            "Churn": np.where(churn == 1, "Yes", "No"),
        }
    )
    return frame.loc[:, list(DATASET_COLUMNS)]
