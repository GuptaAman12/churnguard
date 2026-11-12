"""Batch scoring job with an auditable summary artifact."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from churnguard.api.model_loader import ModelService
from churnguard.constants import RAW_FEATURE_COLUMNS
from churnguard.data import read_dataset, write_parquet
from churnguard.logging import get_logger
from churnguard.scoring import score_frame, score_probabilities

logger = get_logger(__name__)


@dataclass(frozen=True)
class BatchResult:
    rows: int
    output_path: str
    summary_path: str
    mean_probability: float
    predicted_churn_rate: float
    model_version: str | None
    threshold: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "output_path": self.output_path,
            "summary_path": self.summary_path,
            "mean_probability": self.mean_probability,
            "predicted_churn_rate": self.predicted_churn_rate,
            "model_version": self.model_version,
            "threshold": self.threshold,
        }


def score_batch(
    service: ModelService,
    input_path: Path | str,
    output_dir: Path | str = "reports/batch",
    *,
    threshold: float | None = None,
) -> BatchResult:
    """Score a batch file using the active model and persist results + summary."""
    descriptor = service.descriptor()
    chosen_threshold = float(threshold if threshold is not None else descriptor["threshold"])
    frame = read_dataset(input_path)
    missing = [column for column in RAW_FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Batch input is missing required columns: {missing}")

    scored = score_probabilities(
        frame[list(RAW_FEATURE_COLUMNS)],
        service.predict_proba(frame[list(RAW_FEATURE_COLUMNS)]),
        chosen_threshold,
    )
    # Preserve any passthrough columns (e.g. customerID) next to the scores.
    for column in frame.columns:
        if column not in scored.columns:
            scored[column] = frame[column].to_numpy()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "predictions.parquet"
    write_parquet(scored, output_path)

    mean_probability = round(float(scored["churn_probability"].mean()), 6)
    predicted_churn_rate = round(float(scored["churn_prediction"].mean()), 6)

    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "rows": len(scored),
        "mean_probability": mean_probability,
        "predicted_churn_rate": predicted_churn_rate,
        "risk_tier_counts": {
            str(key): int(value)
            for key, value in scored["risk_tier"].value_counts().sort_index().items()
        },
        "threshold": chosen_threshold,
        "model": descriptor,
        "input_path": str(input_path),
        "output_path": str(output_path),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info(
        "batch_scored",
        rows=len(scored),
        output=str(output_path),
        model_version=descriptor.get("version"),
    )
    return BatchResult(
        rows=len(scored),
        output_path=str(output_path),
        summary_path=str(summary_path),
        mean_probability=mean_probability,
        predicted_churn_rate=predicted_churn_rate,
        model_version=descriptor.get("version"),
        threshold=chosen_threshold,
    )


def score_dataframe_with_pipeline(
    pipeline: Any,
    frame: pd.DataFrame,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Pure helper for tests and notebooks."""
    return score_frame(pipeline, frame[list(RAW_FEATURE_COLUMNS)], threshold)
