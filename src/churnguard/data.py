"""Data ingestion, cleaning, splitting and lineage helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from churnguard.constants import DATASET_COLUMNS, ID_COLUMN, TARGET_COLUMN
from churnguard.schema import validate_processed, validate_raw

DATASET_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)


@dataclass(frozen=True)
class DatasetManifest:
    """Reproducibility metadata stored next to every dataset artifact."""

    name: str
    rows: int
    columns: list[str]
    fingerprint: str
    created_at: str
    source: str

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "rows": self.rows,
            "columns": self.columns,
            "fingerprint": self.fingerprint,
            "created_at": self.created_at,
            "source": self.source,
        }


def read_dataset(path: Path | str) -> pd.DataFrame:
    """Read CSV or Parquet based on the file suffix."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the bronze -> silver transformations and validate the contract."""
    validate_raw(df)
    cleaned = df.copy()

    # TotalCharges is blank for brand-new customers with zero tenure.
    cleaned["TotalCharges"] = (
        pd.to_numeric(cleaned["TotalCharges"], errors="coerce").fillna(0.0).astype(float)
    )
    cleaned["tenure"] = cleaned["tenure"].astype(int)
    cleaned["SeniorCitizen"] = cleaned["SeniorCitizen"].astype(int)
    cleaned["MonthlyCharges"] = cleaned["MonthlyCharges"].astype(float)

    if TARGET_COLUMN in cleaned.columns:
        cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].map({"Yes": 1, "No": 0}).astype("int8")

    cleaned = cleaned.drop(columns=[ID_COLUMN], errors="ignore")
    cleaned = cleaned.loc[:, list(DATASET_COLUMNS[1:])]
    return validate_processed(cleaned.reset_index(drop=True))


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split on the churn target."""
    if TARGET_COLUMN not in df.columns:
        raise ValueError("Cannot split a dataset without the target column")
    train, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[TARGET_COLUMN],
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def fingerprint(df: pd.DataFrame) -> str:
    """Stable content hash used for data lineage (order independent)."""
    row_hashes = pd.util.hash_pandas_object(df, index=False).to_numpy()
    return hashlib.sha256(row_hashes.tobytes()).hexdigest()[:16]


def build_manifest(df: pd.DataFrame, name: str, source: str) -> DatasetManifest:
    return DatasetManifest(
        name=name,
        rows=len(df),
        columns=[str(c) for c in df.columns],
        fingerprint=fingerprint(df),
        created_at=datetime.now(UTC).isoformat(),
        source=source,
    )


def write_parquet(df: pd.DataFrame, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, compression="zstd")


def write_manifest(manifest: DatasetManifest, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")


def prepare_datasets(
    raw_path: Path,
    train_path: Path,
    test_path: Path,
    reference_path: Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, object]:
    """Clean, split, persist datasets and write manifests. Used by DVC + Prefect."""
    raw = read_dataset(raw_path)
    cleaned = clean_data(raw)
    train, test = split_data(cleaned, test_size=test_size, random_state=random_state)

    write_parquet(train, train_path)
    write_parquet(test, test_path)
    write_parquet(train, reference_path)

    train_manifest = build_manifest(train, "train", str(raw_path))
    test_manifest = build_manifest(test, "test", str(raw_path))
    write_manifest(train_manifest, Path(train_path).with_suffix(".manifest.json"))
    write_manifest(test_manifest, Path(test_path).with_suffix(".manifest.json"))

    return {
        "train_rows": len(train),
        "test_rows": len(test),
        "churn_rate": round(float(train[TARGET_COLUMN].mean()), 4),
        "train_fingerprint": train_manifest.fingerprint,
        "test_fingerprint": test_manifest.fingerprint,
    }
