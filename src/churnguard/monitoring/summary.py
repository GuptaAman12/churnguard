"""Drift summary persistence (kept dependency-free so the API can import it)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass
class FeatureDrift:
    feature: str
    drift_detected: bool
    drift_score: float | None = None
    method: str | None = None
    stat: float | None = None


@dataclass
class DriftSummary:
    generated_at: str
    reference_rows: int
    current_rows: int
    drift_share: float
    drifted_features: int
    total_features: int
    dataset_drift: bool
    threshold: float
    features: list[FeatureDrift] = field(default_factory=list)
    report_html: str | None = None
    reference_fingerprint: str | None = None
    current_fingerprint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DriftSummary:
        features = [
            FeatureDrift(
                feature=str(item.get("feature", "")),
                drift_detected=bool(item.get("drift_detected", False)),
                drift_score=_optional_float(item.get("drift_score")),
                method=item.get("method"),
                stat=_optional_float(item.get("stat")),
            )
            for item in payload.get("features", [])
        ]
        return cls(
            generated_at=str(payload.get("generated_at") or datetime.now(UTC).isoformat()),
            reference_rows=int(payload.get("reference_rows", 0)),
            current_rows=int(payload.get("current_rows", 0)),
            drift_share=float(payload.get("drift_share", 0.0)),
            drifted_features=int(payload.get("drifted_features", 0)),
            total_features=int(payload.get("total_features", 0)),
            dataset_drift=bool(payload.get("dataset_drift", False)),
            threshold=float(payload.get("threshold", 0.3)),
            features=features,
            report_html=payload.get("report_html"),
            reference_fingerprint=payload.get("reference_fingerprint"),
            current_fingerprint=payload.get("current_fingerprint"),
        )


def _optional_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def write_summary(summary: DriftSummary, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary.to_dict(), indent=2), encoding="utf-8")
    return path


def load_summary(path: Path | str) -> DriftSummary | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return DriftSummary.from_dict(payload)
    except (json.JSONDecodeError, OSError, ValueError):
        return None
