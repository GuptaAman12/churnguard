"""Data-drift detection with statistical tests, Prometheus export and HTML/JSON artifacts."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from scipy import stats

from churnguard.config import Settings, get_settings
from churnguard.constants import RAW_FEATURE_COLUMNS
from churnguard.data import fingerprint, read_dataset
from churnguard.logging import get_logger
from churnguard.monitoring.summary import DriftSummary, FeatureDrift, write_summary

logger = get_logger(__name__)


def _sample(frame: pd.DataFrame, size: int, seed: int = 42) -> pd.DataFrame:
    if len(frame) <= size:
        return frame.reset_index(drop=True)
    return frame.sample(n=size, random_state=seed).reset_index(drop=True)


def simple_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    columns: list[str],
    significance: float = 0.05,
) -> tuple[list[FeatureDrift], int, int]:
    """KS test for numeric features, chi-square for categorical features."""
    results: list[FeatureDrift] = []
    for column in columns:
        if column not in reference.columns or column not in current.columns:
            continue
        ref = reference[column].dropna()
        cur = current[column].dropna()
        if ref.empty or cur.empty:
            continue
        try:
            if pd.api.types.is_numeric_dtype(ref) and pd.api.types.is_numeric_dtype(cur):
                stat, p_value = stats.ks_2samp(ref.astype(float), cur.astype(float))
                method = "ks_2samp"
            else:
                contingency = pd.crosstab(
                    pd.concat([ref, cur], ignore_index=True),
                    pd.Series(
                        ["reference"] * len(ref) + ["current"] * len(cur),
                        name="dataset",
                    ),
                )
                if contingency.shape[1] < 2 or contingency.shape[0] < 2:
                    continue
                stat, p_value, _, _ = stats.chi2_contingency(contingency)
                method = "chi2_contingency"
        except ValueError:  # pragma: no cover - degenerate groups
            continue
        results.append(
            FeatureDrift(
                feature=column,
                drift_detected=bool(p_value < significance),
                drift_score=round(float(p_value), 6),
                method=method,
                stat=round(float(stat), 6),
            )
        )
    drifted = sum(item.drift_detected for item in results)
    return results, drifted, len(results) or len(columns)


def _write_html_report(summary: DriftSummary, path: Path) -> None:
    rows = "\n".join(
        f"<tr><td>{item.feature}</td><td>{'⚠️ yes' if item.drift_detected else 'no'}</td>"
        f"<td>{item.drift_score if item.drift_score is not None else ''}</td>"
        f"<td>{item.method or ''}</td></tr>"
        for item in summary.features
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""<!doctype html><html><head><meta charset="utf-8">
<title>ChurnGuard drift report</title>
<style>body{{font-family:system-ui;margin:2rem}}table{{border-collapse:collapse}}
td,th{{border:1px solid #ccc;padding:.4rem .8rem}}</style></head><body>
<h1>ChurnGuard data-drift report</h1>
<p>Generated {summary.generated_at} — drift share
<strong>{summary.drift_share:.2%}</strong> ({summary.drifted_features}/{summary.total_features} features)</p>
<table><thead><tr><th>Feature</th><th>Drifted</th><th>Score</th><th>Method</th></tr></thead>
<tbody>{rows}</tbody></table></body></html>""",
        encoding="utf-8",
    )


def _push_to_gateway(summary: DriftSummary, gateway_url: str) -> None:
    try:
        from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

        registry = CollectorRegistry()
        Gauge(
            "churnguard_drift_share",
            "Share of drifted features",
            registry=registry,
        ).set(summary.drift_share)
        Gauge(
            "churnguard_drifted_features",
            "Number of drifted features",
            registry=registry,
        ).set(summary.drifted_features)
        push_to_gateway(gateway_url, job="churnguard-drift", registry=registry)
        logger.info("drift_pushed_to_gateway", url=gateway_url)
    except Exception as exc:  # pragma: no cover - gateway optional
        logger.warning("drift_push_failed", error=str(exc))


def run_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    settings: Settings | None = None,
    *,
    output_dir: Path | str | None = None,
    push: bool = True,
) -> DriftSummary:
    """Compare reference vs current data and persist summary + HTML artifacts."""
    settings = settings or get_settings()
    columns = [column for column in RAW_FEATURE_COLUMNS if column in reference.columns]
    reference_sample = _sample(reference[columns], settings.drift_sample_size)
    current_sample = _sample(current[columns], settings.drift_sample_size)

    output_dir = Path(output_dir or settings.reports_dir / "drift")
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / "latest.html"
    features, drifted, total = simple_drift(reference_sample, current_sample, columns)

    drift_share = round(drifted / max(total, 1), 6)
    summary = DriftSummary(
        generated_at=datetime.now(UTC).isoformat(),
        reference_rows=len(reference_sample),
        current_rows=len(current_sample),
        drift_share=drift_share,
        drifted_features=drifted,
        total_features=total,
        dataset_drift=drift_share >= settings.drift_share_threshold,
        threshold=settings.drift_share_threshold,
        features=features,
        report_html=str(html_path),
        reference_fingerprint=fingerprint(reference_sample),
        current_fingerprint=fingerprint(current_sample),
    )
    _write_html_report(summary, html_path)

    write_summary(summary, settings.drift_summary_path)
    history_dir = output_dir / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    write_summary(summary, history_dir / f"drift-{stamp}.json")

    if push and settings.pushgateway_url:
        _push_to_gateway(summary, settings.pushgateway_url)

    logger.info(
        "drift_report_complete",
        drift_share=drift_share,
        drifted=drifted,
        total=total,
        dataset_drift=summary.dataset_drift,
        html=str(html_path),
    )
    return summary


def run_drift_from_paths(
    settings: Settings | None = None,
    *,
    reference_path: Path | str | None = None,
    current_path: Path | str | None = None,
    push: bool = True,
) -> DriftSummary:
    settings = settings or get_settings()
    reference = read_dataset(reference_path or settings.reference_path)
    current = read_dataset(current_path or settings.current_data_path)
    return run_drift(reference, current, settings, push=push)


def load_prediction_features(path: Path | str, limit: int = 5000) -> pd.DataFrame:
    """Read the API prediction log (JSONL) into a feature dataframe."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Prediction log not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"Prediction log is empty: {path}")
    frame = pd.DataFrame([row.get("features", {}) for row in rows[-limit:]])
    return frame
