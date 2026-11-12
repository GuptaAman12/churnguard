"""Data-drift detection with Evidently, Prometheus export and HTML/JSON artifacts."""

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

_COLUMN_PATTERN = r"column=([^,)]+)"


def _sample(frame: pd.DataFrame, size: int, seed: int = 42) -> pd.DataFrame:
    if len(frame) <= size:
        return frame.reset_index(drop=True)
    return frame.sample(n=size, random_state=seed).reset_index(drop=True)


def _to_dict(snapshot: Any) -> dict[str, Any]:
    for attribute in ("dict", "as_dict"):
        method = getattr(snapshot, attribute, None)
        if callable(method):
            payload = method()
            if isinstance(payload, dict):
                return payload
    raise RuntimeError("Unable to serialize Evidently snapshot")


def _run_evidently(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    output_html: Path,
) -> dict[str, Any] | None:
    """Run the Evidently data-drift preset. Returns None if unavailable."""
    try:
        try:
            from evidently import Report
            from evidently.presets import DataDriftPreset
        except ImportError:  # pragma: no cover - Evidently < 0.7
            from evidently.metric_preset import DataDriftPreset
            from evidently.report import Report

        report = Report(metrics=[DataDriftPreset()])
        snapshot = report.run(reference_data=reference, current_data=current)
        output_html.parent.mkdir(parents=True, exist_ok=True)
        save_html = getattr(snapshot, "save_html", None) or getattr(report, "save_html", None)
        if callable(save_html):
            save_html(str(output_html))
        return _to_dict(snapshot)
    except Exception as exc:  # pragma: no cover - optional dependency/runtime issue
        logger.warning("evidently_report_failed", error=str(exc))
        return None


def _parse_evidently(
    payload: dict[str, Any],
    columns: list[str],
) -> tuple[list[FeatureDrift], int, int]:
    """Parse Evidently 0.7+ (``metric_name``/p-value) and legacy 0.6 snapshots."""
    per_column: dict[str, FeatureDrift] = {}
    drifted_count: int | None = None
    total_count: int | None = None

    for metric in payload.get("metrics", []):
        metric_id = str(
            metric.get("metric_name") or metric.get("metric_id") or metric.get("metric") or ""
        )
        config = metric.get("config") if isinstance(metric.get("config"), dict) else {}
        value = metric.get("value", metric.get("result"))

        if "DriftedColumnsCount" in metric_id and isinstance(value, dict):
            drifted_count = int(value.get("count", 0))
            total_count = int(value.get("total", len(columns)))
            continue

        if metric_id == "DatasetDriftMetric" and isinstance(value, dict):
            drifted_count = int(value.get("number_of_drifted_columns", 0))
            total_count = int(value.get("number_of_columns", len(columns)))
            continue

        if "ValueDrift" in metric_id or metric_id == "ColumnDriftMetric":
            column_name = str(config.get("column") or "")
            threshold = _maybe_float(config.get("threshold")) or 0.05
            method = config.get("method")
            if isinstance(value, dict):
                column_name = column_name or str(
                    value.get("column_name") or value.get("column") or ""
                )
                score = _maybe_float(value.get("drift_score", value.get("p_value")))
                detected = bool(
                    value.get("drift_detected", score is not None and score < threshold)
                )
                method = value.get("stattest_name") or value.get("method") or method
            else:
                # Evidently 0.7 ValueDrift emits a p-value (drift if below threshold)
                # or a distance (drift if above threshold), depending on the method.
                score = _maybe_float(value)
                method_text = str(method or "").lower()
                is_p_value = (
                    "p_value" in method_text or "p-value" in method_text or "p value" in method_text
                )
                if score is None:
                    detected = False
                elif is_p_value:
                    detected = score < threshold
                else:
                    detected = score > threshold
                column_name = column_name or _extract_column(metric)
            if column_name:
                per_column[column_name] = FeatureDrift(
                    feature=column_name,
                    drift_detected=detected,
                    drift_score=score,
                    method=str(method) if method else None,
                )

    features = [per_column[column] for column in columns if column in per_column]
    features.extend(item for key, item in per_column.items() if key not in columns)

    drifted = (
        drifted_count
        if drifted_count is not None
        else sum(item.drift_detected for item in features)
    )
    total = total_count or len(features) or len(columns)
    return features, drifted, total


def _extract_column(metric: dict[str, Any]) -> str:
    config = metric.get("config")
    if isinstance(config, dict) and config.get("column"):
        return str(config["column"])
    import re

    match = re.search(_COLUMN_PATTERN, str(metric.get("metric_id", "")))
    if match:
        return match.group(1)
    for key in ("column_name", "column"):
        if metric.get(key):
            return str(metric[key])
    return ""


def _maybe_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def simple_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    columns: list[str],
    significance: float = 0.05,
) -> tuple[list[FeatureDrift], int, int]:
    """Dependency-light fallback: KS test for numeric, chi-square for categorical."""
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


def _write_html_fallback(summary: DriftSummary, path: Path) -> None:
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
    payload = _run_evidently(reference_sample, current_sample, html_path)

    if payload is not None:
        features, drifted, total = _parse_evidently(payload, columns)
        if not features:
            features, drifted, total = simple_drift(reference_sample, current_sample, columns)
    else:
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
    if not html_path.exists():
        _write_html_fallback(summary, html_path)

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
