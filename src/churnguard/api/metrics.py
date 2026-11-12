"""Prometheus metric definitions and helpers."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

HTTP_REQUESTS = Counter(
    "churnguard_http_requests_total",
    "HTTP requests processed by the prediction API.",
    ["method", "endpoint", "status"],
)

HTTP_LATENCY = Histogram(
    "churnguard_http_request_duration_seconds",
    "HTTP request latency in seconds.",
    ["endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

PREDICTIONS = Counter(
    "churnguard_predictions_total",
    "Predictions grouped by outcome and risk tier.",
    ["prediction", "risk_tier"],
)

PREDICTION_PROBABILITY = Histogram(
    "churnguard_prediction_probability",
    "Distribution of predicted churn probabilities.",
    buckets=(0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95),
)

MODEL_INFO = Gauge(
    "churnguard_model_info",
    "Active model metadata (always 1).",
    ["name", "version", "source", "alias"],
)

MODEL_LOADED = Gauge(
    "churnguard_model_loaded",
    "1 when a model is loaded and serving, 0 otherwise.",
)

MODEL_LOAD_TIMESTAMP = Gauge(
    "churnguard_model_loaded_timestamp_seconds",
    "Unix timestamp of the last successful model load.",
)

DRIFT_SHARE = Gauge(
    "churnguard_drift_share",
    "Share of monitored features flagged as drifted in the latest report.",
)

DRIFTED_FEATURES = Gauge(
    "churnguard_drifted_features",
    "Number of features flagged as drifted in the latest report.",
)

DRIFT_TIMESTAMP = Gauge(
    "churnguard_drift_report_timestamp_seconds",
    "Unix timestamp of the latest drift report.",
)


def record_model_loaded(
    *,
    name: str,
    version: str | None,
    source: str,
    alias: str | None,
    loaded_at_epoch: float,
) -> None:
    MODEL_INFO.clear()
    MODEL_INFO.labels(
        name=name,
        version=version or "unknown",
        source=source,
        alias=alias or "none",
    ).set(1)
    MODEL_LOADED.set(1)
    MODEL_LOAD_TIMESTAMP.set(loaded_at_epoch)


def record_model_unloaded() -> None:
    MODEL_INFO.clear()
    MODEL_LOADED.set(0)


def record_drift(
    *,
    drift_share: float,
    drifted_features: int,
    generated_at_epoch: float,
) -> None:
    DRIFT_SHARE.set(drift_share)
    DRIFTED_FEATURES.set(drifted_features)
    DRIFT_TIMESTAMP.set(generated_at_epoch)
