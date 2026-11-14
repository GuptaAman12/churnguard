# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1] - 2026-11-14

### Fixed

- `churnguard --version` reports the installed package version instead of a
  hardcoded `0.1.0` string.

## [0.2.0] - 2026-11-14

Calibrated, explainable scoring.

### Added

- Out-of-fold isotonic calibration: a `CalibratedClassifier` artifact is fitted on
  `cross_val_predict` scores from the training split and kept only when it
  improves test log loss, so the test set never influences the curve.
- Calibration diagnostics logged with every run: log loss, Brier score,
  expected calibration error (10 bins) and a reliability curve
  (`evaluation/calibration.json`).
- `POST /explain` returns exact TreeSHAP contributions from the XGBoost booster,
  ranked by absolute impact, plus the raw (uncalibrated) and served
  (calibrated) probabilities.
- Model card now documents the calibration method, raw-vs-served metrics and an
  explanation section.
- Registry lineage tag `calibration_method` (`isotonic` or `identity`).
- Ten new tests covering calibration math, artifact picklability, the promotion
  guard, SHAP ranking, contribution/margin consistency and non-tree rejection
  (88 total).

### Changed

- Decision threshold is now tuned on calibrated probabilities (0.27 for the
  deployed champion, down from 0.58); risk tiers remain threshold-relative.
- `churnguard.modeling` no longer imports training dependencies eagerly, so the
  slim API image can load pickled artifacts without Optuna, Prefect or DVC.

### Fixed

- MLflow accepts `127.0.0.1:5000` in `--allowed-hosts`, fixing `403 Invalid Host
  header` errors from host-side clients.
- Explanation endpoint reports the positive-class probability for both the raw
  pipeline and the calibrated wrapper.

### Results

| Metric | Raw | Served |
|---|---|---|
| Log loss | 0.4808 | **0.4157** |
| Brier score | 0.1615 | **0.1354** |
| ECE (10 bins) | 0.1404 | **0.0174** |

ROC-AUC stayed flat at 0.8463 — calibration improved probability quality without
sacrificing ranking.

## [0.1.0] - 2026-11-12

Initial production-grade platform.

### Added

- Reproducible data pipeline (DVC) with Pandera contracts, SHA-256 dataset
  fingerprints and per-artifact manifests.
- Optuna TPE tuning and an XGBoost pipeline with embedded feature engineering,
  class weighting, tuned threshold and subgroup metrics.
- MLflow tracking, model registry and `champion` alias with gated promotion
  (absolute AUC floor plus a no-regression tolerance) and lineage tags.
- FastAPI service with typed contracts, single and batch inference, API-key auth,
  structured JSON logs, Prometheus metrics and hot-reload from the registry with a
  local joblib fallback.
- Evidently drift detection with a KS/chi-square fallback, Prometheus gauges,
  Grafana dashboard and alert rules.
- Prefect flows for nightly training and six-hourly drift checks.
- Multi-target Docker images (API / pipeline / MLflow), a full Compose stack
  (PostgreSQL, MLflow, API, Prefect, Prometheus, Pushgateway, Grafana), Helm
  chart and Terraform stack for ECS Fargate, RDS, S3 and ALB.
- CI (lint, strict typing, tests with coverage, `pip-audit`, Trivy, image builds,
  infra validation), release publishing to GHCR and opt-in ECS deployment.
- Architecture notes, ADRs, runbook and a generated model card.

[0.2.1]: https://github.com/GuptaAman12/churnguard/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/GuptaAman12/churnguard/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/GuptaAman12/churnguard/releases/tag/v0.1.0
