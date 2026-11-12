# ADR 0003 — One lockfile, per-target dependency groups

- **Status:** Accepted
- **Date:** 2026-11-12

## Context

A single image that can train, serve and run MLflow needs every library:
XGBoost, Optuna, Evidently, Prefect, DVC, FastAPI, psycopg2, boto3. That image
was ~7 GB because changing file ownership with `chown -R /app` duplicated every
layer copy-on-write.

Serving only needs FastAPI, scikit-learn/XGBoost, the MLflow client, Pandera and
PyArrow. Training needs Optuna/Evidently/Prefect/DVC. The MLflow server needs
psycopg2 and boto3 but not XGBoost.

## Decision

- Keep a **single `uv.lock`** for reproducibility.
- Put the serving core in `[project.dependencies]`.
- Put training/orchestration libraries in the `pipeline` dependency group and
  MLflow-server libraries in the `mlflow` group.
- Build three targets that share a cached core layer:
  - `api`: `uv sync --frozen --no-dev`
  - `pipeline`: `uv sync --frozen --no-dev --group pipeline`
  - `mlflow`: `uv sync --frozen --no-dev --group mlflow`
- Create the runtime user **before** copying source, use `COPY --chown`, and
  only chown the small writable directories instead of the whole app.

## Consequences

- The API image ships without Prefect, DVC, Evidently, Optuna or psycopg2.
- All targets still resolve from the same lockfile, so versions cannot drift.
- A dependency used by two targets must be declared in the shared core (or in
  both groups); this is enforced by CI building every target.
- FastAPI is pinned `<0.151` because Prefect 3.9's server still passes the
  deprecated `min_items` argument to `Body`; CI catches the conflict at build
  time rather than at runtime.
