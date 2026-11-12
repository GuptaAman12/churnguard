# Operations runbook

## Service map

| Service | Port | Purpose | Health probe |
|---|---|---|---|
| `api` | 8000 | FastAPI prediction service | `GET /health`, `GET /ready` |
| `mlflow` | 5000 | Tracking server + registry UI | `GET /health` |
| `prefect-server` | 4200 | Orchestration UI/API (PostgreSQL-backed) | `GET /api/health` |
| `prometheus` | 9090 | Metrics + alert rules | `/-/healthy` |
| `pushgateway` | 9091 | Batch drift metric ingestion | `/-/healthy` |
| `grafana` | 3000 | Dashboards | `/api/health` |
| `postgres` | 5432 | MLflow backend store | `pg_isready` |

## First-time bring-up

```bash
cp .env.example .env          # set CHURNGUARD_API_KEY if you want auth
docker compose build          # ~10-20 min first time
docker compose up -d postgres mlflow
docker compose run --rm trainer   # ingest, split, train, register champion
docker compose up -d              # API, Prefect, Prometheus, Grafana
```

> Prefect stores its metadata in a dedicated `prefect` database on the same
> PostgreSQL instance. `configs/postgres/init.sql` creates it on first
> initialisation; for a pre-existing volume run:
> `docker compose exec postgres psql -U mlflow -c "CREATE DATABASE prefect"`.

Verify:

```bash
curl -s localhost:8000/health
curl -s localhost:8000/ready
curl -s localhost:8000/model/info
curl -s localhost:8000/metrics | grep churnguard_
open http://localhost:5000     # MLflow
open http://localhost:3000     # Grafana (admin/admin)
open http://localhost:4200     # Prefect
```

## Scenario: `/ready` returns 503

1. `curl localhost:8000/model/info` → expect 503 with a message.
2. Check API logs: `docker compose logs api | grep model_`.
3. Verify MLflow: `curl localhost:5000/health`, then open the registry UI and
   confirm the `champion` alias points at a ready version.
4. If the registry is down but a fallback exists, the API logs
   `registry_load_failed` and serves `source=fallback` — `/ready` will be 200.
5. Force a reload after fixing the registry: `POST /admin/reload` (needs API key
   when configured).

## Scenario: drift alert (`ChurnGuardHighDriftShare`)

1. Read `GET /monitoring/drift` for the per-feature breakdown.
2. Open `reports/drift/latest.html` for distributions and p-values.
3. Interpret:
   - Numeric shift → check upstream pricing / tenure calculation.
   - Categorical mix change → check campaign or plan changes.
4. If the new distribution is legitimate, retrain so the reference advances:
   `docker compose run --rm trainer` (or trigger the Prefect `daily-training`
   deployment). Promotion only happens if AUC gates pass.
5. If the shift is a data bug, fix upstream and regenerate; **do not** retrain on
   corrupted data.

## Scenario: roll back a bad champion

```bash
# List versions and their AUC
uv run python - <<'PY'
from mlflow import MlflowClient
c = MlflowClient("http://localhost:5000")
for v in c.search_model_versions("name='churnguard-churn-classifier'"):
    run = c.get_run(v.run_id) if v.run_id else None
    print(v.version, v.current_stage, v.tags.get("promotion_reason"),
          run.data.metrics.get("test_roc_auc") if run else None)
PY

# Point the alias back at the last known-good version
uv run python - <<'PY'
from mlflow import MlflowClient
c = MlflowClient("http://localhost:5000")
c.set_registered_model_alias("churnguard-churn-classifier", "champion", "3")
PY
```

The API hot-reloads within `CHURNGUARD_MODEL_REFRESH_SECONDS` (default 60s).
No restart required. Confirm with `GET /model/info`.

## Scenario: retrain on demand

```bash
# Locally
uv run python -m churnguard.cli bootstrap --n-trials 15

# In Compose (writes artifacts into the mounted ./models and ./reports)
docker compose run --rm trainer

# Through Prefect
docker compose exec prefect-worker python -c \
  "from churnguard.flows.training import training_pipeline; training_pipeline(n_trials=15)"
```

## Scenario: CI is red on `mypy`

`mypy` runs in strict mode. Common fixes:

- Add a `py.typed`-backed types package to the dev group.
- Add the library to `[[tool.mypy.overrides]]` in `pyproject.toml` **only** if it
  genuinely ships no stubs.
- Prefer narrowing (`assert x is not None`) over `# type: ignore`.

## SLOs and alerts

| Alert | Expression | Threshold | Action |
|---|---|---|---|
| `ChurnGuardModelNotLoaded` | `churnguard_model_loaded == 0` | 2 min | Check registry + fallback |
| `ChurnGuardHighErrorRate` | 5xx rate | > 5% for 5 min | Check logs, roll back if new version |
| `ChurnGuardHighLatency` | p95 latency | > 1s for 5 min | Check CPU, HPA, model size |
| `ChurnGuardHighDriftShare` | `churnguard_drift_share` | > 0.3 for 5 min | Follow drift runbook |

## Data retention

- Prediction logs rotate at 50 MB (`logs/predictions.jsonl.1` keeps one generation).
- MLflow artifacts live on the `./mlartifacts` bind mount locally and in S3 in
  production; ECR keeps the last 20 images.
- Drift reports keep a timestamped history under `reports/drift/history/`.
