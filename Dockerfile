# syntax=docker/dockerfile:1.7
# Multi-target image. All targets share a locked core environment; heavier
# training/orchestration and MLflow-only dependencies are added per target so
# the API image stays lean.
FROM python:3.15-slim AS base

COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    GIT_PYTHON_REFRESH=quiet

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install the locked core dependency set first for maximal layer caching.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

RUN useradd --uid 1000 --user-group --create-home --shell /usr/sbin/nologin appuser \
    && mkdir -p /app/data /app/models /app/logs /app/reports /app/.prefect \
    && chown -R appuser:appuser /app/logs /app/reports /app/data /app/models /app/.prefect

COPY --chown=appuser:appuser src ./src

# --- API: core deps only (fastapi, sklearn, xgboost, mlflow client) ---------
FROM base AS api
RUN uv sync --frozen --no-dev
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1
CMD ["uvicorn", "churnguard.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers"]

# --- Pipeline: training, tuning, drift and Prefect flows --------------------
FROM base AS pipeline
RUN uv sync --frozen --no-dev --group pipeline
USER appuser
CMD ["python", "-m", "churnguard.cli", "--help"]

# --- MLflow server: tracking/registry + Postgres and S3 clients -------------
FROM base AS mlflow
RUN uv sync --frozen --no-dev --group mlflow
USER appuser
EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=5 \
    CMD curl -fsS http://localhost:5000/health || exit 1
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
