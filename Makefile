# ChurnGuard — developer task runner
# Usage: make <target>.  On Windows without `make`, use `uv run ...` directly.

.DEFAULT_GOAL := help
SHELL := /bin/bash

PY := uv run
COMPOSE := docker compose

.PHONY: help install data pipeline train serve drift batch test lint format typecheck check \
        docker-build docker-up docker-down docker-logs mlflow clean

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Create the uv environment and install all dependency groups
	uv sync --all-groups
	uv run pre-commit install

data: ## Download + validate + split the dataset (DVC pipeline)
	uv run dvc repro

pipeline: ## Run the full Prefect training pipeline locally
	$(PY) python -m prefect.flows 2>/dev/null || true
	$(PY) python -c "from churnguard.flows.training import training_pipeline; training_pipeline(n_trials=10)"

bootstrap: ## Ingest → split → train → score → drift in one command
	$(PY) python -m churnguard.cli bootstrap --n-trials 10

train: ## Train, tune, evaluate and register a model
	$(PY) python -m churnguard.cli train

serve: ## Run the FastAPI prediction service with autoreload
	$(PY) python -m churnguard.cli serve --reload

batch: ## Score the current batch with the champion model
	$(PY) python -m churnguard.cli batch

drift: ## Generate a drift report against the reference dataset
	$(PY) python -m churnguard.cli drift

mlflow: ## Start a local MLflow tracking server
	$(PY) mlflow server --host 127.0.0.1 --port 5000 \
	  --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlruns

test: ## Run the test suite with coverage
	$(PY) pytest

lint: ## Ruff lint
	$(PY) ruff check src tests

format: ## Ruff format
	$(PY) ruff format src tests

typecheck: ## Strict mypy
	$(PY) mypy src

check: lint typecheck test ## Full local quality gate

docker-build: ## Build all container images
	$(COMPOSE) build

docker-up: ## Start the full local platform (MLflow, API, Prefect, monitoring)
	$(COMPOSE) up -d

docker-down: ## Stop the local platform and remove volumes
	$(COMPOSE) down -v

docker-logs: ## Tail API logs
	$(COMPOSE) logs -f api

clean: ## Remove caches and generated artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
