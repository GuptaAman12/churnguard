"""Serve Prefect deployments for the training and monitoring flows."""

from __future__ import annotations

from prefect import serve

from churnguard.flows.monitoring import monitoring_pipeline
from churnguard.flows.training import training_pipeline


def main() -> None:
    serve(
        training_pipeline.to_deployment(  # type: ignore[arg-type]
            name="daily-training",
            cron="0 2 * * *",
            parameters={"n_trials": 10, "register": True, "run_drift": True},
            description="Nightly retraining with gated champion promotion.",
            tags=["training", "production"],
        ),
        monitoring_pipeline.to_deployment(  # type: ignore[arg-type]
            name="drift-check",
            interval=21600,
            parameters={"use_prediction_log": True},
            description="Six-hourly drift check over live prediction logs.",
            tags=["monitoring", "production"],
        ),
    )


if __name__ == "__main__":
    main()
