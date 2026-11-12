"""Training orchestration: Optuna tuning, evaluation, MLflow tracking, promotion."""

from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import joblib
import mlflow
import numpy as np
import optuna
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.model_selection import StratifiedKFold, cross_val_score

from churnguard.config import Settings
from churnguard.constants import TARGET_COLUMN
from churnguard.data import build_manifest
from churnguard.features import (
    build_pipeline,
    get_feature_names,
    model_feature_importance,
)
from churnguard.logging import get_logger
from churnguard.modeling.evaluate import EvaluationResult, evaluate, subgroup_metrics
from churnguard.modeling.registry import decide_promotion, promote
from churnguard.schema import validate_processed

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = get_logger(__name__)

DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 400,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "min_child_weight": 2.0,
    "gamma": 0.0,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
}


@dataclass(frozen=True)
class TrainResult:
    run_id: str
    model_uri: str
    version: str | None
    metrics: dict[str, float]
    threshold: float
    promoted: bool
    promotion_reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "model_uri": self.model_uri,
            "version": self.version,
            "metrics": self.metrics,
            "threshold": self.threshold,
            "promoted": self.promoted,
            "promotion_reason": self.promotion_reason,
        }


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unknown"


def make_classifier(
    params: dict[str, Any],
    *,
    random_state: int,
    scale_pos_weight: float,
) -> Any:
    from xgboost import XGBClassifier

    merged = {**DEFAULT_PARAMS, **params}
    return XGBClassifier(
        **merged,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=int(os.environ.get("CHURNGUARD_N_JOBS", "-1")),
    )


def suggest_params(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 900, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 12.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def scale_pos_weight(y: pd.Series) -> float:
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    return round(negatives / max(positives, 1), 4)


def tune_hyperparameters(
    features_train: pd.DataFrame,
    target_train: pd.Series,
    *,
    n_trials: int,
    timeout_seconds: int,
    cv_folds: int,
    random_state: int,
    pos_weight: float,
) -> tuple[dict[str, Any], float]:
    """Bayesian hyperparameter search (TPE) with stratified K-fold ROC-AUC."""
    if n_trials <= 0:
        return {}, 0.0

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        pipeline = build_pipeline(
            make_classifier(params, random_state=random_state, scale_pos_weight=pos_weight)
        )
        scores = cross_val_score(
            pipeline,
            features_train,
            target_train,
            cv=cv,
            scoring="roc_auc",
            n_jobs=1,
        )
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
        study_name="churnguard-xgb",
    )
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout_seconds if timeout_seconds > 0 else None,
        show_progress_bar=False,
    )
    logger.info(
        "tuning_complete",
        trials=len(study.trials),
        best_cv_roc_auc=round(study.best_value, 5),
    )
    return dict(study.best_params), float(study.best_value)


def build_model_card(
    *,
    metrics: dict[str, float],
    params: dict[str, Any],
    threshold: float,
    data_fingerprint: str,
    subgroup: dict[str, dict[str, float]],
    feature_importance: dict[str, float],
) -> str:
    top_features = list(feature_importance.items())[:8]
    feature_lines = "\n".join(f"| `{name}` | {value:.4f} |" for name, value in top_features)
    subgroup_lines = "\n".join(
        f"| `{group}` | {values.get('roc_auc', 0):.4f} | {values.get('recall', 0):.4f} | "
        f"{int(values.get('n', 0))} |"
        for group, values in subgroup.items()
    )
    return f"""# Model Card — ChurnGuard churn classifier

**Generated:** {datetime.now(UTC).isoformat()}  
**Algorithm:** XGBoost ({params.get("n_estimators")} trees, depth {params.get("max_depth")})  
**Data fingerprint:** `{data_fingerprint}`  
**Decision threshold:** {threshold:.4f}

## Intended use
Estimate the probability that a telecom customer will churn so that retention
teams can prioritise outreach. Not intended for automated adverse decisions
(e.g. denying service) without human review.

## Test-set performance
| Metric | Value |
|---|---|
| ROC-AUC | {metrics.get("roc_auc", 0):.4f} |
| PR-AUC | {metrics.get("pr_auc", 0):.4f} |
| F1 | {metrics.get("f1", 0):.4f} |
| Precision | {metrics.get("precision", 0):.4f} |
| Recall | {metrics.get("recall", 0):.4f} |
| Log loss | {metrics.get("log_loss", 0):.4f} |
| Brier score | {metrics.get("brier_score", 0):.4f} |

## Disaggregated performance
| Group | ROC-AUC | Recall | n |
|---|---|---|---|
{subgroup_lines}

## Most important features
| Feature | Importance |
|---|---|
{feature_lines}

## Limitations
- Trained on a single public IBM telecom dataset; geography and product mix differ
  from production populations.
- Churn label reflects historical outcomes and may encode selection bias.
- Drift detection must be paired with periodic retraining and human review.
- Probabilities are not calibrated post-hoc; use rank ordering for prioritisation.
"""


def train_and_log(
    settings: Settings,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    n_trials: int | None = None,
    register: bool = True,
    save_fallback: bool = True,
    experiment_name: str | None = None,
    extra_tags: dict[str, str] | None = None,
) -> TrainResult:
    """Train, evaluate and register a model; promote only if quality gates pass."""
    validate_processed(train_df)
    validate_processed(test_df)
    settings.ensure_directories()

    n_trials = settings.optuna_trials if n_trials is None else n_trials
    experiment = experiment_name or settings.experiment_name
    mlflow.set_tracking_uri(settings.tracking_uri)
    mlflow.set_experiment(experiment)

    features_train = train_df.drop(columns=[TARGET_COLUMN])
    target_train = train_df[TARGET_COLUMN].astype(int)
    features_test = test_df.drop(columns=[TARGET_COLUMN])
    target_test = test_df[TARGET_COLUMN].astype(int)

    pos_weight = scale_pos_weight(target_train)
    fingerprint = build_manifest(train_df, "train", "churnguard.training").fingerprint
    sha = git_sha()

    logger.info(
        "training_started",
        train_rows=len(train_df),
        test_rows=len(test_df),
        n_trials=n_trials,
        tracking_uri=settings.tracking_uri,
    )

    with mlflow.start_run(run_name=f"xgb-{datetime.now(UTC):%Y%m%d-%H%M%S}") as run:
        mlflow.set_tags(
            {
                "git_sha": sha,
                "data_fingerprint": fingerprint,
                "python_version": platform.python_version(),
                "owner": "churnguard",
                **(extra_tags or {}),
            }
        )
        mlflow.log_params({"scale_pos_weight": pos_weight, "decision_threshold_search": True})

        try:
            train_dataset = mlflow.data.from_pandas(  # type: ignore[attr-defined]
                train_df, name="telco-churn-train", targets=TARGET_COLUMN
            )
            mlflow.log_input(train_dataset, context="training")
        except Exception as exc:  # pragma: no cover - dataset logging is best effort
            logger.warning("dataset_logging_failed", error=str(exc))

        best_params: dict[str, Any] = {}
        cv_auc = 0.0
        if n_trials > 0:
            best_params, cv_auc = tune_hyperparameters(
                features_train,
                target_train,
                n_trials=n_trials,
                timeout_seconds=settings.optuna_timeout_seconds,
                cv_folds=settings.cv_folds,
                random_state=settings.random_state,
                pos_weight=pos_weight,
            )
            mlflow.log_params({f"best_{key}": value for key, value in best_params.items()})
            mlflow.log_metric("cv_roc_auc", cv_auc)

        pipeline = build_pipeline(
            make_classifier(
                best_params,
                random_state=settings.random_state,
                scale_pos_weight=pos_weight,
            )
        )
        pipeline.fit(features_train, target_train)

        proba = pipeline.predict_proba(features_test)[:, 1]
        evaluation: EvaluationResult = evaluate(target_test.to_numpy(), proba)
        metrics = evaluation.metrics

        subgroups = subgroup_metrics(
            features_test,
            target_test.to_numpy(),
            proba,
            columns=("gender", "SeniorCitizen", "Contract"),
            threshold=evaluation.threshold,
        )
        importance = model_feature_importance(pipeline)

        mlflow.log_params(
            {
                "n_estimators": best_params.get("n_estimators", DEFAULT_PARAMS["n_estimators"]),
                "max_depth": best_params.get("max_depth", DEFAULT_PARAMS["max_depth"]),
                "learning_rate": best_params.get("learning_rate", DEFAULT_PARAMS["learning_rate"]),
                "model_type": "xgboost",
                "feature_count": len(get_feature_names(pipeline)),
            }
        )
        mlflow.log_metrics({f"test_{key}": value for key, value in metrics.items()})
        mlflow.log_metric("decision_threshold", evaluation.threshold)
        mlflow.log_dict(evaluation.to_dict(), "evaluation/metrics.json")
        mlflow.log_dict(subgroups, "evaluation/subgroup_metrics.json")
        mlflow.log_dict(importance, "evaluation/feature_importance.json")
        mlflow.log_dict({"fpr_tpr": evaluation.roc_curve}, "evaluation/roc_curve.json")

        model_card = build_model_card(
            metrics=metrics,
            params=best_params,
            threshold=evaluation.threshold,
            data_fingerprint=fingerprint,
            subgroup=subgroups,
            feature_importance=importance,
        )
        mlflow.log_text(model_card, "model_card.md")

        signature = infer_signature(
            features_train.head(50), pipeline.predict_proba(features_train.head(50))[:, 1]
        )
        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            signature=signature,
            input_example=features_train.head(5),
            registered_model_name=settings.registry_model_name if register else None,
            metadata={"decision_threshold": evaluation.threshold},
            # The artifact contains a trusted, in-repo feature-engineering function.
            # Pickle keeps the registry artifact portable across the API/pipeline images.
            serialization_format="pickle",
        )
        run_id = run.info.run_id

    version: str | None = None
    promoted = False
    reason = "registration disabled"
    if register:
        client = MlflowClient()
        versions = client.search_model_versions(
            f"name='{settings.registry_model_name}' and run_id='{run_id}'"
        )
        if versions:
            version = max(versions, key=lambda item: int(item.version)).version
            decision = decide_promotion(client, settings, version, metrics["roc_auc"])
            promote(
                client,
                settings,
                version,
                decision,
                git_sha=sha,
                data_fingerprint=fingerprint,
                threshold=evaluation.threshold,
            )
            promoted = decision.promoted
            reason = decision.reason
        else:  # pragma: no cover - defensive
            reason = "registered model version not found"

    if save_fallback:
        joblib.dump(pipeline, settings.fallback_model_path)
        metadata = {
            "run_id": run_id,
            "version": version,
            "model_uri": model_info.model_uri,
            "threshold": evaluation.threshold,
            "metrics": metrics,
            "trained_at": datetime.now(UTC).isoformat(),
            "data_fingerprint": fingerprint,
            "git_sha": sha,
            "feature_names": get_feature_names(pipeline),
        }
        settings.fallback_model_path.with_suffix(".json").write_text(
            _json_dumps(metadata), encoding="utf-8"
        )

    logger.info(
        "training_finished",
        run_id=run_id,
        roc_auc=metrics["roc_auc"],
        version=version,
        promoted=promoted,
    )
    return TrainResult(
        run_id=run_id,
        model_uri=model_info.model_uri,
        version=version,
        metrics=metrics,
        threshold=evaluation.threshold,
        promoted=promoted,
        promotion_reason=reason,
    )


def _json_dumps(payload: object) -> str:
    import json

    return json.dumps(payload, indent=2, default=str)
