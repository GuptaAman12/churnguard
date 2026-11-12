"""MLflow registry interactions: champion lookup and gated promotion."""

from __future__ import annotations

from dataclasses import dataclass

from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException

from churnguard.config import Settings
from churnguard.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class PromotionDecision:
    promoted: bool
    reason: str
    champion_version: str | None
    champion_auc: float | None
    candidate_auc: float


def get_champion(client: MlflowClient, settings: Settings) -> ModelVersion | None:
    try:
        return client.get_model_version_by_alias(
            settings.registry_model_name, settings.champion_alias
        )
    except MlflowException:
        return None


def get_version_auc(client: MlflowClient, version: ModelVersion) -> float | None:
    """Read the test ROC-AUC attached to the run that produced a model version."""
    if version.run_id is None:
        return None
    try:
        run = client.get_run(version.run_id)
    except MlflowException:
        return None
    auc = run.data.metrics.get("test_roc_auc")
    return float(auc) if auc is not None else None


def decide_promotion(
    client: MlflowClient,
    settings: Settings,
    candidate_version: str,
    candidate_auc: float,
) -> PromotionDecision:
    """Gate: absolute quality floor + no material regression vs. champion."""
    champion = get_champion(client, settings)
    if champion is None:
        return PromotionDecision(
            promoted=candidate_auc >= settings.min_roc_auc,
            reason="no champion exists; applying absolute quality floor",
            champion_version=None,
            champion_auc=None,
            candidate_auc=candidate_auc,
        )

    champion_auc = get_version_auc(client, champion)
    if champion_auc is None:
        return PromotionDecision(
            promoted=candidate_auc >= settings.min_roc_auc,
            reason="champion metric unavailable; applying absolute quality floor",
            champion_version=champion.version,
            champion_auc=None,
            candidate_auc=candidate_auc,
        )

    passes_floor = candidate_auc >= settings.min_roc_auc
    no_regression = candidate_auc >= champion_auc - settings.promotion_tolerance
    promoted = passes_floor and no_regression

    if not passes_floor:
        reason = f"candidate AUC {candidate_auc:.4f} below quality floor {settings.min_roc_auc:.4f}"
    elif not no_regression:
        reason = (
            f"candidate AUC {candidate_auc:.4f} regresses vs champion {champion_auc:.4f} "
            f"beyond tolerance {settings.promotion_tolerance}"
        )
    else:
        reason = (
            f"candidate AUC {candidate_auc:.4f} >= champion {champion_auc:.4f} "
            f"- tolerance {settings.promotion_tolerance}"
        )

    return PromotionDecision(
        promoted=promoted,
        reason=reason,
        champion_version=champion.version,
        champion_auc=champion_auc,
        candidate_auc=candidate_auc,
    )


def promote(
    client: MlflowClient,
    settings: Settings,
    version: str,
    decision: PromotionDecision,
    *,
    git_sha: str,
    data_fingerprint: str,
    threshold: float,
) -> None:
    """Attach lineage tags and (optionally) move the champion alias."""
    client.set_model_version_tag(settings.registry_model_name, version, "git_sha", git_sha)
    client.set_model_version_tag(
        settings.registry_model_name, version, "data_fingerprint", data_fingerprint
    )
    client.set_model_version_tag(
        settings.registry_model_name, version, "decision_threshold", f"{threshold:.4f}"
    )
    client.set_model_version_tag(
        settings.registry_model_name, version, "promotion_reason", decision.reason
    )
    if decision.promoted:
        client.set_registered_model_alias(
            settings.registry_model_name, settings.champion_alias, version
        )
        logger.info("champion_promoted", version=version, reason=decision.reason)
    else:
        logger.warning("champion_retained", candidate_version=version, reason=decision.reason)
