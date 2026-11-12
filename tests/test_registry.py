"""Promotion-gate and registry tag tests."""

from __future__ import annotations

from types import SimpleNamespace

from mlflow.exceptions import MlflowException

from churnguard.modeling.registry import decide_promotion, get_champion, promote


class FakeClient:
    def __init__(self, champion=None, champion_auc: float | None = 0.85) -> None:
        self.champion = champion
        self.champion_auc = champion_auc
        self.tags: list[tuple[str, str, str]] = []
        self.aliases: list[tuple[str, str, str]] = []

    def get_model_version_by_alias(self, name: str, alias: str):
        if self.champion is None:
            raise MlflowException("alias not found")
        return self.champion

    def get_run(self, run_id: str):
        metrics = {} if self.champion_auc is None else {"test_roc_auc": self.champion_auc}
        return SimpleNamespace(data=SimpleNamespace(metrics=metrics))

    def set_model_version_tag(self, name: str, version: str, key: str, value: str) -> None:
        self.tags.append((version, key, value))

    def set_registered_model_alias(self, name: str, alias: str, version: str) -> None:
        self.aliases.append((alias, version, name))


def test_no_champion_applies_absolute_floor(settings) -> None:
    client = FakeClient(champion=None)
    rejected = decide_promotion(client, settings, "1", 0.79)
    assert rejected.promoted is False

    promoted = decide_promotion(client, settings, "1", 0.81)
    assert promoted.promoted is True
    assert promoted.champion_version is None


def test_candidate_must_not_regress_beyond_tolerance(settings) -> None:
    champion = SimpleNamespace(version="3", run_id="run-3")
    client = FakeClient(champion=champion, champion_auc=0.85)

    regressed = decide_promotion(client, settings, "4", 0.84)
    assert regressed.promoted is False
    assert "regresses" in regressed.reason

    acceptable = decide_promotion(client, settings, "4", 0.848)
    assert acceptable.promoted is True
    assert acceptable.champion_version == "3"


def test_quality_floor_still_applies_with_champion(settings) -> None:
    champion = SimpleNamespace(version="3", run_id="run-3")
    client = FakeClient(champion=champion, champion_auc=0.70)
    decision = decide_promotion(client, settings, "4", 0.75)
    assert decision.promoted is False
    assert "quality floor" in decision.reason


def test_missing_champion_metric_falls_back_to_floor(settings) -> None:
    champion = SimpleNamespace(version="3", run_id="run-3")
    client = FakeClient(champion=champion, champion_auc=None)
    decision = decide_promotion(client, settings, "4", 0.82)
    assert decision.promoted is True


def test_get_champion_returns_none_when_alias_missing(settings) -> None:
    assert get_champion(FakeClient(champion=None), settings) is None


def test_promote_tags_and_moves_alias(settings) -> None:
    client = FakeClient(champion=None)
    decision = decide_promotion(client, settings, "2", 0.9)
    promote(
        client,
        settings,
        "2",
        decision,
        git_sha="abc123",
        data_fingerprint="fp123",
        threshold=0.42,
    )
    tag_keys = {key for _, key, _ in client.tags}
    assert {"git_sha", "data_fingerprint", "decision_threshold", "promotion_reason"} <= tag_keys
    assert client.aliases == [("champion", "2", settings.registry_model_name)]


def test_rejected_candidate_does_not_move_alias(settings) -> None:
    champion = SimpleNamespace(version="3", run_id="run-3")
    client = FakeClient(champion=champion, champion_auc=0.90)
    decision = decide_promotion(client, settings, "4", 0.80)
    promote(
        client,
        settings,
        "4",
        decision,
        git_sha="abc123",
        data_fingerprint="fp123",
        threshold=0.42,
    )
    assert client.aliases == []
    assert any(key == "promotion_reason" for _, key, _ in client.tags)
