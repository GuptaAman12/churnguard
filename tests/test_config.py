"""Configuration and CLI surface tests."""

from __future__ import annotations

import json

from churnguard.cli import _parser, main
from churnguard.config import Settings


def test_parser_exposes_all_commands() -> None:
    parser = _parser()
    for command in (
        "ingest",
        "split",
        "train",
        "batch",
        "drift",
        "simulate-drift",
        "serve",
        "bootstrap",
        "info",
    ):
        assert parser.parse_args([command]).command == command


def test_train_defaults() -> None:
    args = _parser().parse_args(["train"])
    assert args.n_trials is None
    assert args.no_register is False


def test_ensure_directories(settings: Settings) -> None:
    settings.ensure_directories()
    assert settings.models_dir.exists()
    assert settings.reports_dir.exists()
    assert settings.prediction_log_path.parent.exists()


def test_environment_overrides(monkeypatch) -> None:
    monkeypatch.setenv("CHURNGUARD_LOG_LEVEL", "debug")
    monkeypatch.setenv("CHURNGUARD_API_KEY", "super-secret")
    monkeypatch.setenv("CHURNGUARD_DECISION_THRESHOLD", "0.61")
    configured = Settings(_env_file=None)
    assert configured.log_level == "DEBUG"
    assert configured.api_key == "super-secret"
    assert configured.decision_threshold == 0.61


def test_cli_info_outputs_json(capsys) -> None:
    exit_code = main(["info"])
    captured = capsys.readouterr().out
    payload = json.loads(captured)
    assert exit_code == 0
    assert payload["app_name"] == "churnguard"
    assert "tracking_uri" in payload
