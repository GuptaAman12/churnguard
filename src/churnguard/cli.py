"""Unified command-line interface for ingestion, training, scoring and monitoring."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from churnguard import __version__
from churnguard.config import Settings, get_settings
from churnguard.logging import configure_logging, get_logger

logger = get_logger(__name__)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="churnguard",
        description="ChurnGuard MLOps platform CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"churnguard {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Download the raw dataset (with synthetic fallback)")
    ingest.add_argument("--force", action="store_true")
    ingest.add_argument("--no-synthetic", action="store_true")
    ingest.add_argument("--url", default=None)

    split = sub.add_parser("split", help="Clean, validate and split the dataset")
    split.add_argument("--raw", type=Path, default=None)
    split.add_argument("--test-size", type=float, default=None)
    split.add_argument("--seed", type=int, default=None)

    train = sub.add_parser("train", help="Tune, train, evaluate and register a model")
    train.add_argument("--n-trials", type=int, default=None)
    train.add_argument("--no-register", action="store_true")
    train.add_argument("--tracking-uri", default=None)
    train.add_argument("--experiment", default=None)
    train.add_argument("--train-path", type=Path, default=None)
    train.add_argument("--test-path", type=Path, default=None)
    train.add_argument("--trigger", default="cli")

    batch = sub.add_parser("batch", help="Score a batch file with the active model")
    batch.add_argument("--input", type=Path, default=None)
    batch.add_argument("--output-dir", type=Path, default=None)
    batch.add_argument("--threshold", type=float, default=None)

    drift = sub.add_parser("drift", help="Run a data-drift report")
    drift.add_argument("--reference", type=Path, default=None)
    drift.add_argument("--current", type=Path, default=None)
    drift.add_argument("--from-prediction-log", action="store_true")
    drift.add_argument("--no-push", action="store_true")

    simulate = sub.add_parser("simulate-drift", help="Create a drifted current batch")
    simulate.add_argument("--source", type=Path, default=None)
    simulate.add_argument("--output", type=Path, default=None)
    simulate.add_argument("--rows", type=int, default=2000)
    simulate.add_argument("--seed", type=int, default=7)

    serve = sub.add_parser("serve", help="Run the prediction API")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")

    bootstrap = sub.add_parser(
        "bootstrap", help="Ingest + split + train + score + drift (one command)"
    )
    bootstrap.add_argument("--n-trials", type=int, default=10)
    bootstrap.add_argument("--skip-drift", action="store_true")

    sub.add_parser("info", help="Print resolved configuration")
    return parser


def _masked_settings(settings: Settings) -> dict[str, Any]:
    payload = settings.model_dump(mode="json")
    for key in ("api_key", "aws_secret_access_key", "aws_access_key_id"):
        if payload.get(key):
            payload[key] = "***"
    return payload


def _cmd_ingest(args: argparse.Namespace, settings: Settings) -> int:
    from churnguard.ingest import download_dataset, ensure_dataset

    if args.url:
        path = download_dataset(settings.raw_data_path, url=args.url, force=args.force)
        source = "remote"
    else:
        path, source = ensure_dataset(
            settings, force=args.force, allow_synthetic=not args.no_synthetic
        )
    print(json.dumps({"dataset": str(path), "source": source}, indent=2))
    return 0


def _cmd_split(args: argparse.Namespace, settings: Settings) -> int:
    from churnguard.data import prepare_datasets
    from churnguard.ingest import ensure_dataset

    raw = args.raw or settings.raw_data_path
    if not Path(raw).exists():
        raw, _ = ensure_dataset(settings)
    result = prepare_datasets(
        raw_path=Path(raw),
        train_path=settings.train_path,
        test_path=settings.test_path,
        reference_path=settings.reference_path,
        test_size=args.test_size or settings.test_size,
        random_state=args.seed if args.seed is not None else settings.random_state,
    )
    print(json.dumps(result, indent=2))
    return 0


def _cmd_train(args: argparse.Namespace, settings: Settings) -> int:
    from churnguard.data import read_dataset
    from churnguard.modeling.train import train_and_log

    if args.tracking_uri:
        settings = settings.model_copy(update={"tracking_uri": args.tracking_uri})
    train_df = read_dataset(args.train_path or settings.train_path)
    test_df = read_dataset(args.test_path or settings.test_path)
    result = train_and_log(
        settings,
        train_df,
        test_df,
        n_trials=args.n_trials,
        register=not args.no_register,
        experiment_name=args.experiment,
        extra_tags={"trigger": args.trigger},
    )
    print(json.dumps(result.to_dict(), indent=2))
    return 0


def _cmd_batch(args: argparse.Namespace, settings: Settings) -> int:
    from churnguard.api.model_loader import ModelService
    from churnguard.batch import score_batch

    service = ModelService(settings)
    service.load()
    result = score_batch(
        service,
        args.input or settings.current_data_path,
        args.output_dir or settings.reports_dir / "batch",
        threshold=args.threshold,
    )
    print(json.dumps(result.to_dict(), indent=2))
    return 0


def _cmd_drift(args: argparse.Namespace, settings: Settings) -> int:
    from churnguard.data import read_dataset
    from churnguard.monitoring.drift import load_prediction_features, run_drift

    reference = read_dataset(args.reference or settings.reference_path)
    if args.from_prediction_log:
        current = load_prediction_features(settings.prediction_log_path)
    else:
        current = read_dataset(args.current or settings.current_data_path)
    summary = run_drift(reference, current, settings, push=not args.no_push)
    print(json.dumps(summary.to_dict(), indent=2, default=str))
    return 0


def _cmd_simulate(args: argparse.Namespace, settings: Settings) -> int:
    from churnguard.monitoring.simulate import simulate_production_drift

    source = args.source or settings.test_path
    output = args.output or settings.current_data_path
    frame = simulate_production_drift(source, output, rows=args.rows, seed=args.seed)
    print(json.dumps({"output": str(output), "rows": len(frame)}, indent=2))
    return 0


def _cmd_serve(args: argparse.Namespace, settings: Settings) -> int:
    import uvicorn

    uvicorn.run(
        "churnguard.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=settings.log_level.lower(),
    )
    return 0


def _cmd_bootstrap(args: argparse.Namespace, settings: Settings) -> int:
    from churnguard.api.model_loader import ModelService
    from churnguard.batch import score_batch
    from churnguard.data import prepare_datasets, read_dataset
    from churnguard.ingest import ensure_dataset
    from churnguard.modeling.train import train_and_log
    from churnguard.monitoring.drift import run_drift
    from churnguard.monitoring.simulate import simulate_production_drift

    raw, source = ensure_dataset(settings)
    split_result = prepare_datasets(
        raw_path=raw,
        train_path=settings.train_path,
        test_path=settings.test_path,
        reference_path=settings.reference_path,
        test_size=settings.test_size,
        random_state=settings.random_state,
    )
    result = train_and_log(
        settings,
        read_dataset(settings.train_path),
        read_dataset(settings.test_path),
        n_trials=args.n_trials,
        extra_tags={"trigger": "bootstrap"},
    )
    payload: dict[str, Any] = {
        "dataset_source": source,
        "split": split_result,
        "training": result.to_dict(),
    }

    service = ModelService(settings)
    service.load()
    batch = score_batch(service, settings.current_data_path, settings.reports_dir / "batch")
    payload["batch"] = batch.to_dict()

    if not args.skip_drift:
        simulate_production_drift(settings.test_path, settings.current_data_path)
        summary = run_drift(
            read_dataset(settings.reference_path),
            read_dataset(settings.current_data_path),
            settings,
        )
        payload["drift"] = {
            "drift_share": summary.drift_share,
            "drifted_features": summary.drifted_features,
            "dataset_drift": summary.dataset_drift,
        }

    print(json.dumps(payload, indent=2, default=str))
    return 0


def _cmd_info(args: argparse.Namespace, settings: Settings) -> int:
    print(json.dumps(_masked_settings(settings), indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    args = _parser().parse_args(argv)
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)

    handlers = {
        "ingest": _cmd_ingest,
        "split": _cmd_split,
        "train": _cmd_train,
        "batch": _cmd_batch,
        "drift": _cmd_drift,
        "simulate-drift": _cmd_simulate,
        "serve": _cmd_serve,
        "bootstrap": _cmd_bootstrap,
        "info": _cmd_info,
    }
    try:
        return handlers[args.command](args, settings)
    except Exception as exc:  # surfaced cleanly to operators / CI
        logger.error("command_failed", command=args.command, error=str(exc))
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
