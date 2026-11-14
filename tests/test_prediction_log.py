"""Prediction logger tests: batching, rotation and back-pressure."""

from __future__ import annotations

import time

from churnguard.api.prediction_log import PredictionLogger


def _records(path) -> int:
    if not path.exists():
        return 0
    return len(path.read_text(encoding="utf-8").splitlines())


def test_records_are_written_in_batches(tmp_path) -> None:
    path = tmp_path / "predictions.jsonl"
    logger = PredictionLogger(path, flush_interval=0.01)
    try:
        for index in range(50):
            logger.log({"index": index}, {"churn_probability": 0.5})
        assert logger.flush()
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 50
        assert '"index": 0' in lines[0]
        assert '"index": 49' in lines[-1]
    finally:
        logger.close()


def test_rotation_moves_the_previous_file(tmp_path) -> None:
    path = tmp_path / "predictions.jsonl"
    # One record per append so the size check runs on every write.
    logger = PredictionLogger(path, max_bytes=300, flush_interval=0.01, batch_size=1)
    try:
        for index in range(40):
            logger.log({"index": index}, {})
        assert logger.flush()
        rotated = path.with_suffix(".jsonl.1")
        assert rotated.exists()
        assert _records(path) > 0
        assert _records(path) < 40
    finally:
        logger.close()


def test_disabled_logger_never_touches_disk(tmp_path) -> None:
    path = tmp_path / "predictions.jsonl"
    logger = PredictionLogger(path, enabled=False)
    logger.log({"index": 1}, {})
    assert logger.flush()
    logger.close()
    assert not path.exists()


def test_overflow_drops_records_instead_of_blocking(tmp_path, monkeypatch) -> None:
    path = tmp_path / "predictions.jsonl"
    logger = PredictionLogger(path, queue_size=2, flush_interval=0.01)
    real_write = logger._write

    def slow_write(batch):
        time.sleep(0.05)
        real_write(batch)

    monkeypatch.setattr(logger, "_write", slow_write)
    try:
        for index in range(200):
            logger.log({"index": index}, {})
        assert logger.dropped > 0
        assert logger.flush()
    finally:
        logger.close()

    # Records that were accepted are still persisted.
    assert _records(path) + logger.dropped == 200
