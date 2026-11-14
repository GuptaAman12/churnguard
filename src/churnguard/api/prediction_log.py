"""Append-only prediction logging used as the source for online drift checks."""

from __future__ import annotations

import json
import queue
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from churnguard.logging import get_logger

logger = get_logger(__name__)

_SENTINEL = None


class PredictionLogger:
    """Non-blocking JSONL writer with size-based rotation.

    Requests hand a serialised line to a bounded queue and return; one daemon
    thread owns the file and writes batched appends, so neither disk latency nor
    rotation can appear in the prediction path. Records are dropped (and counted)
    when the writer cannot keep up: losing log lines is preferable to adding
    latency or unbounded memory to the API.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        enabled: bool = True,
        max_bytes: int = 50_000_000,
        queue_size: int = 20_000,
        flush_interval: float = 0.2,
        batch_size: int = 500,
    ) -> None:
        self.path = Path(path)
        self.enabled = enabled
        self.max_bytes = max_bytes
        self.flush_interval = flush_interval
        self.batch_size = batch_size
        self.dropped = 0
        self._queue: queue.Queue[str | None] = queue.Queue(maxsize=queue_size)
        self._thread: threading.Thread | None = None
        self._closed = False
        if enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._thread = threading.Thread(
                target=self._writer, name="prediction-log-writer", daemon=True
            )
            self._thread.start()

    # -- public API ---------------------------------------------------------
    def log(self, features: dict[str, Any], prediction: dict[str, Any]) -> None:
        if not self.enabled or self._closed:
            return
        record = {
            "logged_at": datetime.now(UTC).isoformat(),
            "features": features,
            "prediction": prediction,
        }
        line = json.dumps(record, default=str)
        try:
            self._queue.put_nowait(line)
        except queue.Full:
            self.dropped += 1
            if self.dropped % 100 == 1:
                logger.warning("prediction_log_overflow", dropped=self.dropped)

    def flush(self, timeout: float = 10.0) -> bool:
        """Block until every queued record has been written. Returns success."""
        if not self.enabled:
            return True
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._queue.unfinished_tasks == 0:
                return True
            time.sleep(0.01)
        return False

    def close(self, timeout: float = 10.0) -> None:
        """Drain the queue and stop the writer thread."""
        if not self.enabled or self._closed:
            return
        self._closed = True
        self.flush(timeout=timeout)
        try:
            self._queue.put(_SENTINEL, timeout=timeout)
        except queue.Full:  # pragma: no cover - shutdown under extreme load
            logger.warning("prediction_log_close_timeout")
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    # -- writer thread ------------------------------------------------------
    def _writer(self) -> None:
        while True:
            try:
                item = self._queue.get(timeout=self.flush_interval)
            except queue.Empty:
                continue
            if item is _SENTINEL:
                self._queue.task_done()
                return

            batch = [item]
            stop = False
            while len(batch) < self.batch_size:
                try:
                    nxt = self._queue.get_nowait()
                except queue.Empty:
                    break
                if nxt is _SENTINEL:
                    self._queue.task_done()
                    stop = True
                    break
                batch.append(nxt)
            try:
                self._write(batch)
            except OSError as exc:  # pragma: no cover - disk failure
                logger.warning("prediction_log_write_failed", error=str(exc))
            finally:
                for _ in batch:
                    self._queue.task_done()
            if stop:
                return

    def _write(self, batch: list[str]) -> None:
        payload = "\n".join(batch) + "\n"
        self._rotate_if_needed(len(payload))
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()

    def _rotate_if_needed(self, incoming: int) -> None:
        try:
            if self.path.exists() and self.path.stat().st_size + incoming > self.max_bytes:
                rotated = self.path.with_suffix(self.path.suffix + ".1")
                if rotated.exists():
                    rotated.unlink()
                self.path.rename(rotated)
        except OSError:
            # Never fail a prediction because log rotation failed.
            pass
