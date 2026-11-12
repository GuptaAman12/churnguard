"""Append-only prediction logging used as the source for online drift checks."""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class PredictionLogger:
    """Thread-safe JSONL writer with simple size-based rotation."""

    def __init__(self, path: Path | str, *, enabled: bool = True, max_bytes: int = 50_000_000):
        self.path = Path(path)
        self.enabled = enabled
        self.max_bytes = max_bytes
        self._lock = threading.Lock()
        if enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, features: dict[str, Any], prediction: dict[str, Any]) -> None:
        if not self.enabled:
            return
        record = {
            "logged_at": datetime.now(UTC).isoformat(),
            "features": features,
            "prediction": prediction,
        }
        line = json.dumps(record, default=str)
        with self._lock:
            self._rotate_if_needed(len(line) + 1)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

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
