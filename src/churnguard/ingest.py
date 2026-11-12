"""Dataset acquisition with a deterministic synthetic fallback."""

from __future__ import annotations

import shutil
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

from churnguard.config import Settings
from churnguard.data import DATASET_URL
from churnguard.logging import get_logger
from churnguard.schema import validate_raw
from churnguard.synthetic import generate_churn_dataset

logger = get_logger(__name__)


def download_dataset(
    output: Path,
    *,
    url: str = DATASET_URL,
    force: bool = False,
    timeout: int = 60,
) -> Path:
    """Download the IBM Telco churn CSV and validate it before committing to disk."""
    if output.exists() and not force:
        logger.info("dataset_already_present", path=str(output))
        return output

    output.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "churnguard/0.1 (+mlops)"})
    with urlopen(request, timeout=timeout) as response:
        payload = response.read()
    if len(payload) < 10_000:
        raise RuntimeError(f"Downloaded dataset looks truncated ({len(payload)} bytes)")

    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_bytes(payload)
    try:
        frame = pd.read_csv(temporary)
        validate_raw(frame)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    shutil.move(str(temporary), str(output))
    logger.info("dataset_downloaded", path=str(output), bytes=len(payload), rows=len(frame))
    return output


def ensure_dataset(
    settings: Settings,
    *,
    force: bool = False,
    allow_synthetic: bool = True,
) -> tuple[Path, str]:
    """Return a usable raw dataset, falling back to synthetic data when offline."""
    try:
        return download_dataset(settings.raw_data_path, force=force), "remote"
    except Exception as exc:
        if not allow_synthetic:
            raise
        logger.warning("dataset_download_failed_using_synthetic", error=str(exc))
        settings.raw_data_path.parent.mkdir(parents=True, exist_ok=True)
        frame = generate_churn_dataset()
        frame.to_csv(settings.raw_data_path, index=False)
        logger.info("synthetic_dataset_written", path=str(settings.raw_data_path), rows=len(frame))
        return settings.raw_data_path, "synthetic"
