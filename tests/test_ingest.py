"""Dataset acquisition tests (file:// transport and offline fallback)."""

from __future__ import annotations

import pandas as pd
import pytest

from churnguard import ingest as ingest_module
from churnguard.ingest import download_dataset, ensure_dataset
from churnguard.synthetic import generate_churn_dataset


def test_download_dataset_from_file_url(tmp_path) -> None:
    source = tmp_path / "source.csv"
    generate_churn_dataset(rows=100, seed=2).to_csv(source, index=False)
    output = tmp_path / "raw" / "telco.csv"
    result = download_dataset(output, url=source.as_uri())
    assert result == output
    assert output.exists()
    assert len(pd.read_csv(output)) == 100


def test_download_dataset_rejects_truncated_payload(tmp_path) -> None:
    tiny = tmp_path / "tiny.csv"
    tiny.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="truncated"):
        download_dataset(tmp_path / "out.csv", url=tiny.as_uri())


def test_existing_dataset_is_not_redownloaded(tmp_path) -> None:
    output = tmp_path / "raw.csv"
    generate_churn_dataset(rows=80, seed=3).to_csv(output, index=False)
    result = download_dataset(output, url="http://127.0.0.1:1/never")
    assert result == output


def test_ensure_dataset_falls_back_to_synthetic(tmp_path, monkeypatch, settings) -> None:
    def boom(*args, **kwargs):
        raise OSError("network down")

    monkeypatch.setattr(ingest_module, "download_dataset", boom)
    path, source = ensure_dataset(settings, allow_synthetic=True)
    assert source == "synthetic"
    assert path.exists()
    frame = pd.read_csv(path)
    assert len(frame) == 7043


def test_ensure_dataset_raises_when_fallback_disabled(tmp_path, monkeypatch, settings) -> None:
    def boom(*args, **kwargs):
        raise OSError("network down")

    monkeypatch.setattr(ingest_module, "download_dataset", boom)
    with pytest.raises(OSError):
        ensure_dataset(settings, allow_synthetic=False)
