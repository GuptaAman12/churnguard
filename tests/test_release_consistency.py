"""The chart must point at the image the release workflow actually publishes.

GitHub Container Registry names a package after the repository path
(``ghcr.io/<owner>/<repo>/<image>``), while ``docker/metadata-action`` refs are
built from ``${IMAGE_PREFIX}/${matrix.image}``. These two drifted once already:
the chart shipped ``ghcr.io/<owner>/<image>`` and a fresh install could not pull.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_REPOSITORY = "ghcr.io/guptaaman12/churnguard/churnguard-api"


def _chart_values() -> str:
    return (ROOT / "infra" / "helm" / "churnguard" / "values.yaml").read_text(encoding="utf-8")


def _release_workflow() -> str:
    return (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")


def test_chart_default_image_matches_the_published_path() -> None:
    match = re.search(r"^  repository: (\S+)$", _chart_values(), re.MULTILINE)
    assert match, "image.repository missing from values.yaml"
    assert match.group(1) == EXPECTED_REPOSITORY


def test_release_publishes_under_the_repository_path() -> None:
    workflow = _release_workflow()
    # The prefix is derived from the repository, so the expected path above is
    # only valid while this holds.
    assert "IMAGE_PREFIX=${GITHUB_REPOSITORY,,}" in workflow
    assert "images: ${{ env.REGISTRY }}/${{ env.IMAGE_PREFIX }}/${{ matrix.image }}" in workflow
    assert "image: churnguard-api" in workflow


def test_chart_tag_matches_the_package_version() -> None:
    package_version = re.search(
        r'^version = "([^"]+)"$',
        (ROOT / "pyproject.toml").read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    assert package_version, "version missing from pyproject.toml"
    values = _chart_values()
    tag = re.search(r'^  tag: "([^"]+)"$', values, re.MULTILINE)
    assert tag, "image.tag missing from values.yaml"
    if tag.group(1) != package_version.group(1):
        pytest.fail(
            f"chart tag {tag.group(1)} != package version {package_version.group(1)}; "
            "bump values.yaml when cutting a release"
        )
