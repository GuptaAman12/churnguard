"""Guard the image coordinates the release workflow publishes.

GitHub Container Registry names a package after the repository path
(``ghcr.io/<owner>/<repo>/<image>``), while ``docker/metadata-action`` refs are
built from ``${IMAGE_PREFIX}/${matrix.image}``. These drifted once already, so
the mapping is checked here.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_REPOSITORY = "ghcr.io/guptaaman12/churnguard/churnguard-api"


def _release_workflow() -> str:
    return (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")


def test_release_publishes_under_the_repository_path() -> None:
    workflow = _release_workflow()
    assert "IMAGE_PREFIX=${GITHUB_REPOSITORY,,}" in workflow
    assert "images: ${{ env.REGISTRY }}/${{ env.IMAGE_PREFIX }}/${{ matrix.image }}" in workflow
    assert "image: churnguard-api" in workflow
    assert EXPECTED_REPOSITORY == "ghcr.io/guptaaman12/churnguard/churnguard-api"


def test_release_tags_images_with_version_sha_and_latest() -> None:
    workflow = _release_workflow()
    assert "type=semver,pattern={{version}}" in workflow
    assert "type=sha,prefix=sha-" in workflow
    assert "type=raw,value=latest,enable={{is_default_branch}}" in workflow
