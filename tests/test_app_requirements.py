"""app/requirements.txt must give EVERY platform/Python exactly one torch and one torchvision requirement.

Regression test for a Community Cloud deployment where Python 3.14 on Linux x86_64 matched no torch line,
so everything except torch was installed ("ModuleNotFoundError: torch"). Network-free: evaluates the PEP 508
markers with `packaging`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from packaging.requirements import Requirement

REQS = Path(__file__).resolve().parents[1] / "app" / "requirements.txt"
PLATFORMS = {
    "linux-x86_64": {"sys_platform": "linux", "platform_machine": "x86_64"},
    "linux-aarch64": {"sys_platform": "linux", "platform_machine": "aarch64"},
    "macos-arm64": {"sys_platform": "darwin", "platform_machine": "arm64"},
    "windows-amd64": {"sys_platform": "win32", "platform_machine": "AMD64"},
}
PYTHONS = ["3.10", "3.11", "3.12", "3.13", "3.14", "3.15", "3.16"]


def _requirements() -> list[Requirement]:
    lines = [ln.strip() for ln in REQS.read_text().splitlines()]
    return [Requirement(ln) for ln in lines if ln and not ln.startswith(("#", "-"))]


@pytest.mark.parametrize("python", PYTHONS)
@pytest.mark.parametrize("platform", list(PLATFORMS))
@pytest.mark.parametrize("pkg", ["torch", "torchvision"])
def test_exactly_one_requirement_applies(pkg: str, platform: str, python: str):
    env = {**PLATFORMS[platform], "python_version": python, "python_full_version": f"{python}.0"}
    applicable = [r for r in _requirements() if r.name == pkg and (r.marker is None or r.marker.evaluate(env))]
    assert len(applicable) == 1, f"{pkg} on {platform} / Python {python}: {len(applicable)} matching lines"


@pytest.mark.parametrize("python", ["3.10", "3.11", "3.12", "3.13", "3.14", "3.15"])
def test_cloud_gets_cpu_wheels(python: str):
    env = {**PLATFORMS["linux-x86_64"], "python_version": python, "python_full_version": f"{python}.0"}
    for pkg in ("torch", "torchvision"):
        (req,) = [r for r in _requirements() if r.name == pkg and r.marker.evaluate(env)]
        assert req.url and "%2Bcpu" in req.url and "#sha256=" in req.url, f"{pkg} py{python}: {req}"


def test_no_extra_index_url():
    assert "--extra-index-url" not in REQS.read_text().replace("No --extra-index-url", "")
