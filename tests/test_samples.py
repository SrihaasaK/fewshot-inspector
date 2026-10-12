"""The bundled demo samples are complete, licensed separately and match their manifest (no MVTec needed)."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
from PIL import Image

SAMPLES = Path(__file__).resolve().parents[1] / "app" / "samples"


@pytest.fixture(scope="module")
def manifest() -> list[dict[str, str]]:
    with (SAMPLES / "MANIFEST.csv").open() as f:
        return list(csv.DictReader(f))


def test_license_is_cc_by_nc_sa_and_separate_from_mit():
    text = (SAMPLES / "LICENSE.md").read_text()
    assert "CC BY-NC-SA 4.0" in text and "MVTec" in text and "not** covered by the repository's MIT" in text


def test_manifest_matches_files(manifest):
    listed = {r["sample_path"] for r in manifest}
    on_disk = {p.relative_to(SAMPLES).as_posix() for p in SAMPLES.rglob("*.jpg")}
    assert listed == on_disk


@pytest.mark.parametrize("category", ["bottle", "screw"])
def test_each_example_has_4_refs_2_good_4_defects(manifest, category):
    rows = [r for r in manifest if r["category"] == category]
    assert sum(r["split"] == "refs" for r in rows) == 4
    tests = [r for r in rows if r["split"] == "tests"]
    assert sum(r["label"] == "0" for r in tests) == 2 and sum(r["label"] == "1" for r in tests) == 4


def test_images_are_downscaled_and_readable(manifest):
    for r in manifest:
        with Image.open(SAMPLES / r["sample_path"]) as im:
            assert max(im.size) <= 512
            assert f"{im.width}x{im.height}" == r["saved_size"]
