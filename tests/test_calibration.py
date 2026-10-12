"""Synthetic defects and cutoff policies (mock backbone, no downloads)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from inspector import InspectorConfig
from inspector.calibrate import (
    CalibrationSet,
    build_calibration_set,
    cutoff_for,
    escape_cutoff,
    good_only_cutoff,
    youden_cutoff,
)
from inspector.synth import GENERATOR_ORDER, augment_good, make_defect
from tests.conftest import textured_image


@pytest.mark.parametrize("kind", GENERATOR_ORDER)
def test_generators_are_seeded_and_return_masks(kind: str):
    img = textured_image(256, 256)
    place = np.ones(img.shape[:2], bool)
    out1, m1 = make_defect(kind, img, place, np.random.default_rng(3))
    out2, m2 = make_defect(kind, img, place, np.random.default_rng(3))
    assert out1.shape == img.shape and out1.dtype == np.uint8 and m1.dtype == bool
    assert np.array_equal(out1, out2) and np.array_equal(m1, m2)
    assert m1.any(), "defect mask must not be empty"
    assert np.abs(out1.astype(int) - img.astype(int))[m1].mean() > 1.0, "pixels under the mask must change"


@pytest.mark.parametrize("kind", ["scratch", "stain"])
def test_defects_are_placed_in_the_placement_region(kind: str):
    img = textured_image(256, 256)
    place = np.zeros(img.shape[:2], bool)
    place[100:140, 100:140] = True
    for s in range(5):
        _, mask = make_defect(kind, img, place, np.random.default_rng(s))
        ys, xs = np.nonzero(mask)
        cy, cx = ys.mean(), xs.mean()
        assert 60 <= cy <= 180 and 60 <= cx <= 180


def test_unknown_generator_raises():
    with pytest.raises(ValueError, match="unknown synthetic defect"):
        make_defect("dent", textured_image(), np.ones((224, 224), bool), np.random.default_rng(0))


def test_augment_good_is_mild_and_seeded():
    img = textured_image()
    place = np.ones(img.shape[:2], bool)
    a1, p1 = augment_good(img, place, np.random.default_rng(1), rotate=True)
    a2, _ = augment_good(img, place, np.random.default_rng(1), rotate=True)
    assert np.array_equal(a1, a2) and a1.shape == img.shape and p1.shape == place.shape


def test_policies_on_separable_scores():
    good = np.linspace(0.10, 0.20, 50)
    defect = np.linspace(0.30, 0.60, 50)
    assert 0.20 < youden_cutoff(good, defect) < 0.30
    assert good_only_cutoff(good) == pytest.approx(np.quantile(good, 0.99) * 1.10)
    c = escape_cutoff(defect)
    assert (defect <= c).mean() <= 0.05
    assert (defect <= np.nextafter(c, np.inf) + 1e-9).mean() > (defect <= c).mean() or c < defect.min()


def test_escape_cutoff_is_highest_allowed():
    defect = np.arange(1, 101, dtype=float)  # m = 5 allowed escapes
    c = escape_cutoff(defect)
    assert (defect <= c).sum() == 5 and c < 6.0 and c > 5.0


def test_cutoff_for_dispatch_and_oracle_guard():
    cal = CalibrationSet(np.array([0.1, 0.2]), np.array([0.5, 0.6]), ["scratch", "stain"], 2, True)
    assert cutoff_for("good_only_m0", cal) == pytest.approx(np.quantile([0.1, 0.2], 0.99))
    with pytest.raises(ValueError, match="real test labels"):
        cutoff_for("oracle", cal)
    with pytest.raises(ValueError, match="unknown policy"):
        cutoff_for("magic", cal)


@pytest.mark.parametrize("k", [1, 3])
def test_build_calibration_set_sizes_and_signal(backbone, k: int):
    cfg = InspectorConfig(resolution=112, masking="off", rotate_references=False, device="cpu")
    refs = [textured_image(112, 112, seed=s) for s in range(k)]
    cal = build_calibration_set(cfg, backbone, refs, seed=0, n_total=8, keep_examples=2)
    n = math.ceil(8 / k) * k
    assert len(cal.good_scores) == n and len(cal.defect_scores) == n
    assert cal.leave_one_out is (k >= 2)
    assert len(cal.examples) == 2
    assert np.median(cal.defect_scores) > np.median(cal.good_scores)
