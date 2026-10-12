"""End-to-end behaviour of Inspector with the mock backbone (no weight downloads)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from inspector import ImageLoadError
from tests.conftest import textured_image


def paste_noise(image: np.ndarray, top: int, left: int, size: int, seed: int = 1) -> np.ndarray:
    out = image.copy()
    noise = np.random.default_rng(seed).integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    out[top:top + size, left:left + size] = noise
    return out


def test_reference_scored_against_itself_is_zero(make_inspector):
    ref = textured_image()
    insp = make_inspector().fit([ref])
    result = insp.score(ref)
    assert result.score == pytest.approx(0.0, abs=1e-5)
    assert np.abs(result.patch_map).max() < 1e-5


def test_reference_with_rotations_still_scores_zero(make_inspector):
    ref = textured_image()
    insp = make_inspector(rotate_references=True).fit([ref])
    assert len(insp.bank) == 8 * 16 * 16  # 8 rotations × 16×16 patches at 224 px
    assert insp.score(ref).score == pytest.approx(0.0, abs=1e-5)


def test_noise_square_scores_higher_and_heatmap_peaks_inside(make_inspector):
    ref = textured_image()
    top, left, size = 98, 126, 42
    defect = paste_noise(ref, top, left, size)
    insp = make_inspector().fit([ref])
    clean, bad = insp.score(ref), insp.score(defect)
    assert bad.score > clean.score + 0.05
    assert bad.heatmap.shape == ref.shape[:2]
    y, x = np.unravel_index(np.argmax(bad.heatmap), bad.heatmap.shape)
    assert top <= y < top + size and left <= x < left + size


def test_predict_uses_cutoff(make_inspector):
    ref = textured_image()
    insp = make_inspector().fit([ref])
    defect = paste_noise(ref, 98, 126, 42)
    assert insp.predict(defect, cutoff=0.01).defective
    pred = insp.predict(ref, cutoff=0.01)
    assert not pred.defective and pred.margin < 0


def test_grayscale_inputs(make_inspector, tmp_path: Path):
    gray = textured_image()[:, :, 0]
    insp = make_inspector().fit([gray])  # 2-D array
    path = tmp_path / "gray.png"
    Image.fromarray(gray, mode="L").save(path)
    r = insp.score(path)  # grayscale file
    assert r.score == pytest.approx(0.0, abs=1e-5)
    assert insp.score(Image.fromarray(gray, mode="L")).heatmap.shape == gray.shape


def test_k1_and_multiple_references(make_inspector):
    refs = [textured_image(seed=s) for s in range(3)]
    one = make_inspector().fit(refs[:1])
    many = make_inspector().fit(refs)
    assert one.n_references == 1 and many.n_references == 3
    assert len(many.bank) == 3 * len(one.bank)
    # adding references can only lower (or keep) the NN distance of any test image
    test = textured_image(seed=7)
    assert many.score(test).score <= one.score(test).score + 1e-6


def test_non_square_input(make_inspector):
    ref = textured_image(300, 200)
    insp = make_inspector(resolution=112).fit([ref])
    r = insp.score(ref)
    # smaller edge 200 -> 112 px, height 300 -> 168 px -> 12×8 patch grid
    assert r.patch_map.shape == (12, 8)
    assert r.heatmap.shape == (300, 200)
    assert r.fg_mask.shape == (12, 8)


def test_corrupt_and_missing_files_raise_clear_errors(make_inspector, tmp_path: Path):
    bad = tmp_path / "corrupt.png"
    bad.write_bytes(b"definitely not a png")
    insp = make_inspector()
    with pytest.raises(ImageLoadError, match=r"corrupt\.png"):
        insp.fit([bad])
    with pytest.raises(ImageLoadError, match="not found"):
        insp.fit([tmp_path / "missing.png"])


def test_score_before_fit_and_empty_fit(make_inspector):
    insp = make_inspector()
    with pytest.raises(RuntimeError, match="not fitted"):
        insp.score(textured_image())
    with pytest.raises(ValueError, match="at least one"):
        insp.fit([])


def test_masking_zeroes_background_and_scores(make_inspector):
    ref = textured_image()
    insp = make_inspector(masking="on").fit([ref])
    r = insp.score(paste_noise(ref, 98, 126, 42))
    assert insp.use_masking is True
    assert np.all(r.patch_map[~r.fg_mask] == 0)


def test_latency_is_recorded(make_inspector):
    insp = make_inspector().fit([textured_image()])
    assert insp.score(textured_image()).latency_ms > 0
