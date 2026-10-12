"""Component tests: each building block is checked against the reference implementation."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

from inspector import InspectorConfig, PatchBank
from inspector.data import DataNotFoundError, load_category, load_mask, sample_references
from inspector.inspector import patch_to_heatmap, rotate_image, top_pct_mean
from inspector.masking import automatic_masking_test, pca_foreground_mask, resolve_masking
from inspector.metrics import aupro, cutoff_metrics, pixel_auroc


# --------------------------------------------------------------------------- reference copies
# The `ref_*` oracles below are copied with minimal edits from AnomalyDINO (src/post_eval.py, src/utils.py,
# src/backbones.py; Copyright the AnomalyDINO authors, Apache-2.0 — LICENSES/Apache-2.0-AnomalyDINO.txt).
# Changes: stand-alone functions, type hints, svd_solver='full' for determinism. See THIRD_PARTY_NOTICES.md.
def ref_mean_top1p(distances: np.ndarray) -> float:  # AnomalyDINO src/post_eval.py:293-297
    if int(len(distances) * 0.01) == 0:
        return float(np.max(distances))
    return float(np.mean(sorted(distances.flatten(), reverse=True)[: int(len(distances) * 0.01)]))


def ref_dists2map(dists: np.ndarray, img_shape: tuple[int, int]) -> np.ndarray:  # src/utils.py:25-30
    dists = cv2.resize(dists, (img_shape[1], img_shape[0]), interpolation=cv2.INTER_LINEAR)
    return gaussian_filter(dists, sigma=4)


def ref_background_mask(feats: np.ndarray, grid: tuple[int, int], threshold: float = 10) -> np.ndarray:
    pc1 = PCA(n_components=1, svd_solver="full").fit_transform(feats.astype(np.float32))  # src/backbones.py:159
    mask = pc1 > threshold
    m = mask.reshape(grid)[int(grid[0] * 0.2):int(grid[0] * 0.8), int(grid[1] * 0.2):int(grid[1] * 0.8)]
    if m.sum() <= m.size * 0.35:
        mask = -pc1 > threshold
    mask = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)).astype(bool)
    return mask.squeeze()


# --------------------------------------------------------------------------- aggregation / heatmap
@pytest.mark.parametrize("n", [50, 1024, 2304])
def test_top_pct_matches_reference(n: int):
    d = np.random.default_rng(n).random(n).astype(np.float32)
    ours = float(top_pct_mean(torch.from_numpy(d)[None], 0.01)[0])
    assert ours == pytest.approx(ref_mean_top1p(d), rel=1e-6)


@pytest.mark.parametrize(("grid", "size"), [((32, 32), (900, 900)), ((48, 48), (1024, 1024)), ((12, 8), (300, 200))])
def test_heatmap_matches_cv2_resize_plus_scipy_gaussian(grid, size):
    d = np.random.default_rng(0).random(grid).astype(np.float32)
    ours = patch_to_heatmap(torch.from_numpy(d)[None], size, sigma=4.0)[0].numpy()
    ref = ref_dists2map(d, size)
    assert ours.shape == ref.shape
    assert np.abs(ours - ref).max() < 1e-5


def test_rotation_matches_reference_and_keeps_size():
    img = (np.random.default_rng(0).random((90, 120, 3)) * 255).astype(np.uint8)
    center = tuple(np.array(img.shape[1::-1]) / 2)
    rot = cv2.getRotationMatrix2D(center, 45, 1.0)
    ref = cv2.warpAffine(img, rot, img.shape[1::-1], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_DEFAULT)
    assert np.array_equal(rotate_image(img, 45), ref)
    assert rotate_image(img, 0) is img


# --------------------------------------------------------------------------- bank
@pytest.mark.parametrize("chunk", [2**27, 1000])
def test_bank_knn_matches_brute_force(chunk: int):
    rng = np.random.default_rng(0)
    bank_np, q_np = rng.normal(size=(500, 16)), rng.normal(size=(70, 16))
    bank = PatchBank(max_chunk_elems=chunk)
    bank.add(torch.from_numpy(bank_np[:300]))
    bank.add(torch.from_numpy(bank_np[300:]))
    ours = bank.knn_distance(torch.from_numpy(q_np)).numpy()
    bn = bank_np / np.linalg.norm(bank_np, axis=1, keepdims=True)
    qn = q_np / np.linalg.norm(q_np, axis=1, keepdims=True)
    ref = 1 - (qn @ bn.T).max(axis=1)
    assert np.allclose(ours, ref, atol=1e-5)


def test_bank_mask_and_empty():
    feats = torch.randn(4, 4, 8)
    mask = torch.zeros(4, 4, dtype=torch.bool)
    mask[0, 0] = True
    bank = PatchBank()
    bank.add(feats, mask)
    assert len(bank.build()) == 1
    with pytest.raises(ValueError, match="empty"):
        PatchBank().build()


# --------------------------------------------------------------------------- masking
def _object_features(h: int = 32, w: int = 32, c: int = 64, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    fg = (yy - h / 2) ** 2 + (xx - w / 2) ** 2 < (h / 3) ** 2
    direction = rng.normal(size=c)
    direction /= np.linalg.norm(direction)
    feats = rng.normal(size=(h, w, c)) + np.where(fg[..., None], 30.0, -12.0) * direction
    return feats.astype(np.float32), fg


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_pca_mask_matches_reference(seed: int):
    feats, fg = _object_features(seed=seed)
    ours = pca_foreground_mask(torch.from_numpy(feats))
    ref = ref_background_mask(feats.reshape(-1, feats.shape[-1]), feats.shape[:2]).reshape(feats.shape[:2])
    assert np.array_equal(ours, ref)
    assert (ours & fg).sum() >= 0.95 * fg.sum()  # recovers the object


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple MPS")
def test_pca_mask_on_mps_tensor_matches_cpu():
    feats, _ = _object_features()
    cpu = pca_foreground_mask(torch.from_numpy(feats))
    mps = pca_foreground_mask(torch.from_numpy(feats).to("mps"))
    assert cpu.any() and np.array_equal(cpu, mps)


def test_pca_mask_flips_when_center_is_background():
    feats, _ = _object_features()
    flipped = -feats  # PC1 sign reversed relative to fg
    assert pca_foreground_mask(torch.from_numpy(flipped))[16, 16]


def test_resolve_masking_modes():
    assert resolve_masking("on", None, None) is True
    assert resolve_masking("off", "screw", None) is False
    assert resolve_masking("auto", "screw", None) is True
    assert resolve_masking("auto", "bottle", None) is False
    _, fg = _object_features()
    assert resolve_masking("auto", "my_part", fg) is True
    assert automatic_masking_test(np.ones((32, 32), bool)) is False  # texture-like: fills the frame
    with pytest.raises(ValueError):
        resolve_masking("auto", "my_part", None)


# --------------------------------------------------------------------------- metrics
def test_pixel_auroc_matches_sklearn_with_ties():
    rng = np.random.default_rng(0)
    scores = np.round(rng.random(20000), 2).astype(np.float32)  # many ties
    labels = rng.random(20000) < 0.1 + 0.5 * scores
    assert pixel_auroc(scores, labels) == pytest.approx(roc_auc_score(labels, scores), abs=1e-9)
    with pytest.raises(ValueError):
        pixel_auroc(scores, np.zeros_like(labels))


def test_aupro_perfect_and_random():
    gts = np.zeros((2, 32, 32), bool)
    gts[0, 5:10, 5:10] = True
    perfect = gts.astype(np.float32) + 0.01 * np.random.default_rng(0).random(gts.shape).astype(np.float32)
    assert aupro(perfect, gts) > 0.99
    noise = np.random.default_rng(1).random(gts.shape).astype(np.float32)
    assert aupro(noise, gts) < 0.5


def test_cutoff_metrics():
    m = cutoff_metrics(np.array([0.1, 0.2, 0.9, 0.3, 0.8]), np.array([0, 0, 0, 1, 1]), cutoff=0.5)
    assert m["frr"] == pytest.approx(1 / 3)
    assert m["escape"] == pytest.approx(1 / 2)
    assert m["balanced_accuracy"] == pytest.approx(1 - (1 / 3 + 1 / 2) / 2)


# --------------------------------------------------------------------------- data
def test_sample_references_contiguous_matches_reference_slicing():
    paths = [Path(f"{i:03d}.png") for i in range(20)]
    assert sample_references(paths, 4, 0) == paths[0:4]
    assert sample_references(paths, 4, 2) == paths[8:12]
    r1 = sample_references(paths, 5, 3, strategy="random")
    assert r1 == sample_references(paths, 5, 3, strategy="random") and len(set(r1)) == 5
    with pytest.raises(ValueError, match="cannot draw"):
        sample_references(paths, 8, 2)


def _write_png(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def test_load_category_on_fake_tree(tmp_path: Path):
    img = np.zeros((20, 20, 3), np.uint8)
    for i in range(3):
        _write_png(tmp_path / "bottle/train/good" / f"{i:03d}.png", img)
    _write_png(tmp_path / "bottle/test/good/000.png", img)
    _write_png(tmp_path / "bottle/test/broken_large/000.png", img)
    mask = np.zeros((20, 20), np.uint8)
    mask[2:5, 2:5] = 255
    _write_png(tmp_path / "bottle/ground_truth/broken_large/000_mask.png", mask)
    cat = load_category(tmp_path, "bottle")
    assert len(cat.train_paths) == 3 and cat.labels.tolist() == [1, 0]
    assert load_mask(cat.test[0].mask_path, (20, 20)).sum() == 9
    assert not load_mask(None, (20, 20)).any()
    with pytest.raises(DataNotFoundError):
        load_category(tmp_path, "cable")


def test_config_validation():
    with pytest.raises(ValueError):
        InspectorConfig(masking="sometimes")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        InspectorConfig(top_pct=0)
    assert InspectorConfig().rotation_angles[1] == 45.0
