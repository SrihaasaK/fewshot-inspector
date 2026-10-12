"""Choose a deployable pass/fail cutoff from the k good references only (docs/calibration_protocol.md).

``build_calibration_set`` scores augmented good copies and synthetic-defect copies of each reference
against a leave-one-out bank (k ≥ 2) or the single reference (k = 1). Policies turn those scores into
a cutoff; an image is flagged defective iff ``score > cutoff``.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch

from inspector.backbone import Backbone, ImageLike, load_image, synchronize
from inspector.config import InspectorConfig
from inspector.inspector import Inspector, top_pct_mean
from inspector.synth import GENERATOR_ORDER, augment_good, make_defect

LOG = logging.getLogger(__name__)

GOOD_ONLY_MARGIN = 0.10
GOOD_ONLY_QUANTILE = 0.99
MAX_SYNTH_ESCAPE = 0.05
CALIBRATION_SET_SIZE = 64


@dataclass
class CalibrationSet:
    """Calibration scores of one (references, config) setup."""

    good_scores: np.ndarray
    defect_scores: np.ndarray
    defect_kinds: list[str]
    k: int
    leave_one_out: bool
    examples: list[dict[str, np.ndarray]] = field(default_factory=list)

    @property
    def auroc(self) -> float:
        from inspector.metrics import image_auroc

        labels = np.r_[np.zeros(len(self.good_scores)), np.ones(len(self.defect_scores))]
        return image_auroc(labels, np.r_[self.good_scores, self.defect_scores])


# ----------------------------------------------------------------------------- policies
def youden_cutoff(good: np.ndarray, defect: np.ndarray) -> float:
    """Cutoff maximising TPR − FPR (flag iff score > cutoff); ties → median maximiser."""
    good, defect = np.asarray(good, float), np.asarray(defect, float)
    pooled = np.unique(np.r_[good, defect])
    eps = max(1e-12, 1e-9 * float(np.abs(pooled).max(initial=1.0)))
    cands = np.r_[pooled[0] - eps, (pooled[:-1] + pooled[1:]) / 2, pooled[-1] + eps]
    tpr = (defect[None, :] > cands[:, None]).mean(axis=1)
    fpr = (good[None, :] > cands[:, None]).mean(axis=1)
    j = tpr - fpr
    best = np.flatnonzero(np.isclose(j, j.max(), rtol=0, atol=1e-12))
    return float(cands[best[len(best) // 2]])


def good_only_cutoff(good: np.ndarray, margin: float = GOOD_ONLY_MARGIN, q: float = GOOD_ONLY_QUANTILE) -> float:
    return float(np.quantile(np.asarray(good, float), q) * (1.0 + margin))


def escape_cutoff(defect: np.ndarray, max_escape: float = MAX_SYNTH_ESCAPE) -> float:
    """Highest cutoff letting at most ⌊max_escape · n⌋ synthetic defects pass (score ≤ cutoff)."""
    d = np.sort(np.asarray(defect, float))
    m = math.floor(max_escape * len(d))
    return float(np.nextafter(d[m], -np.inf))


DEPLOYABLE_POLICIES = ("good_only", "good_only_m0", "synth_youden", "synth_escape5")
ALL_POLICIES = (*DEPLOYABLE_POLICIES, "oracle")


def cutoff_for(policy: str, cal: CalibrationSet) -> float:
    if policy == "good_only":
        return good_only_cutoff(cal.good_scores)
    if policy == "good_only_m0":
        return good_only_cutoff(cal.good_scores, margin=0.0)
    if policy == "synth_youden":
        return youden_cutoff(cal.good_scores, cal.defect_scores)
    if policy == "synth_escape5":
        return escape_cutoff(cal.defect_scores)
    if policy == "oracle":
        raise ValueError("oracle needs real test labels: use youden_cutoff(test_good, test_defect)")
    raise ValueError(f"unknown policy {policy!r}; choose from {ALL_POLICIES}")


# ----------------------------------------------------------------------------- calibration set
def _placement(insp: Inspector, features: torch.Tensor, shape: tuple[int, int]) -> np.ndarray:
    """Pixel-level placement region: the inspector's fg mask if masking is on, else everything."""
    if not insp.use_masking:
        return np.ones(shape, dtype=bool)
    grid = insp.foreground_mask(features).astype(np.uint8)
    return cv2.resize(grid, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0


@torch.inference_mode()
def _score_arrays(insp: Inspector, images: Sequence[np.ndarray]) -> np.ndarray:
    feats = insp.extract_features(images)
    scores = []
    for f in feats:
        dist, _ = insp.patch_distances(f.unsqueeze(0))
        scores.append(float(top_pct_mean(dist, insp.config.top_pct)[0]))
    return np.array(scores)


def build_calibration_set(config: InspectorConfig, backbone: Backbone, references: Sequence[ImageLike],
                          seed: int = 0, n_total: int = CALIBRATION_SET_SIZE,
                          reference_views: Sequence[Sequence[torch.Tensor]] | None = None,
                          keep_examples: int = 0) -> CalibrationSet:
    """Synthetic-good and synthetic-defect scores from the references alone.

    Args:
        config: deployed inspector configuration.
        backbone: shared backbone.
        references: the k good images.
        seed: run seed (combined with reference index and k).
        n_total: target calibration-set size per class (n = ⌈n_total / k⌉ copies per reference).
        reference_views: optional precomputed features of each reference's rotation views.
        keep_examples: keep this many (good, defect, mask, heatmap-ready) examples for display.
    """
    refs = [load_image(r) for r in references]
    k = len(refs)
    if k == 0:
        raise ValueError("calibration needs at least one reference image")
    probe = Inspector(config, backbone=backbone)
    views = list(reference_views) if reference_views is not None else [
        probe.extract_features(probe.reference_views(r)) for r in refs]
    n_per_ref = math.ceil(n_total / k)
    goods, defects, kinds, examples = [], [], [], []
    for i, ref in enumerate(refs):
        bank_ids = [j for j in range(k) if j != i] if k >= 2 else [0]
        insp = Inspector(config, backbone=backbone)
        insp.fit_features([v for j in bank_ids for v in views[j]], first_reference=views[bank_ids[0]][0])
        place = _placement(insp, views[i][0], ref.shape[:2])
        rng = np.random.default_rng([seed, i, k])
        good_imgs, defect_imgs = [], []
        for j in range(n_per_ref):
            g, g_place = augment_good(ref, place, rng, rotate=config.rotate_references)
            kind = GENERATOR_ORDER[(i * n_per_ref + j) % len(GENERATOR_ORDER)]
            d, d_mask = make_defect(kind, g, g_place, rng)
            good_imgs.append(g)
            defect_imgs.append(d)
            kinds.append(kind)
            if len(examples) < keep_examples:
                examples.append({"good": g, "defect": d, "mask": d_mask, "kind": np.array(kind)})
        goods.append(_score_arrays(insp, good_imgs))
        defects.append(_score_arrays(insp, defect_imgs))
    cal = CalibrationSet(np.concatenate(goods), np.concatenate(defects), kinds, k, k >= 2, examples)
    LOG.info("calibration set: k=%d LOO=%s goods=%d defects=%d synth AUROC=%.3f", k, cal.leave_one_out,
             len(cal.good_scores), len(cal.defect_scores), cal.auroc)
    return cal


def build_calibrated_inspector(config: InspectorConfig, backbone: Backbone, references: Sequence[ImageLike],
                               n_total: int = CALIBRATION_SET_SIZE, seed: int = 0, keep_examples: int = 0,
                               ) -> tuple[Inspector, CalibrationSet, dict[str, float], dict[str, float]]:
    """Fit an inspector and calibrate every deployable cutoff, sharing reference features.

    Returns ``(inspector, calibration_set, cutoffs_by_policy, timings_s)``; this is the app's setup path.
    """
    t0 = time.perf_counter()
    refs = [load_image(r) for r in references]
    if not refs:
        raise ValueError("need at least one reference image")
    insp = Inspector(config, backbone=backbone)
    views = [insp.extract_features(insp.reference_views(r)) for r in refs]
    insp.fit_features([v for vs in views for v in vs], first_reference=views[0][0])
    insp.n_references = len(refs)
    synchronize(insp.device)
    t1 = time.perf_counter()
    cal = build_calibration_set(config, backbone, refs, seed=seed, n_total=n_total, reference_views=views,
                                keep_examples=keep_examples)
    cutoffs = {p: cutoff_for(p, cal) for p in DEPLOYABLE_POLICIES}
    t2 = time.perf_counter()
    insp.fit_seconds = t1 - t0
    return insp, cal, cutoffs, {"fit_s": t1 - t0, "calibration_s": t2 - t1, "total_s": t2 - t0}


def calibrate(inspector: Inspector, references: Sequence[ImageLike], policy: str = "synth_youden",
              seed: int = 0, keep_examples: int = 0) -> tuple[float, CalibrationSet]:
    """Convenience for applications: calibration set + cutoff for a fitted-style inspector config."""
    cal = build_calibration_set(inspector.config, inspector.backbone, references, seed=seed,
                                keep_examples=keep_examples)
    return cutoff_for(policy, cal), cal
