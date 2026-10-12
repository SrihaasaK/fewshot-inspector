"""Evaluation metrics: image/pixel AUROC, AUPRO, and cutoff-dependent FRR / escape rate.

Conventions: higher score = more anomalous; an image is *flagged defective* iff ``score > cutoff``.
"""

from __future__ import annotations

from bisect import bisect

import numpy as np
import torch
from scipy.ndimage import label as connected_components
from sklearn.metrics import roc_auc_score


def image_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Image-level AUROC (labels: 1 = defective)."""
    return float(roc_auc_score(np.asarray(labels), np.asarray(scores)))


def pixel_auroc(scores: torch.Tensor | np.ndarray, labels: torch.Tensor | np.ndarray,
                device: str | torch.device | None = None) -> float:
    """Exact AUROC over all pixels pooled (ties count ½), as ``sklearn.roc_auc_score``.

    Implemented with one sort on ``device`` (Mann–Whitney U with tie groups) so that
    ~10⁸ full-resolution MVTec pixels per category stay fast.
    """
    s = torch.as_tensor(scores).reshape(-1)
    y = torch.as_tensor(labels).reshape(-1).bool()
    if device is not None:
        s, y = s.to(device), y.to(device)
    s = s.float()
    n_pos = int(y.sum())
    n_neg = y.numel() - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("pixel_auroc needs both positive and negative pixels")
    s_sorted, order = torch.sort(s)
    y_sorted = y[order]
    del order, s
    counts = torch.unique_consecutive(s_sorted, return_counts=True)[1]
    del s_sorted
    ends = torch.cumsum(counts, 0) - 1
    pos_at_end = torch.cumsum(y_sorted.to(torch.int64), 0)[ends]
    pos_in = torch.diff(pos_at_end, prepend=pos_at_end.new_zeros(1))
    neg_in = counts - pos_in
    neg_before = torch.cumsum(neg_in, 0) - neg_in
    twice_u = int((pos_in * (2 * neg_before + neg_in)).sum())
    return twice_u / (2.0 * n_pos * n_neg)


# ---------------------------------------------------------------------------------------------------
# `_trapezoid` and `aupro` are adapted from AnomalyDINO `src/post_eval.py` (Apache-2.0,
# LICENSES/Apache-2.0-AnomalyDINO.txt), which builds on the MVTec AD evaluation code
# (`generic_util.trapezoid`, `pro_curve_util.compute_pro`; Copyright 2022 MVTec Software GmbH,
# BSD-3-Clause, LICENSES/BSD-3-Clause-MVTec-AD-evaluation.txt). Modified: merged into one vectorised
# function, renamed variables, removed I/O and printing. See THIRD_PARTY_NOTICES.md.
# ---------------------------------------------------------------------------------------------------
def _trapezoid(x: np.ndarray, y: np.ndarray, x_max: float) -> float:
    """Area under (x, y) up to ``x_max`` (MVTec official evaluation code)."""
    correction = 0.0
    if x_max not in x:
        ins = bisect(x, x_max)
        y_interp = y[ins - 1] + (y[ins] - y[ins - 1]) * (x_max - x[ins - 1]) / (x[ins] - x[ins - 1])
        correction = 0.5 * (y_interp + y[ins - 1]) * (x_max - x[ins - 1])
    keep = x <= x_max
    x, y = x[keep], y[keep]
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1])) + correction)


def aupro(anomaly_maps: np.ndarray, gt_masks: np.ndarray, fpr_limit: float = 0.3) -> float:
    """Area under the per-region-overlap curve up to ``fpr_limit``, normalised to [0, 1].

    Port of the MVTec AD official ``compute_pro`` as used by AnomalyDINO (src/post_eval.py:169-290).
    ``anomaly_maps``/``gt_masks``: ``[N, H, W]``.
    """
    maps = np.asarray(anomaly_maps, dtype=np.float32)
    gts = np.asarray(gt_masks).astype(bool)
    structure = np.ones((3, 3), dtype=int)
    fp_changes = np.zeros(maps.shape, dtype=np.uint32)
    pro_changes = np.zeros(maps.shape, dtype=np.float64)
    num_ok = 0
    num_regions = 0
    for i, gt in enumerate(gts):
        labeled, n_comp = connected_components(gt, structure)
        num_regions += n_comp
        ok = labeled == 0
        num_ok += int(ok.sum())
        fp_changes[i][ok] = 1
        for c in range(1, n_comp + 1):
            region = labeled == c
            pro_changes[i][region] = 1.0 / region.sum()
    if num_regions == 0:
        raise ValueError("aupro needs at least one defect region")
    order = np.argsort(maps.ravel(), kind="stable")[::-1]
    scores_sorted = maps.ravel()[order]
    fprs = np.cumsum(fp_changes.ravel()[order]).astype(np.float64) / num_ok
    pros = np.cumsum(pro_changes.ravel()[order]) / num_regions
    keep = np.append(np.diff(scores_sorted) != 0, True)
    fprs = np.clip(fprs[keep], None, 1.0)
    pros = np.clip(pros[keep], None, 1.0)
    fprs = np.concatenate(([0.0], fprs, [1.0]))
    pros = np.concatenate(([0.0], pros, [1.0]))
    return _trapezoid(fprs, pros, fpr_limit) / fpr_limit


def false_reject_rate(good_scores: np.ndarray, cutoff: float) -> float:
    """Fraction of good parts flagged defective (``score > cutoff``)."""
    good_scores = np.asarray(good_scores)
    return float(np.mean(good_scores > cutoff)) if good_scores.size else float("nan")


def escape_rate(defect_scores: np.ndarray, cutoff: float) -> float:
    """Fraction of defective parts that pass (``score <= cutoff``)."""
    defect_scores = np.asarray(defect_scores)
    return float(np.mean(defect_scores <= cutoff)) if defect_scores.size else float("nan")


def cutoff_metrics(scores: np.ndarray, labels: np.ndarray, cutoff: float) -> dict[str, float]:
    """FRR, escape rate and balanced accuracy ``1 − (FRR + escape)/2`` at ``cutoff``."""
    scores, labels = np.asarray(scores), np.asarray(labels).astype(bool)
    frr = false_reject_rate(scores[~labels], cutoff)
    esc = escape_rate(scores[labels], cutoff)
    return {"frr": frr, "escape": esc, "balanced_accuracy": 1.0 - 0.5 * (frr + esc),
            "n_good": int((~labels).sum()), "n_defect": int(labels.sum())}
