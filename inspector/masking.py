"""Zero-shot PCA foreground masking on DINOv2 patch tokens (AnomalyDINO, src/backbones.py:159-174).

Procedure, per image, on its *own raw* patch tokens:

1. first principal component score ``pc1`` of the (centred) tokens;
2. ``mask = pc1 > threshold`` (threshold 10);
3. if the centre crop ``[b·h:(1-b)·h, b·w:(1-b)·w]`` (b = 0.2) is ≤ 35 % foreground,
   flip: ``mask = -pc1 > threshold``;
4. 3×3 dilation, then 3×3 morphological closing.

Reference quirk (reproduced by default): the reference applies ``cv2.dilate`` /
``cv2.morphologyEx`` to the *flattened* ``N×1`` column of patch flags, so the 3×3 kernel acts as a
1-D 3-tap operation along raster order (horizontal neighbours, wrapping across row ends), not as 2-D
morphology on the patch grid. ``morphology="2d"`` gives the textbook 2-D variant.

Sign convention of ``pc1`` follows scikit-learn ≥ 1.5 ``PCA`` (largest-magnitude loading
positive); step 3 makes the result insensitive to it in the normal case.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
import torch

LOG = logging.getLogger(__name__)

#: Categories AnomalyDINO masks on MVTec AD (src/utils.py:101-116; paper Tab. Preprocessing_Default).
MVTEC_MASKED = frozenset({"capsule", "hazelnut", "pill", "screw", "toothbrush"})
MVTEC_TEXTURES = frozenset({"carpet", "grid", "leather", "tile", "wood"})
MVTEC_OBJECTS_NOT_MASKED = frozenset({"bottle", "cable", "metal_nut", "transistor", "zipper"})
MVTEC_CATEGORIES = tuple(sorted(MVTEC_MASKED | MVTEC_TEXTURES | MVTEC_OBJECTS_NOT_MASKED))

# A-priori parameters of the automatic masking test for non-MVTec categories (our addition;
# the paper decides by visual inspection). Chosen before looking at any result, never tuned.
AUTO_MIN_FG_FRACTION = 0.05
AUTO_MAX_FG_FRACTION = 0.85
AUTO_MAX_BORDER_FG_FRACTION = 0.30


def first_pc_scores(features: torch.Tensor) -> torch.Tensor:
    """Scores of the first principal component. ``features``: ``[N, C]`` → ``[N]`` (float32, CPU)."""
    # Move to CPU *before* casting: MPS has no float64 and `.to("cpu", float64)` from an MPS
    # tensor silently yields zeros on some torch builds.
    x = features.detach().cpu().double()
    x = x - x.mean(dim=0, keepdim=True)
    # Covariance eigendecomposition: C×C (384×384) is cheap and exact.
    cov = x.T @ x / max(x.shape[0] - 1, 1)
    _, eigvecs = torch.linalg.eigh(cov)
    v = eigvecs[:, -1]
    if v[v.abs().argmax()] < 0:  # sklearn>=1.5 svd_flip(u_based_decision=False) convention
        v = -v
    return (x @ v).float()


def pca_foreground_mask(features: torch.Tensor, threshold: float = 10.0, border: float = 0.2,
                        min_center_fraction: float = 0.35, kernel_size: int = 3,
                        morphology: str = "flat") -> np.ndarray:
    """Foreground mask of one image. ``features``: raw tokens ``[h, w, C]`` → bool ``[h, w]``.

    ``morphology="flat"`` reproduces the reference (morphology on the flattened N×1 vector);
    ``"2d"`` applies it on the ``h×w`` grid.
    """
    if features.ndim != 3:
        raise ValueError(f"expected [h, w, C] features, got {tuple(features.shape)}")
    if morphology not in ("flat", "2d"):
        raise ValueError(f"morphology must be 'flat' or '2d', got {morphology!r}")
    h, w, c = features.shape
    pc1 = first_pc_scores(features.reshape(h * w, c)).numpy().reshape(h, w)
    mask = pc1 > threshold
    center = mask[int(h * border):int(h * (1 - border)), int(w * border):int(w * (1 - border))]
    if center.sum() <= center.size * min_center_fraction:
        mask = -pc1 > threshold
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask_u8 = mask.astype(np.uint8).reshape(-1, 1) if morphology == "flat" else mask.astype(np.uint8)
    mask_u8 = cv2.dilate(mask_u8, kernel)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    return mask_u8.reshape(h, w).astype(bool)


def pca_foreground_masks(features: torch.Tensor, **kwargs: float) -> np.ndarray:
    """Batched :func:`pca_foreground_mask`: ``[B, h, w, C]`` → bool ``[B, h, w]``."""
    return np.stack([pca_foreground_mask(f, **kwargs) for f in features])  # type: ignore[arg-type]


def automatic_masking_test(mask: np.ndarray) -> bool:
    """Decide whether masking is trustworthy from the mask of the first reference image.

    Accept when the foreground is a plausible object: between 5 % and 85 % of the image and
    touching at most 30 % of the border patches (textures / failed masks fill the frame).
    """
    fg = float(mask.mean())
    border = np.concatenate([mask[0], mask[-1], mask[1:-1, 0], mask[1:-1, -1]])
    border_fg = float(border.mean()) if border.size else 1.0
    ok = AUTO_MIN_FG_FRACTION <= fg <= AUTO_MAX_FG_FRACTION and border_fg <= AUTO_MAX_BORDER_FG_FRACTION
    LOG.info("automatic masking test: fg=%.2f border_fg=%.2f -> %s", fg, border_fg, "mask" if ok else "no mask")
    return ok


def resolve_masking(mode: str, category: str | None, first_reference_mask: np.ndarray | None) -> bool:
    """Return whether test images should be masked.

    ``on``/``off`` force the decision. ``auto`` uses the paper's table for MVTec categories,
    otherwise :func:`automatic_masking_test` on the first reference image's mask.
    """
    if mode == "on":
        return True
    if mode == "off":
        return False
    if category is not None and category in MVTEC_CATEGORIES:
        return category in MVTEC_MASKED
    if first_reference_mask is None:
        raise ValueError("masking='auto' for a non-MVTec category needs the first reference mask")
    return automatic_masking_test(first_reference_mask)
