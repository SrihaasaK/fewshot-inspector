"""Training-free few-shot inspector (AnomalyDINO reimplementation).

``fit`` builds a memory bank of DINOv2 patch tokens from 1–16 nominal images (+ rotations);
``score`` measures each test patch's cosine distance to its nearest nominal patch, aggregates the
top 1 % into an image score and turns the patch grid into a full-resolution heatmap
(bilinear upsampling + Gaussian σ = 4). See ``docs/method_notes.md`` for every default.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from inspector.backbone import Backbone, ImageLike, Preprocessor, load_image, select_device, synchronize
from inspector.bank import PatchBank, resolve_dtype
from inspector.config import InspectorConfig
from inspector.masking import MVTEC_CATEGORIES, pca_foreground_mask, resolve_masking

LOG = logging.getLogger(__name__)


@dataclass
class InspectionResult:
    """Output of :meth:`Inspector.score`.

    Attributes:
        score: image-level anomaly score (mean of the top ``top_pct`` patch distances).
        patch_map: ``[h, w]`` cosine distance per patch (0 for masked-out background).
        heatmap: ``[H, W]`` anomaly map at the original image resolution.
        fg_mask: ``[h, w]`` patches considered foreground (all True without masking).
        latency_ms: wall time from raw input to score + heatmap (device-synchronised).
    """

    score: float
    patch_map: np.ndarray
    heatmap: np.ndarray
    fg_mask: np.ndarray
    latency_ms: float


@dataclass
class Prediction:
    """Pass/fail decision of :meth:`Inspector.predict`."""

    defective: bool
    margin: float  # score − cutoff (positive ⇒ defective)
    cutoff: float
    result: InspectionResult


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate about the centre, same size, reflect-101 border (reference src/utils.py:18-22)."""
    if angle % 360 == 0:
        return image
    h, w = image.shape[:2]
    rot = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(image, rot, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_DEFAULT)


def top_pct_mean(patch_maps: torch.Tensor, top_pct: float = 0.01) -> torch.Tensor:
    """Mean of the ``top_pct`` highest values per image. ``[B, ...]`` → ``[B]``.

    Count is ``int(n · top_pct)`` (floor), falling back to the max when that is 0
    (reference ``mean_top1p``, src/post_eval.py:293-297). Includes zeroed background patches.
    """
    flat = patch_maps.reshape(patch_maps.shape[0], -1)
    k = max(1, int(flat.shape[1] * top_pct))
    return flat.topk(k, dim=1).values.mean(dim=1)


def _reflect_indices(n: int, radius: int, device: torch.device) -> torch.Tensor:
    """Indices implementing scipy's ``mode='reflect'`` (half-sample symmetric) padding."""
    idx = torch.arange(-radius, n + radius, device=device)
    period = 2 * n
    m = torch.remainder(idx, period)
    return torch.where(m < n, m, period - 1 - m)


def gaussian_blur(maps: torch.Tensor, sigma: float, truncate: float = 4.0) -> torch.Tensor:
    """Separable Gaussian blur of ``[B, H, W]`` maps, numerically equal to
    ``scipy.ndimage.gaussian_filter(sigma, truncate=4.0, mode='reflect')`` per map."""
    if sigma <= 0:
        return maps
    radius = int(truncate * sigma + 0.5)
    x = torch.arange(-radius, radius + 1, device=maps.device, dtype=torch.float32)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()
    _, h, w = maps.shape
    out = maps.float()
    out = out.index_select(2, _reflect_indices(w, radius, maps.device))
    out = F.conv2d(out.unsqueeze(1), kernel.view(1, 1, 1, -1)).squeeze(1)
    out = out.index_select(1, _reflect_indices(h, radius, maps.device))
    out = F.conv2d(out.unsqueeze(1), kernel.view(1, 1, -1, 1)).squeeze(1)
    return out


def patch_to_heatmap(patch_maps: torch.Tensor, size: tuple[int, int], sigma: float = 4.0) -> torch.Tensor:
    """Patch grid ``[B, h, w]`` → bilinear upsample to ``size=(H, W)`` → Gaussian σ.

    Matches the reference ``dists2map`` (``cv2.resize(INTER_LINEAR)`` + ``gaussian_filter``).
    """
    up = F.interpolate(patch_maps.float().unsqueeze(1), size=size, mode="bilinear", align_corners=False)
    return gaussian_blur(up.squeeze(1), sigma)


class Inspector:
    """Few-shot visual inspector.

    Example:
        >>> insp = Inspector(InspectorConfig(category="bottle")).fit(good_images)
        >>> result = insp.score("test.png"); result.score, result.heatmap.shape

    Args:
        config: hyper-parameters (defaults = AnomalyDINO).
        backbone: optional pre-built :class:`Backbone` (tests inject a mock model this way).
    """

    def __init__(self, config: InspectorConfig | None = None, backbone: Backbone | None = None) -> None:
        self.config = config or InspectorConfig()
        if backbone is None:
            backbone = Backbone(self.config.model_name, select_device(self.config.device))
        self.backbone = backbone
        self.device = backbone.device
        self.preprocess = Preprocessor(self.config.resolution, backbone.patch_size)
        self.knn_dtype = resolve_dtype(self.config.knn_dtype, self.device)
        self.bank: PatchBank | None = None
        self.use_masking: bool | None = None
        self.n_references = 0
        self.fit_seconds = 0.0

    # ------------------------------------------------------------------ features
    @torch.inference_mode()
    def extract_features(self, images: Sequence[np.ndarray]) -> list[torch.Tensor]:
        """Raw patch tokens ``[h, w, C]`` for each RGB uint8 image (batched by tensor shape)."""
        tensors = [self.preprocess(img) for img in images]
        out: list[torch.Tensor | None] = [None] * len(tensors)
        groups: dict[tuple[int, ...], list[int]] = {}
        for i, t in enumerate(tensors):
            groups.setdefault(tuple(t.shape), []).append(i)
        bs = self.config.batch_size
        for idxs in groups.values():
            for s in range(0, len(idxs), bs):
                chunk = idxs[s:s + bs]
                feats = self.backbone(torch.stack([tensors[i] for i in chunk]))
                for i, f in zip(chunk, feats, strict=True):
                    out[i] = f
        return [f for f in out if f is not None]

    def foreground_mask(self, features: torch.Tensor) -> np.ndarray:
        cfg = self.config
        return pca_foreground_mask(features, threshold=cfg.mask_threshold, border=cfg.mask_border,
                                   min_center_fraction=cfg.mask_min_center_fraction,
                                   kernel_size=cfg.mask_kernel, morphology=cfg.mask_morphology)

    def reference_views(self, image: np.ndarray) -> list[np.ndarray]:
        if not self.config.rotate_references:
            return [image]
        return [rotate_image(image, a) for a in self.config.rotation_angles]

    # ------------------------------------------------------------------ fit
    def fit(self, images: Sequence[ImageLike]) -> Inspector:
        """Build the nominal memory bank from 1+ good images. Returns ``self``."""
        if len(images) == 0:
            raise ValueError("fit() needs at least one reference image")
        start = time.perf_counter()
        refs = [load_image(im) for im in images]
        views = [v for ref in refs for v in self.reference_views(ref)]
        feats = self.extract_features(views)
        self.fit_features(feats, first_reference=feats[0])
        synchronize(self.device)
        self.n_references = len(refs)
        self.fit_seconds = time.perf_counter() - start
        LOG.info("fitted on %d refs (%d views, %d patches, masking=%s) in %.2fs", len(refs), len(views),
                 len(self.bank or []), self.use_masking, self.fit_seconds)
        return self

    def fit_features(self, features: Sequence[torch.Tensor], first_reference: torch.Tensor | None = None) -> None:
        """Build the bank from pre-extracted raw tokens (used by the cached benchmark)."""
        cfg = self.config
        first = first_reference if first_reference is not None else features[0]
        needs_mask = cfg.masking == "auto" and cfg.category not in MVTEC_CATEGORIES
        self.use_masking = resolve_masking(cfg.masking, cfg.category,
                                           self.foreground_mask(first) if needs_mask else None)
        bank = PatchBank(self.device, self.knn_dtype, backend=cfg.knn_backend)
        for f in features:
            mask = torch.from_numpy(self.foreground_mask(f)) if (cfg.mask_references and self.use_masking) else None
            bank.add(f, mask)
        self.bank = bank.build()

    # ------------------------------------------------------------------ score
    def _require_fit(self) -> PatchBank:
        if self.bank is None or self.use_masking is None:
            raise RuntimeError("Inspector is not fitted; call fit(good_images) first")
        return self.bank

    @torch.inference_mode()
    def patch_distances(self, features: torch.Tensor,
                        fg_masks: np.ndarray | None = None) -> tuple[torch.Tensor, np.ndarray]:
        """``[B, h, w, C]`` raw tokens → (distances ``[B, h, w]`` with background zeroed, fg masks).

        ``fg_masks`` lets callers reuse precomputed test masks (they depend only on the test image).
        """
        bank = self._require_fit()
        dist = bank.knn_distance(features)
        if self.use_masking:
            fg = fg_masks if fg_masks is not None else np.stack([self.foreground_mask(f) for f in features])
            dist = dist * torch.from_numpy(fg).to(dist.device, dist.dtype)
        else:
            fg = np.ones(dist.shape, dtype=bool)
        return dist, fg

    @torch.inference_mode()
    def score_batch(self, images: Sequence[ImageLike], heatmaps: bool = True) -> list[InspectionResult]:
        """Score several images. ``latency_ms`` is the batch wall time divided by batch size."""
        self._require_fit()
        if len(images) == 0:
            return []
        start = time.perf_counter()
        arrays = [load_image(im) for im in images]
        feats = self.extract_features(arrays)
        results: list[InspectionResult] = []
        for img, f in zip(arrays, feats, strict=True):
            dist, fg = self.patch_distances(f.unsqueeze(0))
            score = top_pct_mean(dist, self.config.top_pct)
            hm = (patch_to_heatmap(dist, img.shape[:2], self.config.sigma) if heatmaps
                  else torch.zeros((1, 0, 0), device=dist.device))
            results.append(InspectionResult(
                score=float(score[0]), patch_map=dist[0].cpu().numpy(), heatmap=hm[0].cpu().numpy(),
                fg_mask=fg[0], latency_ms=math.nan))
        synchronize(self.device)
        per_image_ms = (time.perf_counter() - start) * 1000.0 / len(arrays)
        for r in results:
            r.latency_ms = per_image_ms
        return results

    def score(self, image: ImageLike) -> InspectionResult:
        """Score one image (score, patch map, full-resolution heatmap, fg mask, latency)."""
        return self.score_batch([image])[0]

    def predict(self, image: ImageLike, cutoff: float) -> Prediction:
        """PASS/FAIL decision: defective iff ``score > cutoff``."""
        result = self.score(image)
        return Prediction(defective=result.score > cutoff, margin=result.score - cutoff, cutoff=cutoff,
                          result=result)
