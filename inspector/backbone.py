"""Frozen DINOv2 backbone: model loading, image decoding and preprocessing.

Preprocessing follows the reference code exactly (src/backbones.py:102-121):
PIL image → ``Resize(smaller edge, BICUBIC, antialias=True)`` → ``ToTensor`` →
ImageNet normalisation → crop bottom/right to a multiple of the patch size.
Tokens: ``get_intermediate_layers(x)[0]`` = last block, final LayerNorm, patch tokens only.
"""

from __future__ import annotations

import functools
import logging
import math
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

LOG = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

ImageLike = str | Path | np.ndarray | Image.Image


class ImageLoadError(ValueError):
    """Raised when an input cannot be decoded into an RGB image."""


class BackboneLoadError(RuntimeError):
    """Raised when the DINOv2 weights/code cannot be loaded."""


class PatchModel(Protocol):
    """Minimal interface of a DINOv2-like ViT (real model or test mock)."""

    patch_size: int
    embed_dim: int

    def get_intermediate_layers(self, x: torch.Tensor) -> Sequence[torch.Tensor]: ...

    def eval(self) -> PatchModel: ...


def available_cpus(cgroup_root: str | Path = "/sys/fs/cgroup") -> int:
    """CPUs this process may actually use: affinity mask capped by a cgroup CPU quota (v2 ``cpu.max`` or
    v1 ``cpu.cfs_quota_us``). Containers such as Streamlit Community Cloud often expose all host cores
    to ``os.cpu_count()`` but throttle to a 2-core quota; sizing torch's thread pool to the quota avoids
    oversubscription."""
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    root = Path(cgroup_root)
    quota: float | None = None
    try:
        v2 = root / "cpu.max"
        if v2.is_file():
            q, period = v2.read_text().split()[:2]
            quota = None if q == "max" else int(q) / int(period)
        elif (root / "cpu" / "cpu.cfs_quota_us").is_file():
            q = int((root / "cpu" / "cpu.cfs_quota_us").read_text())
            period = int((root / "cpu" / "cpu.cfs_period_us").read_text())
            quota = None if q <= 0 else q / period
    except (OSError, ValueError):
        quota = None
    return max(1, min(n, math.ceil(quota))) if quota else max(1, n)


def select_device(preference: str = "auto") -> torch.device:
    """Return ``preference`` as a device, or auto-select cuda → mps → cpu."""
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    """Block until queued kernels on ``device`` finished (for honest latency timing)."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


@functools.lru_cache(maxsize=4)
def _load_hub_model(model_name: str, device_str: str) -> torch.nn.Module:
    LOG.info("loading %s via torch.hub on %s", model_name, device_str)
    try:
        model = torch.hub.load("facebookresearch/dinov2", model_name, verbose=False, trust_repo=True)
    except Exception as exc:  # network / cache / hub failures all surface here
        raise BackboneLoadError(
            f"could not load DINOv2 '{model_name}' via torch.hub "
            "(needs network access on first use; weights are cached in ~/.cache/torch/hub): "
            f"{exc}"
        ) from exc
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(device_str)


def load_image(image: ImageLike) -> np.ndarray:
    """Decode ``image`` into an ``H×W×3`` uint8 RGB array.

    Grayscale inputs (MVTec grid/screw/zipper) are replicated to 3 channels,
    alpha is dropped, matching ``cv2.imread(IMREAD_COLOR)`` + BGR→RGB in the reference.
    """
    if isinstance(image, (str, Path)):
        path = Path(image)
        if not path.is_file():
            raise ImageLoadError(f"image file not found: {path}")
        try:
            with Image.open(path) as im:
                im.load()
                return np.asarray(im.convert("RGB"), dtype=np.uint8).copy()
        except (UnidentifiedImageError, OSError, SyntaxError) as exc:
            raise ImageLoadError(f"could not decode image {path}: {exc}") from exc
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    if isinstance(image, np.ndarray):
        arr = image
        if arr.ndim == 2:
            arr = np.repeat(arr[:, :, None], 3, axis=2)
        elif arr.ndim == 3 and arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        elif arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ImageLoadError(f"expected H×W, H×W×1, H×W×3 or H×W×4 array, got shape {image.shape}")
        if arr.dtype != np.uint8:
            if np.issubdtype(arr.dtype, np.floating) and arr.size and float(arr.max()) <= 1.0:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if min(arr.shape[:2]) < 1:
            raise ImageLoadError(f"empty image of shape {arr.shape}")
        return np.ascontiguousarray(arr)
    raise ImageLoadError(f"unsupported image type {type(image).__name__}")


class Preprocessor:
    """Resize (smaller edge) + normalise + crop to a multiple of the patch size."""

    def __init__(self, resolution: int, patch_size: int = 14) -> None:
        self.resolution = resolution
        self.patch_size = patch_size
        self._transform = transforms.Compose([
            transforms.Resize(size=resolution, interpolation=transforms.InterpolationMode.BICUBIC,
                              antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __call__(self, image_rgb: np.ndarray) -> torch.Tensor:
        """``H×W×3`` uint8 → ``3×h×w`` float tensor with h, w multiples of the patch size."""
        tensor = self._transform(Image.fromarray(image_rgb))
        h, w = tensor.shape[1:]
        ch, cw = h - h % self.patch_size, w - w % self.patch_size
        if ch == 0 or cw == 0:
            raise ImageLoadError(
                f"image {image_rgb.shape[:2]} resized to {(h, w)} is smaller than one {self.patch_size}px patch"
            )
        return tensor[:, :ch, :cw]


class Backbone:
    """Frozen patch-feature extractor returning raw ``[B, h, w, C]`` patch tokens.

    Args:
        model_name: DINOv2 hub entry (ignored when ``model`` is given).
        device: torch device.
        model: optional pre-built model (e.g. a mock in tests) exposing
            ``patch_size``, ``embed_dim`` and ``get_intermediate_layers``.
    """

    def __init__(self, model_name: str = "dinov2_vits14", device: torch.device | str = "cpu",
                 model: torch.nn.Module | None = None) -> None:
        self.device = torch.device(device)
        self.model_name = model_name if model is None else type(model).__name__
        self.model = model.eval().to(self.device) if model is not None else _load_hub_model(
            model_name, str(self.device))
        self.patch_size = int(self.model.patch_size)
        self.embed_dim = int(self.model.embed_dim)

    @torch.inference_mode()
    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        """``[B, 3, H, W]`` normalised images → raw patch tokens ``[B, H/p, W/p, C]``."""
        if batch.ndim != 4 or batch.shape[1] != 3:
            raise ValueError(f"expected [B, 3, H, W] batch, got {tuple(batch.shape)}")
        b, _, h, w = batch.shape
        gh, gw = h // self.patch_size, w // self.patch_size
        tokens = self.model.get_intermediate_layers(batch.to(self.device, non_blocking=True))[0]
        return tokens.reshape(b, gh, gw, -1).float()
