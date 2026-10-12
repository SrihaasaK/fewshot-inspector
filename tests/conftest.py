"""Shared fixtures. Tests never download DINOv2 weights: they inject :class:`MockViT`."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from inspector import Backbone, Inspector, InspectorConfig


class MockViT(nn.Module):
    """Tiny deterministic stand-in for DINOv2: patch embedding (14×14, stride 14) + tanh."""

    patch_size = 14
    embed_dim = 32

    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.proj = nn.Conv2d(3, self.embed_dim, kernel_size=14, stride=14)
        with torch.no_grad():
            self.proj.weight.copy_(torch.randn(self.proj.weight.shape, generator=gen) * 0.05)
            self.proj.bias.copy_(torch.randn(self.proj.bias.shape, generator=gen) * 0.1)

    def get_intermediate_layers(self, x: torch.Tensor) -> list[torch.Tensor]:
        tokens = torch.tanh(self.proj(x))  # [B, C, h, w]
        return [tokens.flatten(2).transpose(1, 2)]


def textured_image(h: int = 224, w: int = 224, seed: int = 0) -> np.ndarray:
    """Smooth, structured RGB image (stripes + gradient) — a 'good part'."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rng = np.random.default_rng(seed)
    phase = rng.uniform(0, np.pi)
    r = 127 + 80 * np.sin(xx / 9.0 + phase)
    g = 127 + 80 * np.cos(yy / 11.0)
    b = 60 + 120 * (xx + yy) / (h + w)
    return np.clip(np.stack([r, g, b], axis=2), 0, 255).astype(np.uint8)


@pytest.fixture
def backbone() -> Backbone:
    return Backbone(model=MockViT(), device="cpu")


@pytest.fixture
def make_inspector(backbone: Backbone):
    def _make(**overrides: object) -> Inspector:
        params = {"resolution": 224, "masking": "off", "rotate_references": False, "device": "cpu"}
        params.update(overrides)
        return Inspector(InspectorConfig(**params), backbone=backbone)  # type: ignore[arg-type]

    return _make
