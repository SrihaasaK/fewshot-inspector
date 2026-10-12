"""Heatmap overlays and failure-case grids (matplotlib is imported lazily)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np


def normalize_map(heatmap: np.ndarray, vmin: float | None = None, vmax: float | None = None) -> np.ndarray:
    """Scale a heatmap to [0, 1] using ``vmin``/``vmax`` (defaults: its own min/max)."""
    lo = float(np.min(heatmap)) if vmin is None else vmin
    hi = float(np.max(heatmap)) if vmax is None else vmax
    return np.clip((heatmap - lo) / max(hi - lo, 1e-12), 0.0, 1.0)


def overlay(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5, vmin: float | None = None,
            vmax: float | None = None) -> np.ndarray:
    """Blend a JET-coloured heatmap onto an RGB uint8 image (same H×W)."""
    if heatmap.shape != image.shape[:2]:
        heatmap = cv2.resize(heatmap.astype(np.float32), (image.shape[1], image.shape[0]))
    colored = cv2.applyColorMap((normalize_map(heatmap, vmin, vmax) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return (alpha * colored + (1 - alpha) * image).astype(np.uint8)


def contour(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int] = (0, 255, 0),
            thickness: int = 3) -> np.ndarray:
    """Draw the outline of a binary mask on a copy of ``image``."""
    out = image.copy()
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, thickness)
    return out


def save_grid(rows: Sequence[Sequence[np.ndarray]], titles: Sequence[Sequence[str]], path: str | Path,
              suptitle: str | None = None, cell: float = 2.6) -> Path:
    """Save a grid of RGB images (rows × cols) with per-cell titles."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_rows, n_cols = len(rows), max(len(r) for r in rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(cell * n_cols, cell * n_rows + (0.4 if suptitle else 0)),
                             squeeze=False)
    for i, row in enumerate(rows):
        for j in range(n_cols):
            ax = axes[i][j]
            ax.axis("off")
            if j < len(row):
                ax.imshow(row[j])
                ax.set_title(titles[i][j], fontsize=8)
    if suptitle:
        fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
