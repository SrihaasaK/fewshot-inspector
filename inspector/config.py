"""Configuration for :class:`inspector.Inspector`.

Every default mirrors AnomalyDINO (Damm et al., WACV 2025); see ``docs/method_notes.md``
for the file:line / paper-line citation of each value.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

MaskingMode = Literal["auto", "on", "off"]
DTypeName = Literal["auto", "float32", "float16"]

#: Rotation angles of the `agnostic` reference augmentation (src/utils.py:7).
DEFAULT_ROTATIONS: tuple[float, ...] = (0, 45, 90, 135, 180, 225, 270, 315)


@dataclass
class InspectorConfig:
    """Hyper-parameters of the inspector. Defaults = AnomalyDINO defaults.

    Attributes:
        model_name: torch.hub entry of ``facebookresearch/dinov2``.
        resolution: target length of the *smaller* image edge (multiple of 14 recommended).
            448 is the reference-code default; the 96.6 % MVTec result uses 672.
        masking: ``"on"``/``"off"`` force PCA foreground masking of test images;
            ``"auto"`` uses the paper's per-category table for MVTec categories and an
            automatic masking test on the first reference image otherwise.
        category: optional MVTec category name, used by ``masking="auto"``.
        rotate_references: augment each reference with :attr:`rotation_angles` (``agnostic``).
        rotation_angles: rotation angles in degrees (the 0 entry keeps the original).
        mask_references: also drop background patches of references (reference default: no).
        mask_threshold: threshold on the first principal component.
        mask_border: centre crop margin used to decide the sign of the first PC.
        mask_min_center_fraction: minimum foreground fraction of the centre crop.
        mask_kernel: kernel size of the dilation + closing post-processing.
        mask_morphology: ``"flat"`` = reference behaviour (morphology on the flattened N×1 patch
            vector, i.e. 1-D along raster order); ``"2d"`` = morphology on the patch grid.
        top_pct: fraction of highest patch distances averaged into the image score.
        sigma: Gaussian smoothing (pixels at original resolution) of the heatmap.
        device: ``"auto"`` (cuda → mps → cpu) or an explicit torch device string.
        knn_dtype: matmul dtype for nearest-neighbour search; ``"auto"`` = fp16 on CUDA, else fp32.
        knn_backend: ``"torch"`` (chunked matmul) or ``"faiss"`` (if installed).
        batch_size: images per backbone forward pass.
    """

    model_name: str = "dinov2_vits14"
    resolution: int = 448
    masking: MaskingMode = "auto"
    category: str | None = None
    rotate_references: bool = True
    rotation_angles: tuple[float, ...] = field(default_factory=lambda: DEFAULT_ROTATIONS)
    mask_references: bool = False
    mask_threshold: float = 10.0
    mask_border: float = 0.2
    mask_min_center_fraction: float = 0.35
    mask_kernel: int = 3
    mask_morphology: Literal["flat", "2d"] = "flat"
    top_pct: float = 0.01
    sigma: float = 4.0
    device: str = "auto"
    knn_dtype: DTypeName = "auto"
    knn_backend: Literal["torch", "faiss"] = "torch"
    batch_size: int = 8

    def __post_init__(self) -> None:
        if self.resolution < 14:
            raise ValueError(f"resolution must be >= 14 (one DINOv2 patch), got {self.resolution}")
        if self.masking not in ("auto", "on", "off"):
            raise ValueError(f"masking must be 'auto', 'on' or 'off', got {self.masking!r}")
        if not 0.0 < self.top_pct <= 1.0:
            raise ValueError(f"top_pct must be in (0, 1], got {self.top_pct}")
        if self.sigma < 0:
            raise ValueError(f"sigma must be >= 0, got {self.sigma}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        self.rotation_angles = tuple(float(a) for a in self.rotation_angles)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
