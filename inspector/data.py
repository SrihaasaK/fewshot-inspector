"""MVTec AD loading, seeded k-shot reference sampling and verified download.

Layout expected under ``root`` (the official tarball layout)::

    <root>/<category>/train/good/000.png
    <root>/<category>/test/{good,<defect>}/000.png
    <root>/<category>/ground_truth/<defect>/000_mask.png

MVTec AD is CC BY-NC-SA 4.0 (MVTec Software GmbH; Bergmann et al., CVPR 2019 / IJCV 2021).
It is downloaded by the user and never redistributed by this repository.
"""

from __future__ import annotations

import hashlib
import logging
import tarfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

LOG = logging.getLogger(__name__)

#: Category order used by AnomalyDINO (src/utils.py:79).
MVTEC_CATEGORIES: tuple[str, ...] = (
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut",
    "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
)
MVTEC_URL = ("https://www.mydrive.ch/shares/150996/b52ecdcbf521176e9db9c731f2304b27/download/"
             "420938113-1629960298/mvtec_anomaly_detection.tar.xz")
MVTEC_URL_FALLBACK = ("https://huggingface.co/datasets/micguida1/mvtech_anomaly_detection/resolve/"
                      "eb692b36147a3b3f12c561b879a062104c1f20b8/mvtec_anomaly_detection.tar.xz")
MVTEC_SHA256 = "cf4313b13603bec67abb49ca959488f7eedce2a9f7795ec54446c649ac98cd3d"
MVTEC_SIZE = 5_264_982_680
#: (train/good, test) image counts of the official release.
MVTEC_COUNTS: dict[str, tuple[int, int]] = {
    "bottle": (209, 83), "cable": (224, 150), "capsule": (219, 132), "carpet": (280, 117),
    "grid": (264, 78), "hazelnut": (391, 110), "leather": (245, 124), "metal_nut": (220, 115),
    "pill": (267, 167), "screw": (320, 160), "tile": (230, 117), "toothbrush": (60, 42),
    "transistor": (213, 100), "wood": (247, 79), "zipper": (240, 151),
}
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


class DataNotFoundError(FileNotFoundError):
    """Raised when the dataset (or part of it) is missing."""


@dataclass(frozen=True)
class TestSample:
    path: Path
    label: int  # 0 = good, 1 = defective
    defect_type: str
    mask_path: Path | None


@dataclass
class MVTecCategory:
    name: str
    root: Path
    train_paths: list[Path]
    test: list[TestSample] = field(default_factory=list)

    @property
    def labels(self) -> np.ndarray:
        return np.array([s.label for s in self.test], dtype=np.int64)


def _images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES and p.is_file())


def resolve_root(root: str | Path) -> Path:
    """Accept ``data/mvtec_ad`` or a parent containing ``mvtec_anomaly_detection/``."""
    root = Path(root)
    if (root / "bottle").is_dir() or any((root / c).is_dir() for c in MVTEC_CATEGORIES):
        return root
    nested = root / "mvtec_anomaly_detection"
    if nested.is_dir():
        return nested
    raise DataNotFoundError(
        f"MVTec AD not found under {root}. Run `python scripts/download_mvtec.py --dest {root}` "
        "(≈5.3 GB, CC BY-NC-SA 4.0)."
    )


def load_category(root: str | Path, category: str) -> MVTecCategory:
    """Index one category (paths only; images are decoded lazily)."""
    root = resolve_root(root)
    cat_dir = root / category
    train_dir, test_dir, gt_dir = cat_dir / "train" / "good", cat_dir / "test", cat_dir / "ground_truth"
    for d in (train_dir, test_dir):
        if not d.is_dir():
            raise DataNotFoundError(f"missing folder {d} (is the MVTec extraction complete?)")
    test: list[TestSample] = []
    for defect_dir in sorted(p for p in test_dir.iterdir() if p.is_dir()):
        defect = defect_dir.name
        for img in _images(defect_dir):
            if defect == "good":
                test.append(TestSample(img, 0, defect, None))
                continue
            mask = gt_dir / defect / f"{img.stem}_mask.png"
            if not mask.is_file():
                raise DataNotFoundError(f"missing ground-truth mask {mask}")
            test.append(TestSample(img, 1, defect, mask))
    train = _images(train_dir)
    if not train:
        raise DataNotFoundError(f"no training images in {train_dir}")
    return MVTecCategory(category, root, train, test)


def sample_references(train_paths: list[Path], k: int, seed: int, strategy: str = "contiguous") -> list[Path]:
    """Pick ``k`` nominal references for run ``seed``.

    ``contiguous`` (default, = AnomalyDINO src/detection.py:76): ``sorted(paths)[seed*k:(seed+1)*k]``.
    ``random``: ``k`` distinct images drawn with ``numpy.random.default_rng(seed)``.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    paths = sorted(train_paths)
    if strategy == "contiguous":
        chosen = paths[seed * k:(seed + 1) * k]
    elif strategy == "random":
        if k > len(paths):
            chosen = []
        else:
            idx = np.random.default_rng(seed).choice(len(paths), size=k, replace=False)
            chosen = [paths[i] for i in sorted(idx)]
    else:
        raise ValueError(f"unknown sampling strategy {strategy!r}")
    if len(chosen) < k:
        raise ValueError(f"only {len(paths)} train images: cannot draw k={k} references for seed={seed} "
                         f"with strategy={strategy!r}")
    return chosen


def load_mask(path: Path | None, shape: tuple[int, int]) -> np.ndarray:
    """Binary ground-truth mask (``> 0``); all-zero for good images."""
    if path is None:
        return np.zeros(shape, dtype=bool)
    mask = np.asarray(Image.open(path)) > 0
    if mask.ndim == 3:
        mask = mask.any(axis=2)
    if mask.shape != tuple(shape):
        raise ValueError(f"mask {path} has shape {mask.shape}, expected {shape}")
    return mask


def verify_mvtec(root: str | Path, categories: tuple[str, ...] = MVTEC_CATEGORIES) -> dict[str, tuple[int, int]]:
    """Check every category is present with the official image counts; returns the counts."""
    counts: dict[str, tuple[int, int]] = {}
    for cat in categories:
        c = load_category(root, cat)
        counts[cat] = (len(c.train_paths), len(c.test))
        expected = MVTEC_COUNTS.get(cat)
        if expected is not None and counts[cat] != expected:
            raise DataNotFoundError(f"{cat}: found (train, test) = {counts[cat]}, expected {expected}")
    return counts


def sha256sum(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    """Resumable HTTP download (Range requests)."""
    have = dest.stat().st_size if dest.exists() else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    with urllib.request.urlopen(req, timeout=60) as resp:  # fixed https URL
        mode = "ab" if have and resp.status == 206 else "wb"
        total = int(resp.headers.get("Content-Length", 0)) + (have if mode == "ab" else 0)
        with dest.open(mode) as f:
            done = have if mode == "ab" else 0
            last_log = done
            while block := resp.read(1 << 22):
                f.write(block)
                done += len(block)
                if done - last_log >= 1 << 29:
                    LOG.info("downloaded %.2f / %.2f GB", done / 1e9, total / 1e9)
                    last_log = done


def download_mvtec(dest: str | Path, keep_archive: bool = False) -> Path:
    """Download, checksum-verify and extract MVTec AD into ``dest``. Returns the data root."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / "mvtec_anomaly_detection.tar.xz"
    if not (archive.exists() and archive.stat().st_size == MVTEC_SIZE):
        for url in (MVTEC_URL, MVTEC_URL_FALLBACK):
            try:
                LOG.info("downloading MVTec AD from %s", url)
                _download(url, archive)
                break
            except OSError as exc:
                LOG.warning("download from %s failed: %s", url, exc)
        else:
            raise DataNotFoundError("all MVTec AD download routes failed; see docs/method_notes.md §5")
    LOG.info("verifying SHA-256 …")
    digest = sha256sum(archive)
    if digest != MVTEC_SHA256:
        raise DataNotFoundError(f"checksum mismatch for {archive}: {digest} != {MVTEC_SHA256}")
    LOG.info("extracting …")
    with tarfile.open(archive, "r:xz") as tar:
        tar.extractall(dest, filter="data")
    if not keep_archive:
        archive.unlink()
    root = resolve_root(dest)
    verify_mvtec(root)
    LOG.info("MVTec AD ready at %s", root)
    return root
