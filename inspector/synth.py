"""Seeded synthetic defects for cutoff calibration (parameters frozen in docs/calibration_protocol.md).

Every generator has the signature ``generator(image, placement, rng) -> (image, defect_mask)`` where
``image`` is ``H×W×3`` uint8, ``placement`` is a bool ``H×W`` region from which the defect centre is
drawn, and ``defect_mask`` is a bool ``H×W`` ground-truth mask. Sizes are relative to the shorter side.
"""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np

Generator = Callable[[np.ndarray, np.ndarray, np.random.Generator], tuple[np.ndarray, np.ndarray]]


def _pick_center(placement: np.ndarray, rng: np.random.Generator) -> tuple[int, int]:
    ys, xs = np.nonzero(placement)
    if ys.size == 0:
        h, w = placement.shape
        return int(rng.integers(0, h)), int(rng.integers(0, w))
    i = int(rng.integers(0, ys.size))
    return int(ys[i]), int(xs[i])


def _blend(image: np.ndarray, layer: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    a = alpha[..., None].astype(np.float32)
    return np.clip(image.astype(np.float32) * (1 - a) + layer.astype(np.float32) * a, 0, 255).astype(np.uint8)


def scratch(image: np.ndarray, placement: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Thin darker/brighter cubic Bézier stroke, slightly blurred."""
    h, w = image.shape[:2]
    s = min(h, w)
    cy, cx = _pick_center(placement, rng)
    length = rng.uniform(0.08, 0.30) * s
    theta = rng.uniform(0, 2 * np.pi)
    d = np.array([np.cos(theta), np.sin(theta)])
    n = np.array([-d[1], d[0]])
    p0 = np.array([cx, cy], dtype=np.float64) - d * length / 2
    p3 = p0 + d * length
    p1 = p0 + d * length / 3 + n * rng.uniform(-0.2, 0.2) * length
    p2 = p0 + 2 * d * length / 3 + n * rng.uniform(-0.2, 0.2) * length
    t = np.linspace(0, 1, 200)[:, None]
    pts = ((1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3)
    width = max(1, round(rng.uniform(1, 4) * s / 1024))
    stroke = np.zeros((h, w), np.uint8)
    cv2.polylines(stroke, [np.round(pts).astype(np.int32).reshape(-1, 1, 2)], False, 255, width, cv2.LINE_AA)
    region = cv2.dilate(stroke, np.ones((9, 9), np.uint8)) > 0
    local = image[region].mean(axis=0) if region.any() else image.reshape(-1, 3).mean(axis=0)
    delta = rng.uniform(40, 90) * (1 if rng.random() < 0.5 else -1)
    color = np.clip(local + delta, 0, 255)
    alpha = cv2.GaussianBlur(stroke.astype(np.float32) / 255.0, (0, 0), max(0.5, width / 3))
    layer = np.broadcast_to(color, image.shape)
    return _blend(image, layer, alpha), alpha > 0.25


def _color_jitter(patch: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-5, 5) / 2) % 180  # OpenCV hue is degrees / 2
    hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.9, 1.1), 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * rng.uniform(0.9, 1.1), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def _paste_rect(image: np.ndarray, placement: np.ndarray, rng: np.random.Generator, ph: int,
                pw: int) -> tuple[np.ndarray, np.ndarray]:
    """Cut a ph×pw patch, jitter + rotate it, paste it centred on a placement pixel."""
    h, w = image.shape[:2]
    ph, pw = int(np.clip(ph, 2, h - 1)), int(np.clip(pw, 2, w - 1))
    y0, x0 = int(rng.integers(0, h - ph + 1)), int(rng.integers(0, w - pw + 1))
    patch = _color_jitter(image[y0:y0 + ph, x0:x0 + pw].copy(), rng)
    cy, cx = _pick_center(placement, rng)
    angle = rng.uniform(-45, 45)
    rot = cv2.getRotationMatrix2D((pw / 2, ph / 2), angle, 1.0)
    rot[:, 2] += (cx - pw / 2, cy - ph / 2)  # translate patch centre onto (cx, cy)
    warped = cv2.warpAffine(patch, rot, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    mask = cv2.warpAffine(np.full((ph, pw), 255, np.uint8), rot, (w, h), flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT) > 0
    out = image.copy()
    out[mask] = warped[mask]
    return out, mask


def cutpaste(image: np.ndarray, placement: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """CutPaste: area 2–15 % of the image, aspect ratio log-U(0.3, 3.3), jittered + rotated."""
    h, w = image.shape[:2]
    area = rng.uniform(0.02, 0.15) * h * w
    aspect = float(np.exp(rng.uniform(np.log(0.3), np.log(3.3))))
    ph, pw = int(np.sqrt(area * aspect)), int(np.sqrt(area / aspect))
    return _paste_rect(image, placement, rng, ph, pw)


def cutpaste_scar(image: np.ndarray, placement: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """CutPaste-scar: thin long rectangle (width 2–16/256, length 10–25/256 of the shorter side)."""
    s = min(image.shape[:2])
    width = max(2, int(rng.uniform(2, 16) / 256 * s))
    length = max(4, int(rng.uniform(10, 25) / 256 * s))
    return _paste_rect(image, placement, rng, length, width)


def stain(image: np.ndarray, placement: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Irregular tinted blob with feathered edge, opacity 0.3–0.7."""
    h, w = image.shape[:2]
    s = min(h, w)
    radius = rng.uniform(0.03, 0.10) * s
    cy, cx = _pick_center(placement, rng)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / radius
    noise = cv2.resize(rng.normal(size=(8, 8)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    field = np.clip(1.0 - dist, 0, None) * (1.0 + 0.6 * noise)
    blob = (field > 0.35).astype(np.float32)
    if blob.sum() == 0:  # degenerate noise draw: fall back to the disc
        blob = (dist < 0.6).astype(np.float32)
    feather = cv2.GaussianBlur(blob, (0, 0), max(1.0, 0.15 * radius))
    opacity = rng.uniform(0.3, 0.7)
    tint = rng.uniform(0, 255, size=3)
    layer = np.broadcast_to(tint, image.shape)
    return _blend(image, layer, feather * opacity), feather > 0.5 * feather.max()


GENERATORS: dict[str, Generator] = {
    "scratch": scratch,
    "cutpaste": cutpaste,
    "cutpaste_scar": cutpaste_scar,
    "stain": stain,
}
GENERATOR_ORDER: tuple[str, ...] = tuple(GENERATORS)


def augment_good(image: np.ndarray, placement: np.ndarray, rng: np.random.Generator,
                 rotate: bool) -> tuple[np.ndarray, np.ndarray]:
    """Photometric jitter + small shift/rotation (nominal-variation proxy). Returns (image, placement)."""
    h, w = image.shape[:2]
    s = min(h, w)
    img = image.astype(np.float32)
    mean = img.mean()
    img = (img - mean) * rng.uniform(0.9, 1.1) + mean
    img = img * rng.uniform(0.9, 1.1)
    img = img + rng.normal(0, 2.0, size=img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    angle = rng.uniform(-10, 10) if rotate else 0.0
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    m[:, 2] += rng.uniform(-0.03, 0.03, size=2) * s
    img = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    place = cv2.warpAffine(placement.astype(np.uint8), m, (w, h), flags=cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_REFLECT_101) > 0
    return img, place


def make_defect(kind: str, image: np.ndarray, placement: np.ndarray,
                rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if kind not in GENERATORS:
        raise ValueError(f"unknown synthetic defect {kind!r}; choose from {sorted(GENERATORS)}")
    return GENERATORS[kind](image, placement, rng)
