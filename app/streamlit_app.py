"""fewshot-inspector demo: learn "normal" from 1–16 good photos, calibrate a cutoff with synthetic defects,
then inspect new images (PASS/FAIL + heatmap).

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402

from inspector import Backbone, ImageLoadError, Inspector, InspectorConfig, load_image, select_device  # noqa: E402
from inspector.calibrate import CalibrationSet, build_calibrated_inspector  # noqa: E402
from inspector.data import MVTEC_CATEGORIES, load_category  # noqa: E402
from inspector.viz import normalize_map, overlay  # noqa: E402

DATA_ROOT = ROOT / "data" / "mvtec_ad"
#: First entry = default. `synth_escape5` was selected on the dev categories (docs/calibration_protocol.md).
POLICIES = {
    "synth_escape5": "Synthetic escape ≤ 5 % (catch ≥ 95 % of synthetic defects) — recommended",
    "synth_youden": "Synthetic Youden (balance FRR vs escapes on synthetic defects)",
    "good_only": "Good-only p99 × 1.10 (needs no defects at all)",
}
#: CPU defaults keep "4 references → calibrated inspector" well under 60 s even on slow cloud CPUs
#: (measured by scripts/time_app_setup.py; see README).
CPU_DEFAULT_RESOLUTION = 448
GPU_DEFAULT_RESOLUTION = 448
CPU_DEFAULT_CAL = 32
GPU_DEFAULT_CAL = 64

st.set_page_config(page_title="fewshot-inspector", page_icon="🔍", layout="wide")


@dataclass
class Session:
    inspector: Inspector
    calibration: CalibrationSet
    cutoffs: dict[str, float]
    setup_s: float
    fit_s: float
    n_refs: int
    signature: tuple


@st.cache_resource(show_spinner="Loading DINOv2 ViT-S/14 (first run downloads ~90 MB)…")
def get_backbone(device: str) -> Backbone:
    return Backbone("dinov2_vits14", device)


def read_uploads(files: list) -> list[tuple[np.ndarray, str]]:
    """Decode uploads; unreadable files are reported and skipped (never crash the app)."""
    from PIL import Image

    images = []
    for f in files:
        name = getattr(f, "name", "upload")
        try:
            images.append((load_image(Image.open(f)), name))
        except (ImageLoadError, OSError, ValueError) as exc:
            st.error(f"Skipping **{name}**: not a readable image ({exc}).")
    return images


def heat_rgb(heatmap: np.ndarray, vmax: float) -> np.ndarray:
    colored = cv2.applyColorMap((normalize_map(heatmap, 0.0, vmax) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def thumb(img: np.ndarray, width: int = 360) -> np.ndarray:
    h, w = img.shape[:2]
    return cv2.resize(img, (width, max(1, round(h * width / w))), interpolation=cv2.INTER_AREA)


# ----------------------------------------------------------------------------- sidebar
# FEWSHOT_DEVICE=cpu forces CPU mode (e.g. to preview a CPU-only deployment on a GPU machine).
device = select_device(os.environ.get("FEWSHOT_DEVICE", "auto"))
on_cpu = device.type == "cpu"
with st.sidebar:
    st.header("Settings")
    resolution = st.select_slider("Resolution (smaller edge, px)", options=[224, 336, 448, 672],
                                  value=CPU_DEFAULT_RESOLUTION if on_cpu else GPU_DEFAULT_RESOLUTION)
    masking = st.radio("Foreground masking", ["auto", "on", "off"], horizontal=True,
                       help="auto: the paper's choice for MVTec categories, otherwise an automatic test on the "
                            "first reference")
    rotate = st.toggle("Rotation augmentation of references", value=True,
                       help="AnomalyDINO 'agnostic' default: 8 rotations per reference")
    policy = st.selectbox("Cutoff policy", list(POLICIES), format_func=lambda p: POLICIES[p])
    n_cal = st.select_slider("Calibration images per class", options=[8, 16, 32, 64],
                             value=CPU_DEFAULT_CAL if on_cpu else GPU_DEFAULT_CAL)
    st.caption(f"Device: **{device.type.upper()}**" + (
        f" — CPU mode defaults to {CPU_DEFAULT_RESOLUTION} px and {CPU_DEFAULT_CAL} calibration images per class "
        "so setup stays well under a minute; lower the resolution on a very slow machine. The benchmark uses "
        "672 px." if on_cpu else ""))

st.title("🔍 fewshot-inspector")
st.markdown("Training-free visual inspection: learn *normal* from **1–16 photos of good parts**, choose a pass/fail "
            "cutoff from **synthetic defects only**, then score new images with a defect heatmap. "
            "Frozen DINOv2 ViT-S/14 + nearest-neighbour patch matching (AnomalyDINO, WACV 2025).")

# ----------------------------------------------------------------------------- MVTec example loader
if DATA_ROOT.exists():
    with st.expander("Load an MVTec AD example (local dataset found)"):
        c1, c2 = st.columns([2, 1])
        cat = c1.selectbox("Category", MVTEC_CATEGORIES, index=0)
        if c2.button("Load 4 refs + random tests", use_container_width=True):
            data = load_category(DATA_ROOT, cat)
            rng = np.random.default_rng(int(time.time()))
            goods = [s for s in data.test if s.label == 0]
            bads = [s for s in data.test if s.label == 1]
            picks = [goods[i] for i in rng.choice(len(goods), 2, replace=False)] + \
                    [bads[i] for i in rng.choice(len(bads), 3, replace=False)]
            st.session_state["example"] = {"category": cat, "refs": data.train_paths[:4],
                                           "tests": [(s.path, s.defect_type) for s in picks]}
        st.caption("MVTec AD © MVTec Software GmbH, CC BY-NC-SA 4.0 — images are read from your local copy, "
                   "never bundled.")
example = st.session_state.get("example")

# ----------------------------------------------------------------------------- step 1
st.subheader("1 · Good reference images")
ref_files = st.file_uploader("Upload 1–16 photos of good parts (4 is a good start)", type=["png", "jpg", "jpeg", "bmp"],
                             accept_multiple_files=True, key="refs")
refs: list[np.ndarray] = [img for img, _ in read_uploads(ref_files)] if ref_files else []
category = None
if not refs and example:
    refs = [load_image(p) for p in example["refs"]]
    category = example["category"]
    st.info(f"Using 4 MVTec `{category}` training images as references.")
if refs:
    st.image([thumb(r, 160) for r in refs[:16]], width=120)
if len(refs) > 16:
    st.warning("Using the first 16 references.")
    refs = refs[:16]

signature = (resolution, masking, rotate, n_cal, category, len(refs), tuple(r.shape for r in refs),
             tuple(int(r.sum()) for r in refs))
if st.button("Build inspector", type="primary", disabled=not refs):
    with st.status("Building inspector…", expanded=True) as status:
        backbone = get_backbone(str(device))
        cfg = InspectorConfig(resolution=resolution, masking=masking, category=category, rotate_references=rotate,
                              device=str(device))
        st.write(f"Extracting DINOv2 patch features of {len(refs)} reference(s) "
                 f"({'8 rotations each' if rotate else 'no rotations'}), then scoring {n_cal} synthetic goods + "
                 f"{n_cal} synthetic defects{' (leave-one-out)' if len(refs) >= 2 else ''}…")
        insp, cal, cutoffs, timing = build_calibrated_inspector(cfg, backbone, refs, n_total=n_cal, keep_examples=3)
        status.update(label=f"Inspector ready in {timing['total_s']:.1f} s — memory bank {len(insp.bank or []):,} "
                            f"patches, masking {'on' if insp.use_masking else 'off'}", state="complete")
    st.session_state["session"] = Session(insp, cal, cutoffs, timing["total_s"], timing["fit_s"], len(refs), signature)

sess: Session | None = st.session_state.get("session")
if sess is not None:
    if sess.signature != signature:
        st.warning("Settings or references changed since the inspector was built — click **Build inspector** again.")
    auto_cutoff = sess.cutoffs[policy]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Setup time", f"{sess.setup_s:.1f} s", help=f"feature bank {sess.fit_s:.1f} s + calibration")
    m2.metric(f"Cutoff ({policy})", f"{auto_cutoff:.4f}")
    m3.metric("Synthetic AUROC", f"{sess.calibration.auroc * 100:.1f} %")
    m4.metric("References", f"{sess.n_refs}")
    if sess.n_refs == 1:
        st.caption("⚠️ With one reference there is no leave-one-out: the cutoff is calibrated on augmentations of "
                   "the bank image itself and tends to be too low (more false rejects). Use ≥ 2–4 references.")
    hi = float(max(sess.calibration.defect_scores.max(), auto_cutoff) * 1.5)
    if st.sidebar.checkbox("Manual cutoff override"):
        cutoff = st.sidebar.slider("Cutoff", 0.0, hi, float(auto_cutoff), step=hi / 500, format="%.4f")
    else:
        cutoff = auto_cutoff
    if sess.calibration.examples:
        st.markdown("**Synthetic defects used for calibration** (scored by the fitted inspector):")
        cols = st.columns(len(sess.calibration.examples))
        for col, ex in zip(cols, sess.calibration.examples, strict=False):
            r = sess.inspector.score(ex["defect"])
            vmax = max(1.5 * cutoff, float(r.heatmap.max()))
            col.image(thumb(overlay(ex["defect"], r.heatmap, 0.45, 0.0, vmax)),
                      caption=f"{ex['kind']}: score {r.score:.4f} → {'FAIL' if r.score > cutoff else 'PASS'}")

    # ------------------------------------------------------------------------- step 2
    st.subheader("2 · Inspect")
    test_files = st.file_uploader("Upload test image(s)", type=["png", "jpg", "jpeg", "bmp"],
                                  accept_multiple_files=True, key="tests")
    tests: list[tuple[np.ndarray, str]] = read_uploads(test_files) if test_files else []
    if not tests and example:
        tests = [(load_image(p), f"{example['category']}/{d}/{p.name}") for p, d in example["tests"]]
    for img, name in tests:
        r = sess.inspector.score(img)
        margin = r.score - cutoff
        st.markdown(f"**{name}**")
        c0, c1, c2, c3 = st.columns([1.1, 1, 1, 1])
        if margin > 0:
            c0.error(f"**FAIL** — score {r.score:.4f} is {margin:+.4f} above the cutoff")
        else:
            c0.success(f"**PASS** — score {r.score:.4f} is {-margin:.4f} below the cutoff")
        c0.caption(f"Latency {r.latency_ms:.0f} ms ({device.type}, {resolution} px)")
        vmax = max(1.5 * cutoff, float(r.heatmap.max()))
        c1.image(thumb(img), caption="input")
        c2.image(thumb(overlay(img, r.heatmap, 0.45, 0.0, vmax)), caption="heatmap overlay")
        c3.image(thumb(heat_rgb(r.heatmap, vmax)), caption=f"raw heatmap (0 – {vmax:.3f})")
else:
    st.info("Upload good reference images (or load an MVTec example) and click **Build inspector**.")
