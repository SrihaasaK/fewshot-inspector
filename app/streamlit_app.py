"""fewshot-inspector demo: learn "normal" from 1–16 good photos, calibrate a cutoff with synthetic defects,
then inspect new images (PASS/FAIL + heatmap).

    streamlit run app/streamlit_app.py

Runs CPU-only on Streamlit Community Cloud (dependencies: app/requirements.txt). "Try an example" uses the
small MVTec AD subset bundled in app/samples/ (CC BY-NC-SA 4.0, see app/samples/LICENSE.md).
"""

from __future__ import annotations

import csv
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
import torch  # noqa: E402

from inspector import Backbone, ImageLoadError, Inspector, InspectorConfig, load_image, select_device  # noqa: E402
from inspector.backbone import available_cpus  # noqa: E402
from inspector.calibrate import CalibrationSet, build_calibrated_inspector  # noqa: E402
from inspector.data import MVTEC_CATEGORIES, load_category  # noqa: E402
from inspector.viz import normalize_map, overlay  # noqa: E402

DATA_ROOT = ROOT / "data" / "mvtec_ad"
SAMPLES = Path(__file__).resolve().parent / "samples"
#: First entry = default. `synth_escape5` was selected on the dev categories (docs/calibration_protocol.md).
POLICIES = {
    "synth_escape5": "Synthetic escape ≤ 5 % (catch ≥ 95 % of synthetic defects) — recommended",
    "synth_youden": "Synthetic Youden (balance FRR vs escapes on synthetic defects)",
    "good_only": "Good-only p99 × 1.10 (needs no defects at all)",
}
#: CPU defaults chosen by scripts/app_cloud_check.py (largest resolution whose 4-reference setup stays
#: ≤ 10 s on 2 CPU threads, i.e. ≥ 3× headroom under the 30 s Community Cloud target).
CPU_DEFAULT_RESOLUTION = 448
GPU_DEFAULT_RESOLUTION = 448
CPU_DEFAULT_CAL = 32
GPU_DEFAULT_CAL = 64
MAX_UPLOAD_SIDE = 1024  # larger uploads are downscaled (2.7 GB RAM on Community Cloud)

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


@st.cache_resource(show_spinner=False)
def configure_threads() -> int:
    """Size torch's CPU thread pool to the cores this container may use (env FEWSHOT_CPU_THREADS overrides)."""
    n = int(os.environ.get("FEWSHOT_CPU_THREADS", available_cpus()))
    torch.set_num_threads(max(1, n))
    return torch.get_num_threads()


@st.cache_resource(show_spinner="Loading DINOv2 ViT-S/14 — the first visit downloads ~90 MB of weights…")
def get_backbone(device: str) -> Backbone:
    return Backbone("dinov2_vits14", device)


@st.cache_data(show_spinner=False)
def load_examples() -> dict[str, dict[str, list]]:
    """Bundled example sets from app/samples/MANIFEST.csv: {category: {"refs": [...], "tests": [(path, truth)]}}."""
    out: dict[str, dict[str, list]] = {}
    manifest = SAMPLES / "MANIFEST.csv"
    if not manifest.is_file():
        return out
    with manifest.open() as f:
        for row in csv.DictReader(f):
            ex = out.setdefault(row["category"], {"refs": [], "tests": []})
            path = str(SAMPLES / row["sample_path"])
            if row["split"] == "refs":
                ex["refs"].append(path)
            else:
                ex["tests"].append((path, row["defect_type"]))
    return out


def downscale(img: np.ndarray, max_side: int = MAX_UPLOAD_SIDE) -> np.ndarray:
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    return img if s >= 1 else cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


def read_uploads(files: list) -> list[tuple[np.ndarray, str]]:
    """Decode uploads (downscaled to ≤ MAX_UPLOAD_SIDE); unreadable files are reported and skipped."""
    from PIL import Image

    images = []
    for f in files:
        name = getattr(f, "name", "upload")
        try:
            images.append((downscale(load_image(Image.open(f))), name))
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
threads = configure_threads() if on_cpu else None
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
        f" ({threads} thread{'s' if threads != 1 else ''}) — CPU mode defaults to {CPU_DEFAULT_RESOLUTION} px and "
        f"{CPU_DEFAULT_CAL} calibration images per class to keep setup short; the benchmark uses 672 px."
        if on_cpu else ""))

st.title("🔍 fewshot-inspector")
st.markdown("Training-free visual inspection: learn *normal* from **1–16 photos of good parts**, choose a pass/fail "
            "cutoff from **synthetic defects only**, then score new images with a defect heatmap. "
            "Frozen DINOv2 ViT-S/14 + nearest-neighbour patch matching (AnomalyDINO, WACV 2025).")

# ----------------------------------------------------------------------------- examples
examples = load_examples()
if examples:
    c1, c2, c3 = st.columns([1.2, 1, 2.2])
    ex_cat = c1.selectbox("Example part", sorted(examples), label_visibility="collapsed")
    if c2.button("▶ Try an example", type="primary", use_container_width=True):
        ex = examples[ex_cat]
        st.session_state["example"] = {"category": ex_cat, "refs": ex["refs"], "tests": ex["tests"],
                                       "source": "bundled"}
        st.session_state["autobuild"] = True
    c3.caption(f"No upload needed: 4 good {ex_cat}s as references, then 2 good + 4 defective test images. "
               "Images: MVTec AD subset, © MVTec Software GmbH, CC BY-NC-SA 4.0 (`app/samples/LICENSE.md`).")

if DATA_ROOT.exists():
    with st.expander("Load an MVTec AD example from the local dataset"):
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
                                           "tests": [(s.path, s.defect_type) for s in picks], "source": "local"}
        st.caption("MVTec AD © MVTec Software GmbH, CC BY-NC-SA 4.0 — read from your local copy.")
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
    st.info(f"Using {len(refs)} good MVTec `{category}` images as references.")
if refs:
    st.image([thumb(r, 160) for r in refs[:16]], width=120)
if len(refs) > 16:
    st.warning("Using the first 16 references.")
    refs = refs[:16]

signature = (resolution, masking, rotate, n_cal, category, len(refs), tuple(r.shape for r in refs),
             tuple(int(r.sum()) for r in refs))
build_clicked = st.button("Build inspector", type="primary", disabled=not refs)
if refs and (build_clicked or st.session_state.pop("autobuild", False)):
    t0 = time.perf_counter()
    with st.status("Building inspector…", expanded=True) as status:
        backbone = get_backbone(str(device))
        cfg = InspectorConfig(resolution=resolution, masking=masking, category=category, rotate_references=rotate,
                              device=str(device))
        st.write(f"Extracting DINOv2 patch features of {len(refs)} reference(s) "
                 f"({'8 rotations each' if rotate else 'no rotations'}), then scoring {n_cal} synthetic goods + "
                 f"{n_cal} synthetic defects{' (leave-one-out)' if len(refs) >= 2 else ''}…")
        insp, cal, cutoffs, timing = build_calibrated_inspector(cfg, backbone, refs, n_total=n_cal, keep_examples=3)
        status.update(label=f"Inspector ready in {timing['total_s']:.1f} s (incl. model load: "
                            f"{time.perf_counter() - t0:.1f} s) — memory bank {len(insp.bank or []):,} patches, "
                            f"masking {'on' if insp.use_masking else 'off'}", state="complete")
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

    # ------------------------------------------------------------------------- step 2
    st.subheader("2 · Inspect")
    test_files = st.file_uploader("Upload test image(s)", type=["png", "jpg", "jpeg", "bmp"],
                                  accept_multiple_files=True, key="tests")
    tests: list[tuple[np.ndarray, str, str | None]] = [(img, name, None) for img, name in read_uploads(test_files)] \
        if test_files else []
    if not tests and example:
        tests = [(load_image(p), Path(p).name, truth) for p, truth in example["tests"]]
    correct = 0
    for img, name, truth in tests:
        r = sess.inspector.score(img)
        margin = r.score - cutoff
        fail = margin > 0
        st.markdown(f"**{name}**")
        c0, c1, c2, c3 = st.columns([1.1, 1, 1, 1])
        if fail:
            c0.error(f"**FAIL** — score {r.score:.4f} is {margin:+.4f} above the cutoff")
        else:
            c0.success(f"**PASS** — score {r.score:.4f} is {-margin:.4f} below the cutoff")
        if truth is not None:
            is_defect = truth != "good"
            ok = fail == is_defect
            correct += ok
            what = f"defective ({truth})" if is_defect else "good"
            c0.caption(f"Ground truth: **{what}** — {'✓ correct' if ok else ('✗ missed defect' if is_defect else '✗ false alarm')}")
        c0.caption(f"Latency {r.latency_ms:.0f} ms ({device.type}, {resolution} px)")
        vmax = max(1.5 * cutoff, float(r.heatmap.max()))
        c1.image(thumb(img), caption="input")
        c2.image(thumb(overlay(img, r.heatmap, 0.45, 0.0, vmax)), caption="heatmap overlay")
        c3.image(thumb(heat_rgb(r.heatmap, vmax)), caption=f"raw heatmap (0 – {vmax:.3f})")
    if tests and all(t[2] is not None for t in tests):
        st.caption(f"Verdicts matching ground truth: {correct} / {len(tests)}.")

    if sess.calibration.examples:
        with st.expander("Synthetic defects used for calibration (scored by the fitted inspector)"):
            cols = st.columns(len(sess.calibration.examples))
            for col, ex in zip(cols, sess.calibration.examples, strict=False):
                r = sess.inspector.score(ex["defect"])
                vmax = max(1.5 * cutoff, float(r.heatmap.max()))
                col.image(thumb(overlay(ex["defect"], r.heatmap, 0.45, 0.0, vmax)),
                          caption=f"{ex['kind']}: score {r.score:.4f} → {'FAIL' if r.score > cutoff else 'PASS'}")
else:
    st.info("Click **▶ Try an example**, or upload good reference images and click **Build inspector**.")
