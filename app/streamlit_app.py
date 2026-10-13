"""fewshot-inspector demo: learn "normal" from 1–16 good photos, calibrate a cutoff with synthetic defects,
then inspect new images (PASS/FAIL + heatmap).

    streamlit run app/streamlit_app.py

Runs CPU-only on Streamlit Community Cloud (dependencies: app/requirements.txt). "Try an example" uses the
small MVTec AD subset bundled in app/samples/ (CC BY-NC-SA 4.0, see app/samples/LICENSE.md).
"""

from __future__ import annotations

import csv
import hashlib
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

try:  # fail with an actionable message instead of Community Cloud's redacted traceback
    import cv2
    import numpy as np
    import torch
    import torchvision  # noqa: F401
except ModuleNotFoundError as exc:  # pragma: no cover - deployment misconfiguration
    import platform

    st.error(
        f"**`{exc.name}` is not installed** (Python {platform.python_version()}, {sys.platform}/"
        f"{platform.machine()}). `app/requirements.txt` pins CPU-only torch/torchvision wheels for every Python "
        "that has them (see its header). On Streamlit Community Cloud open **Manage app → logs** to see why the "
        "install skipped or failed; if your Python version is not covered, delete the app and redeploy it with "
        "**Python 3.12** under *Advanced settings*."
    )
    st.stop()

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
#: Streamlit Community Cloud runs apps from /mount/src/<repo>; its sustained CPU budget is ~1.75 EPYC cores.
ON_COMMUNITY_CLOUD = Path(__file__).resolve().as_posix().startswith("/mount/src/")
#: CPU default resolution: largest whose 4-reference setup stays under 30 s on 2 threads ON THE TARGET HOST.
#: Local CPUs: 448 (results/app_cloud_sweep.json). Community Cloud: 224, measured on the live app
#: (results/app_cloud_live.json; 336 and 448 exceeded 30 s there).
CPU_DEFAULT_RESOLUTION = 224 if ON_COMMUNITY_CLOUD else 448
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
    example_category: str | None  # set only when the references came from an example set (not uploads)


def default_threads() -> int:
    """Default CPU threads: FEWSHOT_CPU_THREADS, else the cgroup CPU quota read NOW, capped at 2 on Community Cloud.

    Not cached on purpose: Community Cloud starts the container without a quota (`cpu.max = max`, 16 visible
    CPUs) and applies its ~1.75-core quota later; a value cached at start-up left torch running 16 threads on
    <2 cores, which made 'Try an example' take minutes. Apps on Community Cloud run from /mount/src/<repo>.
    """
    if os.environ.get("FEWSHOT_CPU_THREADS"):
        return max(1, int(os.environ["FEWSHOT_CPU_THREADS"]))
    n = available_cpus()
    if ON_COMMUNITY_CLOUD:
        n = min(n, 2)
    return max(1, n)


def cpu_diagnostics() -> dict[str, object]:
    """What the host exposes about its CPUs (shown in the sidebar; no secrets, env var NAMES only)."""
    info: dict[str, object] = {
        "python": platform.python_version(), "platform": platform.platform(),
        "os.cpu_count()": os.cpu_count(),
        "sched_getaffinity": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "available_cpus()": available_cpus(), "torch threads": torch.get_num_threads(),
    }
    for f in ("/sys/fs/cgroup/cpu.max", "/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "/sys/fs/cgroup/cpu/cpu.cfs_period_us",
              "/proc/self/cgroup"):
        try:
            info[f] = Path(f).read_text().strip()[:300]
        except OSError as exc:
            info[f] = f"({type(exc).__name__})"
    try:
        info["cpu model"] = next((ln.split(":", 1)[1].strip() for ln in Path("/proc/cpuinfo").read_text().splitlines()
                                  if ln.startswith("model name")), None)
    except OSError:
        info["cpu model"] = None
    info["env (names)"] = sorted(k for k in os.environ if k.startswith(("STREAMLIT", "OMP", "MKL", "HOSTNAME")))
    return info


def image_key(img: np.ndarray) -> tuple:
    return img.shape, hashlib.md5(np.ascontiguousarray(img).tobytes()).hexdigest()


def score_cached(insp: Inspector, img: np.ndarray, max_entries: int = 64):
    """Score once per (inspector, image): changing the policy or cutoff never re-runs the model."""
    cache = insp.__dict__.setdefault("_app_score_cache", {})
    key = image_key(img)
    if key not in cache:
        if len(cache) >= max_entries:
            cache.pop(next(iter(cache)))
        cache[key] = insp.score(img)
    return cache[key]


@st.cache_resource(max_entries=6, show_spinner=False)
def example_inspector(category: str, resolution: int, masking: str, rotate: bool, n_cal: int, device: str,
                      threads: int | None):
    """Bundled examples are identical for every visitor: build once per settings, share across sessions.

    ``threads`` is only part of the cache key, so build times for different thread counts can be compared."""
    cfg = InspectorConfig(resolution=resolution, masking=masking, category=category, rotate_references=rotate,
                          device=device)
    refs = [load_image(p) for p in load_examples()[category]["refs"]]
    return build_calibrated_inspector(cfg, get_backbone(device), refs, n_total=n_cal, keep_examples=3)


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
            st.error(f"Skipping **{name}**: not a readable image file ({type(exc).__name__}).")
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
    threads = None
    if on_cpu:
        with st.expander("Advanced: CPU threads & diagnostics"):
            n_default = default_threads()
            threads = st.select_slider("CPU threads", options=sorted({1, 2, 4, 8, 16, n_default}),
                                       value=n_default,
                                       help="More threads than the host's CPU quota slows torch down (oversubscription)")
            st.json(cpu_diagnostics(), expanded=False)
        torch.set_num_threads(threads)
    st.caption(f"Device: **{device.type.upper()}**" + (
        f" ({threads} thread{'s' if threads != 1 else ''}) — CPU mode defaults to {CPU_DEFAULT_RESOLUTION} px and "
        f"{CPU_DEFAULT_CAL} calibration images per class to keep setup short; the benchmark uses 672 px."
        if on_cpu else ""))

st.title("🔍 fewshot-inspector")
st.markdown("Training-free visual inspection: learn *normal* from **1–16 photos of good parts**, choose a pass/fail "
            "cutoff from **synthetic defects only**, then score new images with a defect heatmap. "
            "Frozen DINOv2 ViT-S/14 + nearest-neighbour patch matching (AnomalyDINO, WACV 2025).")
st.info(f"**Demo vs benchmark.** This demo runs at **{resolution} px**"
        + (" on a shared CPU to stay fast" if on_cpu else "")
        + ". The benchmark numbers in the README use **672 px** (the AnomalyDINO paper's setting; latency is reported "
          "at 448 and 672 px). PASS/FAIL hits on the 6 example images are an illustration, **not benchmark accuracy**.")

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
    c3.caption(f"No upload needed: 4 good {ex_cat}s as references, then 2 good + 4 defective test images "
               f"(demo at {resolution} px — not benchmark accuracy). "
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
        st.write(f"Extracting DINOv2 patch features of {len(refs)} reference(s) "
                 f"({'8 rotations each' if rotate else 'no rotations'}), then scoring {n_cal} synthetic goods + "
                 f"{n_cal} synthetic defects{' (leave-one-out)' if len(refs) >= 2 else ''}…")
        bundled = example is not None and example.get("source") == "bundled" and not ref_files
        if bundled:  # identical for every visitor -> shared cache across sessions
            insp, cal, cutoffs, timing = example_inspector(category, resolution, masking, rotate, n_cal, str(device), threads)
        else:
            backbone = get_backbone(str(device))
            cfg = InspectorConfig(resolution=resolution, masking=masking, category=category, rotate_references=rotate,
                                  device=str(device))
            insp, cal, cutoffs, timing = build_calibrated_inspector(cfg, backbone, refs, n_total=n_cal, keep_examples=3)
        wall = time.perf_counter() - t0
        cached = bundled and wall < 0.5 * timing["total_s"]
        status.update(label=(f"Inspector ready in {wall:.1f} s (shared cache; first build took {timing['total_s']:.1f} s)"
                             if cached else f"Inspector ready in {timing['total_s']:.1f} s (incl. model load: {wall:.1f} s)")
                      + f" — memory bank {len(insp.bank or []):,} patches, masking {'on' if insp.use_masking else 'off'}",
                      state="complete")
    ex_cat = example["category"] if (example and not ref_files) else None
    st.session_state["session"] = Session(insp, cal, cutoffs, timing["total_s"], timing["fit_s"], len(refs), signature,
                                          ex_cat)

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
    # example tests belong to the example's references only — never score them against an uploaded part
    if not tests and example and sess.example_category == example["category"]:
        tests = [(load_image(p), Path(p).name, truth) for p, truth in example["tests"]]
    if not tests:
        st.info("Upload one or more test images of your part to inspect them.")
    correct = 0
    for img, name, truth in tests:
        r = score_cached(sess.inspector, img)
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
        st.caption(f"Verdicts matching ground truth: {correct} / {len(tests)} — a {len(tests)}-image demo at "
                   f"{resolution} px, not benchmark accuracy (benchmark: full MVTec AD test sets at 672 px, README).")

    if sess.calibration.examples:
        with st.expander("Synthetic defects used for calibration (scored by the fitted inspector)"):
            cols = st.columns(len(sess.calibration.examples))
            for col, ex in zip(cols, sess.calibration.examples, strict=False):
                r = score_cached(sess.inspector, ex["defect"])
                vmax = max(1.5 * cutoff, float(r.heatmap.max()))
                col.image(thumb(overlay(ex["defect"], r.heatmap, 0.45, 0.0, vmax)),
                          caption=f"{ex['kind']}: score {r.score:.4f} → {'FAIL' if r.score > cutoff else 'PASS'}")
else:
    st.info("Click **▶ Try an example**, or upload good reference images and click **Build inspector**.")
