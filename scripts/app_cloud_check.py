"""Verify the demo app under Streamlit-Community-Cloud-like conditions (CPU only, 2 threads).

Run it from a venv built ONLY from app/requirements.txt (no pandas or other repo extras needed):

    python scripts/app_cloud_check.py sweep   # setup time per CPU resolution -> pick the app default
    python scripts/app_cloud_check.py e2e     # click "▶ Try an example" in the real app (AppTest), time it

`sweep` rule (fixed before measuring): among the candidate resolutions, choose the LARGEST whose worst-case
4-reference setup (fit + 32+32 synthetic calibration images) is <= --max-setup seconds (default 10 s on 2
threads, i.e. >= 3x headroom under the 30 s Community Cloud target for slower cloud cores).
`e2e` should be run with a cold hub cache (e.g. TORCH_HOME=$(mktemp -d)) so the first click includes the
~90 MB DINOv2 download a first visitor would wait for.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from inspector import Backbone, InspectorConfig  # noqa: E402
from inspector.calibrate import build_calibrated_inspector, cutoff_for  # noqa: E402
from inspector.meta import run_metadata  # noqa: E402

LOG = logging.getLogger("app_cloud_check")
SAMPLES = ROOT / "app" / "samples"
APP = ROOT / "app" / "streamlit_app.py"


def examples() -> dict[str, dict[str, list]]:
    out: dict[str, dict[str, list]] = {}
    with (SAMPLES / "MANIFEST.csv").open() as f:
        for row in csv.DictReader(f):
            ex = out.setdefault(row["category"], {"refs": [], "tests": []})
            p = SAMPLES / row["sample_path"]
            if row["split"] == "refs":
                ex["refs"].append(p)
            else:
                ex["tests"].append((p, row["defect_type"]))
    return out


def versions() -> dict[str, str]:
    import importlib.metadata as md

    return {p: md.version(p) for p in ("torch", "torchvision", "streamlit", "numpy", "opencv-python-headless")}


def sweep(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.threads)
    device = torch.device("cpu")
    base = InspectorConfig(device="cpu")
    meta = run_metadata(device, base.to_dict(), script="app_cloud_check sweep", args=vars(args),
                        python=sys.executable, packages=versions(), threads=torch.get_num_threads())
    t0 = time.perf_counter()
    backbone = Backbone("dinov2_vits14", device)
    meta["model_load_s"] = time.perf_counter() - t0
    ex = examples()
    first = sorted(ex)[0]
    build_calibrated_inspector(replace(base, resolution=min(args.resolutions), category=first), backbone,
                               ex[first]["refs"], n_total=8)  # untimed warm-up
    rows = []
    for res in args.resolutions:
        for cat in sorted(ex):
            insp, cal, _, timing = build_calibrated_inspector(replace(base, resolution=res, category=cat), backbone,
                                                                    ex[cat]["refs"], n_total=args.n_cal)
            cutoff = cutoff_for("synth_escape5", cal)
            t1 = time.perf_counter()
            verdicts = [(insp.score(p).score > cutoff) == (truth != "good") for p, truth in ex[cat]["tests"]]
            score_s = (time.perf_counter() - t1) / len(verdicts)
            rows.append({"category": cat, "resolution": res, "n_cal": args.n_cal, **timing,
                         "score_s_per_image": score_s, "correct": int(sum(verdicts)), "n_tests": len(verdicts)})
            LOG.info("res=%d %s: setup %.1fs (fit %.1fs + cal %.1fs), %.2fs/test image, %d/%d verdicts correct",
                     res, cat, timing["total_s"], timing["fit_s"], timing["calibration_s"], score_s,
                     sum(verdicts), len(verdicts))
    ok = [r for r in args.resolutions if max(x["total_s"] for x in rows if x["resolution"] == r) <= args.max_setup]
    chosen = max(ok) if ok else min(args.resolutions)
    out = {"meta": meta, "rows": rows, "rule": f"largest resolution with max setup <= {args.max_setup} s "
                                                f"on {args.threads} threads", "chosen_resolution": chosen}
    Path(args.out).write_text(json.dumps(out, indent=2, default=str))
    LOG.info("chosen CPU default resolution: %d -> %s", chosen, args.out)


def e2e(args: argparse.Namespace) -> None:
    from streamlit.testing.v1 import AppTest

    os.environ["FEWSHOT_DEVICE"] = "cpu"
    os.environ["FEWSHOT_CPU_THREADS"] = str(args.threads)
    hub = Path(torch.hub.get_dir())
    cold = not any((hub / "checkpoints").glob("dinov2_vits14*")) if (hub / "checkpoints").exists() else True
    meta = run_metadata(torch.device("cpu"), None, script="app_cloud_check e2e", args=vars(args),
                        python=sys.executable, packages=versions(), torch_hub_dir=str(hub), cold_cache=cold)
    at = AppTest.from_file(str(APP), default_timeout=args.timeout)
    t0 = time.perf_counter()
    at.run()
    meta["initial_render_s"] = time.perf_counter() - t0
    assert not at.exception, at.exception
    clicks = []
    for cat in args.categories:
        sel = next(s for s in at.selectbox if s.label == "Example part")
        sel.set_value(cat)
        btn = next(b for b in at.button if b.label == "▶ Try an example")
        t0 = time.perf_counter()
        btn.click().run()
        seconds = time.perf_counter() - t0
        assert not at.exception, at.exception
        status = next((e.label for e in at.status), "")  # "Inspector ready in ..."
        captions = [c.value for c in at.caption]
        clicks.append({"category": cat, "seconds_click_to_verdicts": seconds, "status": status,
                       "n_pass": len(at.success), "n_fail": len(at.error),
                       "n_images": len(at.get("image")),
                       "n_heatmap_images": sum("heatmap" in str(getattr(i, "caption", "")) for i in at.get("image")),
                       "correct_caption": next((c for c in captions if c.startswith("Verdicts matching")), ""),
                       "ground_truth_captions": [c for c in captions if c.startswith("Ground truth")]})
        LOG.info("%s: click -> %d verdicts in %.1fs | %s | %s", cat, len(at.success) + len(at.error), seconds,
                 status, clicks[-1]["correct_caption"])
    out = {"meta": meta, "clicks": clicks}
    Path(args.out).write_text(json.dumps(out, indent=2, default=str))
    LOG.info("wrote %s", args.out)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Community-Cloud-like checks of the demo app")
    sub = ap.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("sweep")
    s.add_argument("--threads", type=int, default=2)
    s.add_argument("--resolutions", nargs="+", type=int, default=[224, 336, 448])
    s.add_argument("--n-cal", type=int, default=32)
    s.add_argument("--max-setup", type=float, default=10.0)
    s.add_argument("--out", default=str(ROOT / "results" / "app_cloud_sweep.json"))
    e = sub.add_parser("e2e")
    e.add_argument("--threads", type=int, default=2)
    e.add_argument("--categories", nargs="+", default=["bottle", "screw"])
    e.add_argument("--timeout", type=float, default=600)
    e.add_argument("--out", default=str(ROOT / "results" / "app_cloud_e2e.json"))
    args = ap.parse_args()
    sweep(args) if args.mode == "sweep" else e2e(args)


if __name__ == "__main__":
    main()
