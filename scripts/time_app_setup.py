"""Time the demo app's setup path on CPU: 4 good references → fitted + calibrated inspector.

    python scripts/time_app_setup.py

Runs ``inspector.calibrate.build_calibrated_inspector`` (exactly what the Streamlit "Build inspector" button
calls) with the model already loaded (the app caches it), for several resolutions / calibration sizes, using
all CPU threads and 2 threads (≈ a small cloud instance). Writes ``results/app_setup_cpu.csv``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd
import torch

from inspector import Backbone, InspectorConfig
from inspector.calibrate import build_calibrated_inspector
from inspector.data import load_category
from inspector.meta import run_metadata

LOG = logging.getLogger("time_app_setup")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Time the app setup path on CPU")
    ap.add_argument("--data-root", default="data/mvtec_ad")
    ap.add_argument("--categories", nargs="+", default=["bottle", "screw", "carpet"])
    ap.add_argument("--resolutions", nargs="+", type=int, default=[224, 448])
    ap.add_argument("--n-cal", nargs="+", type=int, default=[16, 64])
    ap.add_argument("--threads", nargs="+", type=int, default=[0, 2], help="0 = all available threads")
    ap.add_argument("--out", type=Path, default=Path("results/app_setup_cpu.csv"))
    args = ap.parse_args()

    device = torch.device("cpu")
    all_threads = torch.get_num_threads()
    t0 = time.perf_counter()
    backbone = Backbone("dinov2_vits14", device)
    model_load_s = time.perf_counter() - t0
    base = InspectorConfig(device="cpu")
    meta = run_metadata(device, base.to_dict(), script="time_app_setup", args=vars(args),  # at launch
                        model_load_s=model_load_s, n_references=4, cpu_threads_available=all_threads)
    refs = {c: load_category(args.data_root, c).train_paths[:4] for c in args.categories}
    # untimed warm-up (first-call allocations / kernel selection)
    build_calibrated_inspector(replace(base, resolution=args.resolutions[0], category=args.categories[0]), backbone,
                               refs[args.categories[0]], n_total=8)
    rows = []
    for threads, res, n_cal, cat in itertools.product(args.threads, args.resolutions, args.n_cal, args.categories):
        torch.set_num_threads(threads or all_threads)
        _, cal, _, timing = build_calibrated_inspector(replace(base, resolution=res, category=cat), backbone,
                                                       refs[cat], n_total=n_cal)
        rows.append({"category": cat, "resolution": res, "n_cal": n_cal, "threads": threads or all_threads,
                     **{k: round(v, 3) for k, v in timing.items()}, "synth_auroc": cal.auroc})
        LOG.info("%s res=%d n_cal=%d threads=%d -> %.1fs (fit %.1fs + calibration %.1fs)", cat, res, n_cal,
                 threads or all_threads, timing["total_s"], timing["fit_s"], timing["calibration_s"])
    torch.set_num_threads(all_threads)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str))
    LOG.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
