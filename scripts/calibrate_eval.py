"""Evaluate reference-only cutoff policies on the real MVTec test set (docs/calibration_protocol.md).

    python scripts/calibrate_eval.py --categories all --shots 1 2 4 8 16 --seeds 0 1 2

Needs the per-image test scores written by ``scripts/benchmark.py`` (``results/raw/scores_<res>.csv``),
i.e. the deployed inspector with all k references in its bank.

Outputs: ``results/calibration.csv`` (category,k,seed,policy,cutoff,frr,escape,balanced_accuracy,…),
``results/calibration_synth.csv`` (synthetic vs real AUROC per run), ``results/synth_examples.png``,
``results/run_meta_calibration.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from inspector import Backbone, Inspector, InspectorConfig, load_image, select_device
from inspector.calibrate import DEPLOYABLE_POLICIES, build_calibration_set, cutoff_for, youden_cutoff
from inspector.data import MVTEC_CATEGORIES, load_category, sample_references
from inspector.meta import run_metadata
from inspector.metrics import cutoff_metrics, image_auroc
from inspector.synth import GENERATOR_ORDER
from inspector.viz import contour, save_grid

LOG = logging.getLogger("calibrate_eval")
EXAMPLE_CATEGORIES = ("bottle", "screw", "carpet", "transistor")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Evaluate synthetic-defect cutoff calibration on MVTec AD")
    ap.add_argument("--data-root", default="data/mvtec_ad")
    ap.add_argument("--categories", nargs="+", default=["all"])
    ap.add_argument("--shots", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--resolution", type=int, default=672)
    ap.add_argument("--sampling", default="contiguous", choices=["contiguous", "random"])
    ap.add_argument("--n-total", type=int, default=64, help="calibration images per class")
    ap.add_argument("--scores", default=None, help="default: results/raw/scores_<resolution>.csv")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()
    if args.categories == ["all"]:
        args.categories = list(MVTEC_CATEGORIES)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_path = Path(args.scores or out_dir / "raw" / f"scores_{args.resolution}.csv")
    if not scores_path.exists():
        raise FileNotFoundError(f"{scores_path} missing: run scripts/benchmark.py first")
    real_all = pd.read_csv(scores_path)

    device = select_device(args.device)
    base = InspectorConfig(resolution=args.resolution, device=str(device))
    backbone = Backbone(base.model_name, device)
    meta = run_metadata(device, base.to_dict(), script="calibrate_eval", args=vars(args),
                        protocol="docs/calibration_protocol.md")  # provenance captured at launch
    rows: list[dict[str, object]] = []
    synth_rows: list[dict[str, object]] = []
    example_rows: list[list[np.ndarray]] = []
    example_titles: list[list[str]] = []
    t_start = time.perf_counter()

    for cat in args.categories:
        data = load_category(args.data_root, cat)
        cfg = replace(base, category=cat)
        probe = Inspector(cfg, backbone=backbone)
        needed = sorted({p for k in args.shots for s in args.seeds
                         for p in sample_references(data.train_paths, k, s, args.sampling)})
        images = {p: load_image(p) for p in needed}
        views = {p: probe.extract_features(probe.reference_views(images[p])) for p in needed}
        for k in args.shots:
            for seed in args.seeds:
                t0 = time.perf_counter()
                refs = sample_references(data.train_paths, k, seed, args.sampling)
                keep = 4 if (cat in EXAMPLE_CATEGORIES and k == 4 and seed == 0) else 0
                cal = build_calibration_set(cfg, backbone, [images[p] for p in refs], seed=seed,
                                            n_total=args.n_total, reference_views=[views[p] for p in refs],
                                            keep_examples=keep)
                real = real_all[(real_all.category == cat) & (real_all.k == k) & (real_all.seed == seed)]
                if len(real) != len(data.test):
                    raise ValueError(f"{cat} k={k} seed={seed}: {len(real)} real scores, expected {len(data.test)}")
                scores, labels = real.score.to_numpy(), real.label.to_numpy().astype(bool)
                cutoffs = {p: cutoff_for(p, cal) for p in DEPLOYABLE_POLICIES}
                cutoffs["oracle"] = youden_cutoff(scores[~labels], scores[labels])
                for policy, cutoff in cutoffs.items():
                    rows.append({"category": cat, "k": k, "seed": seed, "policy": policy, "cutoff": cutoff,
                                 **cutoff_metrics(scores, labels, cutoff)})
                kinds = np.array(cal.defect_kinds)
                per_kind = {f"synth_auroc_{g}": image_auroc(
                    np.r_[np.zeros(len(cal.good_scores)), np.ones(int((kinds == g).sum()))],
                    np.r_[cal.good_scores, cal.defect_scores[kinds == g]]) for g in GENERATOR_ORDER if (kinds == g).any()}
                synth_rows.append({"category": cat, "k": k, "seed": seed, "synth_auroc": cal.auroc,
                                   "real_auroc": image_auroc(labels, scores), "n_synth_good": len(cal.good_scores),
                                   "n_synth_defect": len(cal.defect_scores), "leave_one_out": cal.leave_one_out,
                                   "synth_good_median": float(np.median(cal.good_scores)),
                                   "real_good_median": float(np.median(scores[~labels])), **per_kind})
                LOG.info("%s k=%d seed=%d synthAUROC=%.3f realAUROC=%.3f | %s (%.1fs)", cat, k, seed, cal.auroc,
                         synth_rows[-1]["real_auroc"],
                         " ".join(f"{r['policy']}:BA={r['balanced_accuracy']:.3f}" for r in rows[-len(cutoffs):]),
                         time.perf_counter() - t0)
                for ex in cal.examples:
                    example_rows.append([ex["good"], contour(ex["defect"], ex["mask"])])
                    example_titles.append([f"{cat}: augmented good", f"synthetic {ex['kind']} (GT outline)"])
        del views, images

    pd.DataFrame(rows).to_csv(out_dir / "calibration.csv", index=False)
    pd.DataFrame(synth_rows).to_csv(out_dir / "calibration_synth.csv", index=False)
    if example_rows:
        # lay out 4 examples per row: (good, defect) pairs
        flat = [(img, title) for row, trow in zip(example_rows, example_titles, strict=True)
                for img, title in zip(row, trow, strict=True)]
        grid = [[f[0] for f in flat[i:i + 8]] for i in range(0, len(flat), 8)]
        titles = [[f[1] for f in flat[i:i + 8]] for i in range(0, len(flat), 8)]
        save_grid(grid, titles, out_dir / "synth_examples.png",
                  suptitle="Synthetic defects used for calibration (MVTec AD images, CC BY-NC-SA 4.0)", cell=2.2)
    meta["total_s"] = time.perf_counter() - t_start
    (out_dir / "run_meta_calibration.json").write_text(json.dumps(meta, indent=2, default=str))
    LOG.info("wrote %s (%d rows) in %.1f min", out_dir / "calibration.csv", len(rows),
             (time.perf_counter() - t_start) / 60)


if __name__ == "__main__":
    main()
