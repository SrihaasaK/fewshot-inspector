"""Phase 1 gate: AnomalyDINO-S reproduction on MVTec `bottle` and `screw` via the public Inspector API.

k = 1, seeds {0, 1, 2}, 672 px, agnostic preprocessing. PASS iff for each category the mean image
AUROC over seeds is within max(2.0 pts, paper std) of the paper's per-category number
(``docs/paper_numbers.csv``, Tab. MVTec-detailed-agnostic).

    python scripts/gate_phase1.py --data-root data/mvtec_ad
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from inspector import Backbone, Inspector, InspectorConfig, select_device
from inspector.data import load_category, sample_references
from inspector.meta import run_metadata
from inspector.metrics import image_auroc

LOG = logging.getLogger("gate_phase1")


def paper_number(path: Path, category: str, k: int, resolution: int) -> tuple[float, float]:
    with path.open() as f:
        for row in csv.DictReader(f):
            if (row["source"] == "tab:MVTec-detailed-agnostic" and row["category"] == category
                    and int(row["k"]) == k and int(row["resolution"]) == resolution
                    and row["metric"] == "image_auroc"):
                return float(row["mean"]), float(row["std"])
    raise KeyError(f"no paper number for {category} k={k} @{resolution}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", default="data/mvtec_ad")
    ap.add_argument("--categories", nargs="+", default=["bottle", "screw"])
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--resolution", type=int, default=672)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--chunk", type=int, default=32, help="test images per score_batch call")
    ap.add_argument("--paper", default="docs/paper_numbers.csv")
    ap.add_argument("--out", default="results/phase1_gate.json")
    args = ap.parse_args()

    device = select_device(args.device)
    base = InspectorConfig(resolution=args.resolution, device=str(device))
    backbone = Backbone(base.model_name, device)
    report: dict[str, object] = {"meta": run_metadata(device, base.to_dict(), script="gate_phase1", args=vars(args)),
                                "categories": {}}
    all_pass = True
    for cat in args.categories:
        data = load_category(args.data_root, cat)
        paths, labels = [s.path for s in data.test], data.labels
        aurocs = []
        for seed in args.seeds:
            t0 = time.perf_counter()
            refs = sample_references(data.train_paths, args.k, seed)
            insp = Inspector(replace(base, category=cat), backbone=backbone).fit(refs)
            scores = np.concatenate([
                [r.score for r in insp.score_batch(paths[i:i + args.chunk], heatmaps=False)]
                for i in range(0, len(paths), args.chunk)
            ])
            auc = image_auroc(labels, scores) * 100
            aurocs.append(auc)
            LOG.info("%s seed=%d refs=%s masking=%s AUROC=%.2f (%.1fs)", cat, seed, [p.name for p in refs],
                     insp.use_masking, auc, time.perf_counter() - t0)
        paper_mean, paper_std = paper_number(Path(args.paper), cat, args.k, args.resolution)
        mean, std = float(np.mean(aurocs)), float(np.std(aurocs))
        tol = max(2.0, paper_std)
        ok = abs(mean - paper_mean) <= tol
        all_pass &= ok
        report["categories"][cat] = {  # type: ignore[index]
            "image_auroc_per_seed": aurocs, "mean": mean, "std": std, "paper_mean": paper_mean,
            "paper_std": paper_std, "tolerance": tol, "pass": ok,
        }
        LOG.info("%s: ours %.2f ± %.2f vs paper %.1f ± %.1f -> %s", cat, mean, std, paper_mean, paper_std,
                 "PASS" if ok else "FAIL")
    report["gate_pass"] = all_pass
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    LOG.info("gate %s -> %s", "PASS" if all_pass else "FAIL", out)


if __name__ == "__main__":
    main()
