"""Failure analysis: per-defect-type AUROC with defect sizes, and heatmap grids for the worst categories.

    python scripts/failure_cases.py --k 4 --seed 0

Reads ``results/benchmark.csv`` and ``results/raw/scores_<res>.csv`` (from scripts/benchmark.py).
Writes ``results/defect_types.csv`` and ``docs/failure_<category>.png`` (MVTec images, CC BY-NC-SA 4.0).
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from inspector import Backbone, Inspector, InspectorConfig, load_image, select_device
from inspector.data import load_category, load_mask, sample_references
from inspector.metrics import image_auroc
from inspector.viz import contour, overlay, save_grid

LOG = logging.getLogger("failure_cases")


def defect_type_table(data_root: str, scores: pd.DataFrame, k: int, resolution: int) -> pd.DataFrame:
    """Per (category, defect type): AUROC vs all good test images (mean over seeds) and GT size."""
    rows = []
    grid = resolution // 14
    for cat, sc in scores[scores.k == k].groupby("category"):
        data = load_category(data_root, cat)
        sizes: dict[str, list[tuple[float, float]]] = {}
        for s in data.test:
            if s.label == 0:
                continue
            img_shape = load_image(s.path).shape[:2]
            mask = load_mask(s.mask_path, img_shape)
            area = float(mask.sum())
            patch_px = min(img_shape) / grid
            sizes.setdefault(s.defect_type, []).append(
                (area / mask.size * 100, 2 * np.sqrt(area / np.pi) / patch_px))
        for dtype in sorted(sizes):
            aucs = []
            for _, run in sc.groupby("seed"):
                sub = run[(run.label == 0) | (run.defect_type == dtype)]
                aucs.append(image_auroc(sub.label.to_numpy(), sub.score.to_numpy()))
            arr = np.array(sizes[dtype])
            rows.append({"category": cat, "defect_type": dtype, "k": k, "n": len(arr),
                         "auroc_vs_good": float(np.mean(aucs)), "median_area_pct": float(np.median(arr[:, 0])),
                         "median_diam_patches": float(np.median(arr[:, 1]))})
    return pd.DataFrame(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Failure analysis for fewshot-inspector on MVTec AD")
    ap.add_argument("--data-root", default="data/mvtec_ad")
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--docs", type=Path, default=Path("docs"))
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resolution", type=int, default=672)
    ap.add_argument("--n-worst", type=int, default=3)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--reference-sheet", default=None, metavar="CATEGORY",
                    help="only write docs/reference_sheet_<category>.png (train images 000-005 = seeds 0-5 at k=1)")
    args = ap.parse_args()
    if args.reference_sheet:
        cat = args.reference_sheet
        train = load_category(args.data_root, cat).train_paths[:6]
        imgs = [load_image(p) for p in train]
        path = save_grid([imgs[:3], imgs[3:]], [[f"{p.name} = k=1 seed {i}" for i, p in enumerate(train[:3])],
                                                 [f"{p.name} = k=1 seed {i + 3}" for i, p in enumerate(train[3:])]],
                         args.docs / f"reference_sheet_{cat}.png",
                         suptitle=f"{cat}: the first six train images (MVTec AD, CC BY-NC-SA 4.0)", cell=2.6)
        LOG.info("wrote %s", path)
        return

    bench = pd.read_csv(args.results / "benchmark.csv")
    scores = pd.read_csv(args.results / "raw" / f"scores_{args.resolution}.csv")
    table = defect_type_table(args.data_root, scores, args.k, args.resolution)
    table.to_csv(args.results / "defect_types.csv", index=False)
    LOG.info("wrote %s (%d rows)", args.results / "defect_types.csv", len(table))

    img = bench[(bench.metric == "image_auroc") & (bench.k == args.k)].groupby("category").value.mean()
    worst = img.sort_values().index[: args.n_worst].tolist()
    LOG.info("worst categories at k=%d: %s", args.k, ", ".join(f"{c} {img[c] * 100:.1f}" for c in worst))

    device = select_device(args.device)
    base = InspectorConfig(resolution=args.resolution, device=str(device))
    backbone = Backbone(base.model_name, device)
    for cat in worst:
        data = load_category(args.data_root, cat)
        by_name = {str(s.path.relative_to(data.root)): s for s in data.test}
        run = scores[(scores.category == cat) & (scores.k == args.k) & (scores.seed == args.seed)]
        missed = run[run.label == 1].nsmallest(3, "score")
        alarms = run[run.label == 0].nlargest(2, "score")
        picks = [("lowest-scoring defect", r) for _, r in missed.iterrows()] + \
                [("highest-scoring good", r) for _, r in alarms.iterrows()]
        insp = Inspector(replace(base, category=cat), backbone=backbone)
        insp.fit(sample_references(data.train_paths, args.k, args.seed))
        results = [(what, r, by_name[r.image], insp.score(by_name[r.image].path)) for what, r in picks]
        vmax = max(float(res.heatmap.max()) for *_, res in results)
        top, bottom, titles_top, titles_bottom = [], [], [], []
        for what, r, sample, res in results:
            image = load_image(sample.path)
            mask = load_mask(sample.mask_path, image.shape[:2])
            top.append(contour(image, mask) if mask.any() else image)
            bottom.append(overlay(image, res.heatmap, 0.5, 0.0, vmax))
            titles_top.append(f"{what}\n{r.defect_type} (GT outline)" if mask.any() else f"{what}\n{r.defect_type}")
            titles_bottom.append(f"score {res.score:.3f}")
        path = save_grid([top, bottom], [titles_top, titles_bottom], args.docs / f"failure_{cat}.png",
                         suptitle=f"{cat}: k={args.k}, seed={args.seed}, image AUROC {img[cat] * 100:.1f} % "
                                  f"(mean of seeds). Heatmap scale 0–{vmax:.3f}. MVTec AD, CC BY-NC-SA 4.0",
                         cell=2.6)
        LOG.info("wrote %s", path)


if __name__ == "__main__":
    main()
