"""MVTec AD benchmark: 15 categories × k-shots × seeds, image + pixel AUROC, plus a latency pass.

    python scripts/benchmark.py --categories all --shots 1 2 4 8 16 --seeds 0 1 2

Efficiency: per category, test features and test foreground masks are extracted ONCE (and cached
in ``--cache-dir``); reference features for every view any (k, seed) needs are extracted once; each
(k, seed) only rebuilds the memory bank and runs the 1-NN search.

Metrics: image AUROC per (category, k, seed); pixel AUROC pooled over all test pixels of a category
at the ORIGINAL image resolution (as the reference: maps upsampled to the image size, σ = 4);
AUPRO (FPR ≤ 0.3) behind ``--aupro``.

Outputs: ``results/benchmark.csv`` (long format: category,k,seed,resolution,metric,value),
``results/latency.csv``, ``results/run_meta.json``, ``results/raw/scores_<res>.csv`` (per image).
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
import torch

from inspector import Backbone, Inspector, InspectorConfig, load_image, select_device
from inspector.backbone import synchronize
from inspector.data import MVTEC_CATEGORIES, MVTecCategory, load_category, load_mask, sample_references
from inspector.inspector import patch_to_heatmap, top_pct_mean
from inspector.masking import resolve_masking
from inspector.meta import run_metadata
from inspector.metrics import aupro, image_auroc, pixel_auroc

LOG = logging.getLogger("benchmark")


def empty_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def extract_in_chunks(insp: Inspector, paths: list[Path], chunk: int) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    feats, shapes = [], []
    for i in range(0, len(paths), chunk):
        imgs = [load_image(p) for p in paths[i:i + chunk]]
        shapes += [im.shape[:2] for im in imgs]
        feats += insp.extract_features(imgs)
    return torch.stack(feats), shapes


def test_cache(insp: Inspector, data: MVTecCategory, cache_dir: Path | None, chunk: int) -> dict[str, object]:
    """Test features ``[N, h, w, C]`` (device), fg masks ``[N, h, w]``, original shapes."""
    cfg = insp.config
    path = cache_dir / f"test_{cfg.model_name}_{cfg.resolution}_{data.name}.pt" if cache_dir else None
    names = [str(s.path.relative_to(data.root)) for s in data.test]
    if path is not None and path.exists():
        cached = torch.load(path, map_location="cpu", weights_only=False)
        if cached["names"] == names:
            LOG.info("%s: loaded cached test features %s", data.name, path)
            return {**cached, "feats": cached["feats"].to(insp.device)}
    t0 = time.perf_counter()
    feats, shapes = extract_in_chunks(insp, [s.path for s in data.test], chunk)
    use_mask = resolve_masking(cfg.masking, data.name, None)
    fg = (np.stack([insp.foreground_mask(f) for f in feats]) if use_mask
          else np.ones(feats.shape[:3], dtype=bool))
    out = {"names": names, "feats": feats, "fg": fg, "shapes": shapes, "use_mask": use_mask}
    LOG.info("%s: extracted %d test images in %.1fs (masking=%s, fg=%.2f)", data.name, len(names),
             time.perf_counter() - t0, use_mask, float(fg.mean()))
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({**out, "feats": feats.cpu()}, path)
    return out


def reference_views(insp: Inspector, paths: list[Path]) -> dict[Path, list[torch.Tensor]]:
    """Features of every rotation view of every needed reference image."""
    views: dict[Path, list[torch.Tensor]] = {}
    for p in paths:
        img = load_image(p)
        views[p] = insp.extract_features(insp.reference_views(img))
    return views


def pooled_heatmaps(dist: torch.Tensor, shapes: list[tuple[int, int]], sigma: float, chunk: int) -> list[torch.Tensor]:
    """Full-resolution heatmaps (flattened) for each test image, grouped to share upsampling calls."""
    out: list[torch.Tensor] = []
    i = 0
    while i < len(shapes):
        j = i
        while j < len(shapes) and j - i < chunk and shapes[j] == shapes[i]:
            j += 1
        out.append(patch_to_heatmap(dist[i:j], shapes[i], sigma).reshape(-1))
        i = j
    return out


def run_category(cat: str, args: argparse.Namespace, base: InspectorConfig, backbone: Backbone,
                 rows: list[dict[str, object]], score_rows: list[dict[str, object]]) -> None:
    data = load_category(args.data_root, cat)
    cfg = replace(base, category=cat)
    insp = Inspector(cfg, backbone=backbone)
    device = insp.device
    tc = test_cache(insp, data, Path(args.cache_dir) if args.cache_dir else None, args.chunk)
    feats, fg, shapes = tc["feats"], tc["fg"], tc["shapes"]
    labels = data.labels
    gts = None
    if not args.no_pixel:
        gts = torch.cat([torch.from_numpy(load_mask(s.mask_path, shp)).reshape(-1)
                         for s, shp in zip(data.test, shapes, strict=True)]).to(device)

    needed = sorted({p for k in args.shots for s in args.seeds
                     for p in sample_references(data.train_paths, k, s, args.sampling)})
    t0 = time.perf_counter()
    ref_feats = reference_views(insp, needed)
    LOG.info("%s: extracted %d reference images × %d views in %.1fs", cat, len(needed),
             len(next(iter(ref_feats.values()))), time.perf_counter() - t0)

    for k in args.shots:
        for seed in args.seeds:
            t0 = time.perf_counter()
            refs = sample_references(data.train_paths, k, seed, args.sampling)
            run = Inspector(cfg, backbone=backbone)
            run.fit_features([v for p in refs for v in ref_feats[p]], first_reference=ref_feats[refs[0]][0])
            dists = []
            for i in range(0, feats.shape[0], args.chunk):
                d, _ = run.patch_distances(feats[i:i + args.chunk], fg[i:i + args.chunk])
                dists.append(d)
            dist = torch.cat(dists)
            scores = top_pct_mean(dist, cfg.top_pct).cpu().numpy()
            metrics = {"image_auroc": image_auroc(labels, scores)}
            if gts is not None:
                heat = torch.cat(pooled_heatmaps(dist, shapes, cfg.sigma, args.chunk))
                metrics["pixel_auroc"] = pixel_auroc(heat, gts)
                if args.aupro:
                    maps = heat.cpu().numpy().reshape(len(shapes), *shapes[0])
                    metrics["aupro"] = aupro(maps, gts.cpu().numpy().reshape(maps.shape))
                del heat
            for m, v in metrics.items():
                rows.append({"category": cat, "k": k, "seed": seed, "resolution": cfg.resolution,
                             "metric": m, "value": v})
            for s, sc in zip(data.test, scores, strict=True):
                score_rows.append({"category": cat, "k": k, "seed": seed, "image": str(s.path.relative_to(data.root)),
                                   "defect_type": s.defect_type, "label": s.label, "score": float(sc)})
            LOG.info("%s k=%d seed=%d bank=%d masking=%s %s (%.1fs)", cat, k, seed, len(run.bank or []),
                     run.use_masking, " ".join(f"{m}={v * 100:.2f}" for m, v in metrics.items()),
                     time.perf_counter() - t0)
            del dist, dists, run
    del feats, ref_feats, gts, tc
    empty_cache(device)


def run_latency(args: argparse.Namespace, base: InspectorConfig, backbone: Backbone) -> list[dict[str, object]]:
    """Batch-1 latency, k = 1 (seed 0): (a) tensor → score + heatmap, (b) file → score + heatmap."""
    rng = np.random.default_rng(0)
    out: list[dict[str, object]] = []
    for res in args.latency_resolutions:
        times: dict[str, list[float]] = {"tensor_to_score_heatmap": [], "end_to_end": []}
        for cat in args.categories:
            data = load_category(args.data_root, cat)
            insp = Inspector(replace(base, resolution=res, category=cat), backbone=backbone)
            insp.fit(sample_references(data.train_paths, 1, 0))
            idx = rng.choice(len(data.test), size=min(args.latency_images, len(data.test)), replace=False)
            paths = [data.test[i].path for i in sorted(idx)]
            for _ in range(args.latency_warmup):
                insp.score(paths[0])
            for p in paths:
                img = load_image(p)
                tensor = insp.preprocess(img)
                synchronize(insp.device)
                t0 = time.perf_counter()
                feats = insp.backbone(tensor.unsqueeze(0))
                dist, _ = insp.patch_distances(feats)
                score = float(top_pct_mean(dist, insp.config.top_pct)[0])
                heat = patch_to_heatmap(dist, img.shape[:2], insp.config.sigma)[0].cpu().numpy()
                synchronize(insp.device)
                times["tensor_to_score_heatmap"].append((time.perf_counter() - t0) * 1000)
                t0 = time.perf_counter()
                r = insp.score(p)
                times["end_to_end"].append((time.perf_counter() - t0) * 1000)
                assert heat.shape == r.heatmap.shape and np.isfinite(score)
        for mode, ts in times.items():
            arr = np.array(ts)
            out.append({"resolution": res, "mode": mode, "median_ms": float(np.median(arr)),
                        "p95_ms": float(np.percentile(arr, 95)), "mean_ms": float(arr.mean()), "n": len(arr),
                        "k": 1, "batch_size": 1, "feature_dtype": "float32",
                        "knn_dtype": str(Inspector(base, backbone=backbone).knn_dtype).replace("torch.", "")})
            LOG.info("latency @%d %s: median %.1f ms, p95 %.1f ms (n=%d)", res, mode, out[-1]["median_ms"],
                     out[-1]["p95_ms"], len(arr))
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="MVTec AD benchmark for fewshot-inspector")
    ap.add_argument("--data-root", default="data/mvtec_ad")
    ap.add_argument("--categories", nargs="+", default=["all"])
    ap.add_argument("--shots", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--resolution", type=int, default=672, help="672 = the paper's 96.6%% configuration")
    ap.add_argument("--sampling", default="contiguous", choices=["contiguous", "random"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--chunk", type=int, default=16, help="test images per kNN / upsampling batch")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--cache-dir", default=".cache", help="'' disables the on-disk test-feature cache")
    ap.add_argument("--no-pixel", action="store_true", help="skip pixel AUROC")
    ap.add_argument("--aupro", action="store_true", help="also compute AUPRO (slow at full resolution)")
    ap.add_argument("--no-latency", action="store_true")
    ap.add_argument("--latency-only", action="store_true")
    ap.add_argument("--latency-images", type=int, default=50)
    ap.add_argument("--latency-warmup", type=int, default=10)
    ap.add_argument("--latency-resolutions", nargs="+", type=int, default=[672, 448])
    args = ap.parse_args()
    if args.categories == ["all"]:
        args.categories = list(MVTEC_CATEGORIES)

    device = select_device(args.device)
    base = InspectorConfig(resolution=args.resolution, device=str(device))
    backbone = Backbone(base.model_name, device)
    out_dir = Path(args.out_dir)
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    timings: dict[str, float] = {}

    if not args.latency_only:
        rows: list[dict[str, object]] = []
        score_rows: list[dict[str, object]] = []
        for cat in args.categories:
            t0 = time.perf_counter()
            run_category(cat, args, base, backbone, rows, score_rows)
            timings[cat] = time.perf_counter() - t0
        pd.DataFrame(rows).to_csv(out_dir / "benchmark.csv", index=False)
        pd.DataFrame(score_rows).to_csv(out_dir / "raw" / f"scores_{args.resolution}.csv", index=False)
        LOG.info("wrote %s (%d rows)", out_dir / "benchmark.csv", len(rows))
    if not args.no_latency:
        t0 = time.perf_counter()
        pd.DataFrame(run_latency(args, base, backbone)).to_csv(out_dir / "latency.csv", index=False)
        timings["latency"] = time.perf_counter() - t0

    meta = run_metadata(device, base.to_dict(), script="benchmark", args=vars(args),
                        pixel_eval_resolution="original image resolution (700-1024 px), all pixels pooled per category",
                        timings_s=timings, total_s=time.perf_counter() - t_start)
    meta_path = out_dir / ("run_meta_latency.json" if args.latency_only else "run_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    LOG.info("done in %.1f min -> %s", (time.perf_counter() - t_start) / 60, meta_path)


if __name__ == "__main__":
    main()
