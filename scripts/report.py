"""Build every reported number/table/chart from result CSVs and inject them into README.md.

    python scripts/report.py

Reads ``results/benchmark.csv``, ``results/latency.csv``, ``results/run_meta*.json``,
``docs/paper_numbers.csv`` (and ``results/calibration.csv`` when present). Writes
``results/auroc_vs_k.png``, ``results/RESULTS.md`` and the block between
``<!-- RESULTS:START -->`` / ``<!-- RESULTS:END -->`` in README.md. Nothing is typed by hand.

Statistics: per category, mean ± std (ddof = 0) over seeds; the macro row is the mean over
categories computed per seed, then mean ± std over seeds (the paper reports mean ± std over
three runs).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

LOG = logging.getLogger("report")

INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
SERIES_1 = "#2a78d6"  # "ours" (reference palette slot 1)
PAPER = "#8a8984"  # neutral reference line
START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
PRETTY = {"bottle": "Bottle", "cable": "Cable", "capsule": "Capsule", "carpet": "Carpet", "grid": "Grid",
          "hazelnut": "Hazelnut", "leather": "Leather", "metal_nut": "Metal nut", "pill": "Pill", "screw": "Screw",
          "tile": "Tile", "toothbrush": "Toothbrush", "transistor": "Transistor", "wood": "Wood", "zipper": "Zipper"}


def short_sha(sha: str) -> str:
    """First 12 hex digits, keeping a ``-dirty`` marker (results produced from uncommitted code)."""
    return sha[:12] + ("-dirty" if sha.endswith("-dirty") else "")


def pm(mean: float, std: float, digits: int = 1) -> str:
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


# ----------------------------------------------------------------------------- AUROC summaries
def per_category(bench: pd.DataFrame, metric: str) -> pd.DataFrame:
    """mean/std over seeds per (category, k), in percent."""
    d = bench[bench.metric == metric]
    g = d.groupby(["category", "k"]).value
    return pd.DataFrame({"mean": g.mean() * 100, "std": g.std(ddof=0) * 100, "n": g.count()}).reset_index()


def macro(bench: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Per seed: mean over categories; then mean/std over seeds, per k (percent)."""
    d = bench[bench.metric == metric]
    per_seed = d.groupby(["k", "seed"]).agg(value=("value", "mean"), n_cat=("category", "nunique")).reset_index()
    g = per_seed.groupby("k")
    return pd.DataFrame({"mean": g.value.mean() * 100, "std": g.value.std(ddof=0) * 100,
                         "n_seeds": g.value.count(), "n_cat": g.n_cat.min()}).reset_index()


def paper_lookup(paper: pd.DataFrame, source: str, resolution: int, metric: str) -> dict[tuple[str, int], tuple[float, float]]:
    d = paper[(paper.source == source) & (paper.resolution == resolution) & (paper.metric == metric)]
    return {(r.category, int(r.k)): (float(r["mean"]), float(r["std"])) for _, r in d.iterrows()}


def image_table(bench: pd.DataFrame, paper: pd.DataFrame, resolution: int) -> str:
    pc = per_category(bench, "image_auroc")
    mc = macro(bench, "image_auroc")
    pp = paper_lookup(paper, "tab:MVTec-detailed-agnostic", resolution, "image_auroc")
    ks = sorted(pc.k.unique())
    head = "| Category | " + " | ".join(f"k={k} ours | k={k} paper" for k in ks) + " |"
    sep = "|---|" + "---:|---:|" * len(ks)
    lines = [head, sep]
    for cat in sorted(pc.category.unique(), key=list(PRETTY).index):
        cells = []
        for k in ks:
            r = pc[(pc.category == cat) & (pc.k == k)].iloc[0]
            p = pp.get((cat, k))
            cells += [pm(r["mean"], r["std"]), pm(*p) if p else "—"]
        lines.append(f"| {PRETTY.get(cat, cat)} | " + " | ".join(cells) + " |")
    cells = []
    for k in ks:
        r = mc[mc.k == k].iloc[0]
        p = pp.get(("MEAN", k))
        cells += [f"**{pm(r['mean'], r['std'])}**", f"**{pm(*p)}**" if p else "—"]
    lines.append("| **Mean** | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def pixel_table(bench: pd.DataFrame, paper: pd.DataFrame, resolution: int) -> str:
    pc = per_category(bench, "pixel_auroc")
    mc = macro(bench, "pixel_auroc")
    pp = paper_lookup(paper, "tab:results_MVTec", resolution, "pixel_auroc")
    ks = sorted(pc.k.unique())
    lines = ["| Category | " + " | ".join(f"k={k}" for k in ks) + " |", "|---|" + "---:|" * len(ks)]
    for cat in sorted(pc.category.unique(), key=list(PRETTY).index):
        cells = [pm(*pc[(pc.category == cat) & (pc.k == k)][["mean", "std"]].iloc[0]) for k in ks]
        lines.append(f"| {PRETTY.get(cat, cat)} | " + " | ".join(cells) + " |")
    lines.append("| **Mean (ours)** | " + " | ".join(
        f"**{pm(*mc[mc.k == k][['mean', 'std']].iloc[0])}**" for k in ks) + " |")
    lines.append("| Mean (paper) | " + " | ".join(pm(*pp[("MEAN", k)]) if ("MEAN", k) in pp else "—" for k in ks) + " |")
    return "\n".join(lines)


def chart(bench: pd.DataFrame, paper: pd.DataFrame, resolution: int, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("image_auroc", "Image AUROC (%)", "tab:MVTec-detailed-agnostic"),
              ("pixel_auroc", "Pixel AUROC (%)", "tab:results_MVTec")]
    panels = [p for p in panels if p[0] in set(bench.metric)]
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 3.6), facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    for ax, (metric, label, source) in zip(axes, panels, strict=True):
        mc = macro(bench, metric)
        pp = paper_lookup(paper, source, resolution, metric)
        ks = mc.k.to_numpy()
        ax.set_facecolor(SURFACE)
        pk = [k for k in ks if ("MEAN", k) in pp]
        if pk:
            pv = np.array([pp[("MEAN", k)][0] for k in pk])
            ax.plot(pk, pv, color=PAPER, lw=2, ls="--", marker="s", ms=6, label=f"Paper (S/{resolution})", zorder=2)
        ax.errorbar(ks, mc["mean"], yerr=mc["std"], color=SERIES_1, lw=2, marker="o", ms=7, capsize=3,
                    label="Ours (mean ± std, 3 seeds)", zorder=3)
        for k, m in zip(ks, mc["mean"], strict=True):
            ax.annotate(f"{m:.1f}", (k, m), textcoords="offset points", xytext=(7, -12), ha="left",
                        fontsize=8, color=INK)
        ax.set_xscale("log", base=2)
        ax.set_xticks(ks, [str(k) for k in ks])
        ax.set_xlabel("k (reference images, log₂ scale)", color=INK_2)
        ax.set_ylabel(label, color=INK_2)
        ax.set_title(f"MVTec AD macro {label.split(' (')[0]} vs k", color=INK, fontsize=11, loc="left")
        ax.grid(True, color=GRID, lw=0.8)
        ax.tick_params(colors=INK_2)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        lo = min(mc["mean"].min() - mc["std"].max(), min((pp[("MEAN", k)][0] for k in pk), default=100))
        ax.set_ylim(np.floor(lo - 0.6), 100)
        ax.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=INK_2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def latency_table(lat: pd.DataFrame, device_name: str) -> str:
    names = {"tensor_to_score_heatmap": "(a) preprocessed tensor → score + full-res heatmap",
             "end_to_end": "(b) image file → decode + preprocess + score + heatmap"}
    lines = ["| Resolution | Path | Median (ms) | p95 (ms) | n | Device | kNN dtype |", "|---:|---|---:|---:|---:|---|---|"]
    for _, r in lat.sort_values(["resolution", "mode"], ascending=[False, False]).iterrows():
        lines.append(f"| {r.resolution} | {names.get(r['mode'], r['mode'])} | {r.median_ms:.1f} | {r.p95_ms:.1f} "
                     f"| {r.n} | {device_name} | {r.knn_dtype} |")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- calibration
POLICY_LABEL = {"good_only": "good_only (p99 × 1.10)", "good_only_m0": "good_only_m0 (p99)",
                "synth_youden": "synth_youden", "synth_escape5": "synth_escape5", "oracle": "oracle †"}
SYNTH_POLICIES = ("synth_youden", "synth_escape5")


DEV_CATEGORIES = ("bottle", "cable", "capsule")  # protocol amendment 1: first three in MVTec order


def calibration_macro(cal: pd.DataFrame, categories: list[str] | None = None) -> pd.DataFrame:
    """Per (k, policy): macro over categories per seed, then mean/std over seeds (percent)."""
    if categories is not None:
        cal = cal[cal.category.isin(categories)]
    per_seed = cal.groupby(["k", "policy", "seed"])[["frr", "escape", "balanced_accuracy"]].mean().reset_index()
    g = per_seed.groupby(["k", "policy"])
    out = g[["frr", "escape", "balanced_accuracy"]].mean().mul(100)
    std = g[["frr", "escape", "balanced_accuracy"]].std(ddof=0).mul(100).add_suffix("_std")
    n_cat = cal.groupby(["k", "policy"]).category.nunique().rename("n_categories")
    return pd.concat([out, std, n_cat], axis=1).reset_index()


def _label_points(ax: object, pts: pd.DataFrame, min_gap: float = 7.0, x_cluster: float = 25.0) -> None:
    """Non-overlapping labels: cluster points by FRR, push labels upward within a cluster, add leader lines."""
    pts = pts.sort_values("frr")
    clusters: list[list[pd.Series]] = []
    for _, r in pts.iterrows():
        if clusters and r.frr - clusters[-1][-1].frr < x_cluster:
            clusters[-1].append(r)
        else:
            clusters.append([r])
    for cluster in clusters:
        prev = -np.inf
        for r in sorted(cluster, key=lambda q: q.escape):
            y = max(float(r.escape), prev + min_gap)
            prev = y
            ax.annotate(PRETTY.get(r.category, r.category), xy=(r.frr, r.escape), xytext=(r.frr + 9, y),
                        textcoords="data", fontsize=6.5, color=INK_2, va="center", annotation_clip=False,
                        arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.5, "shrinkA": 0, "shrinkB": 3})


def calibration_scatter(cal: pd.DataFrame, k: int, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    policies = [p for p in POLICY_LABEL if p in set(cal.policy)]
    d = cal[cal.k == k].groupby(["policy", "category"])[["frr", "escape"]].mean().mul(100).reset_index()
    fig, axes = plt.subplots(1, len(policies), figsize=(2.9 * len(policies), 3.2), sharex=True, sharey=True,
                             facecolor=SURFACE)
    for ax, p in zip(np.atleast_1d(axes), policies, strict=True):
        dp = d[d.policy == p]
        ax.set_facecolor(SURFACE)
        ax.plot([0, 100], [100, 0], color=GRID, lw=1, ls=":")  # BA = 50 % line
        ax.scatter(dp.frr, dp.escape, s=40, color=SERIES_1 if p != "oracle" else PAPER,
                   edgecolor=SURFACE, linewidth=1.5, zorder=3)
        _label_points(ax, dp[(dp.frr > 30) | (dp.escape > 30)])
        ax.set_title(POLICY_LABEL[p], fontsize=9, color=INK, loc="left")
        ax.set_xlim(-3, 103)
        ax.set_ylim(-3, 103)
        ax.grid(True, color=GRID, lw=0.6)
        ax.tick_params(colors=INK_2, labelsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_xlabel("FRR (%)", color=INK_2, fontsize=8)
    np.atleast_1d(axes)[0].set_ylabel("Escape rate (%)", color=INK_2, fontsize=8)
    fig.suptitle(f"Per-category FRR vs escape at k = {k} (mean of 3 seeds; lower-left is better; † non-deployable)",
                 fontsize=9, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def calibration_section(results: Path) -> str:
    from scipy.stats import spearmanr

    cal = pd.read_csv(results / "calibration.csv")
    synth = pd.read_csv(results / "calibration_synth.csv")
    mac = calibration_macro(cal)
    ks = sorted(cal.k.unique())
    policies = [p for p in POLICY_LABEL if p in set(cal.policy)]

    def cell(k: int, p: str) -> str:
        r = mac[(mac.k == k) & (mac.policy == p)].iloc[0]
        return f"**{r.balanced_accuracy:.1f}** · {r.frr:.1f} / {r.escape:.1f}"

    lines = ["| k | " + " | ".join(POLICY_LABEL[p] for p in policies) + " |", "|---:|" + "---|" * len(policies)]
    lines += [f"| {k}{' ‡' if k == 1 else ''} | " + " | ".join(cell(k, p) for p in policies) + " |" for k in ks]

    # gate: every synthetic policy beats good_only on macro BA for every k >= 2
    gate_rows = []
    for k in [k for k in ks if k >= 2]:
        base = mac[(mac.k == k) & (mac.policy == "good_only")].balanced_accuracy.iloc[0]
        for p in SYNTH_POLICIES:
            v = mac[(mac.k == k) & (mac.policy == p)].balanced_accuracy.iloc[0]
            gate_rows.append((k, p, v - base))
    gate_ok = all(d > 0 for _, _, d in gate_rows)
    gate_txt = ", ".join(f"k={k} {p} {d:+.1f}" for k, p, d in gate_rows)

    # amendment 1: select one synthetic policy on the dev categories at k = 4, report on held-out ones
    cats = sorted(cal.category.unique())
    held = [c for c in cats if c not in DEV_CATEGORIES]
    dev_mac, held_mac = calibration_macro(cal, list(DEV_CATEGORIES)), calibration_macro(cal, held)
    k_sel = 4 if 4 in ks else ks[-1]
    d4 = dev_mac[dev_mac.k == k_sel].set_index("policy")
    selected = max(SYNTH_POLICIES, key=lambda p: d4.loc[p, "balanced_accuracy"])
    summary = pd.concat([mac.assign(subset="all"), dev_mac.assign(subset="dev"), held_mac.assign(subset="heldout")])
    summary.to_csv(results / "summary_calibration.csv", index=False)

    h4 = held_mac[held_mac.k == k_sel].set_index("policy")
    a4 = mac[mac.k == k_sel].set_index("policy")
    b, g, o = h4.loc[selected], h4.loc["good_only"], h4.loc["oracle"]
    head = (f"**With only {k_sel} good reference photos and zero real defect images, the selected `{selected}` cutoff "
            f"gives a macro false-reject rate of {b.frr:.1f} % and an escape rate of {b.escape:.1f} % on the real "
            f"test sets of the {len(held)} held-out MVTec categories (balanced accuracy {b.balanced_accuracy:.1f} % "
            f"vs {g.balanced_accuracy:.1f} % for the defect-free `good_only` baseline and {o.balanced_accuracy:.1f} % "
            f"for the non-deployable oracle).** On all 15 categories: FRR {a4.loc[selected, 'frr']:.1f} %, escape "
            f"{a4.loc[selected, 'escape']:.1f} %, balanced accuracy {a4.loc[selected, 'balanced_accuracy']:.1f} %.")
    sel_txt = (f"Policy selection (protocol amendment 1): on the dev categories {', '.join(DEV_CATEGORIES)} at "
               f"k = {k_sel}, macro balanced accuracy was "
               + ", ".join(f"`{p}` {d4.loc[p, 'balanced_accuracy']:.1f} %" for p in SYNTH_POLICIES)
               + f" ⇒ **`{selected}`** (also the demo app's default).")

    def held_cell(k: int, p: str) -> str:
        r = held_mac[(held_mac.k == k) & (held_mac.policy == p)].iloc[0]
        return f"**{r.balanced_accuracy:.1f}** · {r.frr:.1f} / {r.escape:.1f}"

    held_lines = ["| k | " + " | ".join(POLICY_LABEL[p] for p in ("good_only", selected, "oracle")) + " |",
                  "|---:|---|---|---|"]
    held_lines += [f"| {k}{' ‡' if k == 1 else ''} | " + " | ".join(held_cell(k, p) for p in
                                                                  ("good_only", selected, "oracle")) + " |"
                   for k in ks]

    # synthetic vs real AUROC
    rho_rows = []
    for k in ks:
        sk = synth[synth.k == k]
        s = sk.groupby("category")[["synth_auroc", "real_auroc"]].mean()
        rho, pval = spearmanr(s.synth_auroc, s.real_auroc)
        ratio = float((sk.synth_good_median / sk.real_good_median).median())
        rho_rows.append(f"| {k} | {s.synth_auroc.mean() * 100:.1f} | {s.real_auroc.mean() * 100:.1f} | "
                        f"{rho:+.2f} (p = {pval:.2g}) | {ratio:.2f} |")
    calibration_scatter(cal, 4 if 4 in ks else ks[0], results / "frr_escape_k4.png")
    return "\n\n".join([
        "### Deployable cutoff from synthetic defects (FRR / escape on real defects)",
        head,
        sel_txt,
        f"Held-out categories ({', '.join(PRETTY.get(c, c) for c in held)}) — cells: **balanced accuracy** · "
        "FRR / escape rate (%), macro mean, mean of 3 seeds:",
        "\n".join(held_lines),
        "All policies, all 15 categories. Cells: **balanced accuracy** · FRR / escape rate, all in %, macro mean "
        "over 15 categories, mean of 3 "
        "seeds. Cutoffs use only the k good references (protocol frozen in `docs/calibration_protocol.md`). "
        "† `oracle` = Youden cutoff on the real test labels — an upper bound, not deployable. "
        "‡ k = 1 has no leave-one-out: good copies are augmentations of the bank image itself, so calibration "
        "is weaker.",
        "\n".join(lines),
        f"Gate (each synthetic policy beats `good_only` on macro balanced accuracy for every k ≥ 2): "
        f"**{'PASS' if gate_ok else 'FAIL'}** — {gate_txt} (pts).",
        "![FRR vs escape at k=4](results/frr_escape_k4.png)",
        "Does the synthetic task predict real difficulty? Synthetic AUROC (synthetic good vs synthetic defect) vs "
        "real image AUROC, macro and Spearman rank correlation across the 15 categories:",
        "| k | synthetic AUROC (%) | real AUROC (%) | Spearman ρ | synthetic-good ÷ real-good median score |\n"
        "|---:|---:|---:|---|---:|\n" + "\n".join(rho_rows),
        "Last column: per run, median calibration-good score divided by median real-good test score (median over "
        "the 45 runs per k). Above 1 ⇒ calibration goods look *worse* than real goods (good-only cutoffs too "
        "high ⇒ escapes); below 1 ⇒ too optimistic (cutoffs too low ⇒ false rejects).",
    ])


# ----------------------------------------------------------------------------- failure analysis / app
def failure_section(results: Path, bench: pd.DataFrame, n_worst: int = 3) -> str:
    types = pd.read_csv(results / "defect_types.csv")
    k = int(types.k.iloc[0])
    img = bench[(bench.metric == "image_auroc") & (bench.k == k)].groupby("category").value.mean().mul(100)
    worst = img.sort_values().index[:n_worst].tolist()
    blocks = [f"### Failure analysis — the {n_worst} weakest categories at k = {k}",
              "Per defect type: image AUROC of that type against all good test images (mean of 3 seeds), and the "
              "median ground-truth defect size — area in % of the image and equivalent diameter in DINOv2 patches "
              f"(one patch = 14 px at {int(bench.resolution.iloc[0])} px input). Grids: top row = test image with "
              "ground-truth outline, bottom row = heatmap (k = 4, seed 0)."]
    for cat in worst:
        t = types[types.category == cat].sort_values("auroc_vs_good")
        rows = "\n".join(f"| {r.defect_type} | {r.auroc_vs_good * 100:.1f} | {r.n} | {r.median_area_pct:.2f} | "
                         f"{r.median_diam_patches:.1f} |" for r in t.itertuples())
        blocks.append(f"**{PRETTY.get(cat, cat)}** — image AUROC {img[cat]:.1f} %\n\n"
                      "| defect type | AUROC vs good (%) | n | median area (%) | median diameter (patches) |\n"
                      "|---|---:|---:|---:|---:|\n" + rows
                      + f"\n\n![{cat} failure cases](docs/failure_{cat}.png)")
    return "\n\n".join(blocks)


APP_CPU_DEFAULT = (448, 32)  # (resolution, n_cal) — keep in sync with app/streamlit_app.py


def app_section(results: Path) -> str:
    d = pd.read_csv(results / "app_setup_cpu.csv")
    meta_path = results / "app_setup_cpu.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    g = d.groupby(["resolution", "n_cal", "threads"]).total_s.agg(["median", "max"]).reset_index()
    rows = "\n".join(
        f"| {int(r.resolution)} | {int(r.n_cal)} | {int(r.threads)} | {r['median']:.1f} | {r['max']:.1f} |"
        + (" ← app CPU default" if (int(r.resolution), int(r.n_cal)) == APP_CPU_DEFAULT else "")
        for _, r in g.iterrows())
    cats = ", ".join(sorted(d.category.unique()))
    return ("### Demo-app setup time on CPU (4 references → fitted + calibrated inspector)\n\n"
            f"Measured with `scripts/time_app_setup.py` on {meta.get('device_name', 'CPU')} over {cats}; model "
            f"already loaded (first load {meta.get('model_load_s', float('nan')):.1f} s). `n_cal` = synthetic "
            "images per class; 8 rotations per reference.\n\n"
            "| resolution | n_cal | CPU threads | median (s) | max (s) | |\n|---:|---:|---:|---:|---:|---|\n" + rows)


# ----------------------------------------------------------------------------- assembly
def build(results: Path, paper_csv: Path, readme: Path) -> str:
    bench = pd.read_csv(results / "benchmark.csv")
    paper = pd.read_csv(paper_csv)
    meta = json.loads((results / "run_meta.json").read_text())
    resolution = int(bench.resolution.iloc[0])
    mc_img, mc_pix = macro(bench, "image_auroc"), macro(bench, "pixel_auroc")
    p_img = paper_lookup(paper, "tab:MVTec-detailed-agnostic", resolution, "image_auroc")
    p_pix = paper_lookup(paper, "tab:results_MVTec", resolution, "pixel_auroc")
    summary = []
    for metric, mc, pp in (("image_auroc", mc_img, p_img), ("pixel_auroc", mc_pix, p_pix)):
        for _, r in mc.iterrows():
            paper_mean, paper_std = pp.get(("MEAN", int(r.k)), (np.nan, np.nan))
            summary.append({"metric": metric, "k": int(r.k), "mean": r["mean"], "std": r["std"],
                            "n_seeds": int(r.n_seeds), "n_categories": int(r.n_cat), "resolution": resolution,
                            "paper_mean": paper_mean, "paper_std": paper_std})
    pd.DataFrame(summary).to_csv(results / "summary_auroc.csv", index=False)
    n_cat, ks = bench.category.nunique(), sorted(bench.k.unique())
    seeds = sorted(bench.seed.unique())
    chart(bench, paper, resolution, results / "auroc_vs_k.png")

    def row(mc: pd.DataFrame, k: int) -> pd.Series:
        return mc[mc.k == k].iloc[0]

    parts: list[str] = []
    parts.append(
        f"_Generated by `scripts/report.py` from `results/benchmark.csv` (run {meta['timestamp_utc']}, "
        f"commit `{short_sha(meta['git_sha'])}`, {meta['device_name']}, torch {meta['torch_version']}). "
        f"AnomalyDINO-S defaults at {resolution} px; {n_cat} categories × k ∈ {{{', '.join(map(str, ks))}}} "
        f"× seeds {{{', '.join(map(str, seeds))}}}; pixel AUROC at original image resolution._")
    head = []
    for k in (1, 4):
        if k in ks:
            r, rp = row(mc_img, k), row(mc_pix, k) if k in set(mc_pix.k) else None
            ptxt = f" (paper {pm(*p_img[('MEAN', k)])})" if ("MEAN", k) in p_img else ""
            pixtxt = (f", pixel AUROC **{pm(rp['mean'], rp['std'])}**"
                      + (f" (paper {pm(*p_pix[('MEAN', k)])})" if ("MEAN", k) in p_pix else "")) if rp is not None else ""
            head.append(f"- **{k}-shot**: image AUROC **{pm(r['mean'], r['std'])}**{ptxt}{pixtxt}")
    parts.append("### Headline\n\n" + "\n".join(head))

    # gap analysis vs paper (macro image AUROC)
    gaps = []
    for k in ks:
        if ("MEAN", k) in p_img:
            gaps.append((k, row(mc_img, k)["mean"] - p_img[("MEAN", k)][0]))
    pc_img = per_category(bench, "image_auroc")
    pc_img["paper"] = [p_img.get((c, k), (np.nan, np.nan))[0] for c, k in zip(pc_img.category, pc_img.k, strict=True)]
    pc_img["gap"] = (pc_img["mean"] - pc_img["paper"]).abs()
    worst_gap = pc_img.loc[pc_img.gap.idxmax()]
    big = [f"k={k}: {g:+.2f}" for k, g in gaps if abs(g) > 1.0]
    parts.append("Gap to paper (macro image AUROC, ours − paper): " + ", ".join(f"k={k} {g:+.2f}" for k, g in gaps)
                 + (f". **Gaps > 1 pt: {', '.join(big)}** — see *Paper comparison*." if big else
                    ". All within 1 pt.")
                 + f" Largest per-category gap over all {len(pc_img)} (category, k) cells: "
                   f"{worst_gap['gap']:.2f} pts ({PRETTY.get(worst_gap.category, worst_gap.category)}, "
                   f"k={int(worst_gap.k)}; the paper reports values rounded to 0.1).")
    parts.append(f"![AUROC vs k]({'results/auroc_vs_k.png'})")
    parts.append(f"### Image AUROC (%) per category — ours vs AnomalyDINO-S ({resolution}) paper\n\n"
                 + image_table(bench, paper, resolution))
    parts.append("### Pixel AUROC (%) per category\n\nThe paper publishes only the macro mean of pixel AUROC "
                 "(no per-category values).\n\n" + pixel_table(bench, paper, resolution))
    lat_path = results / "latency.csv"
    if lat_path.exists():
        lat = pd.read_csv(lat_path)
        lat_meta_path = results / "run_meta_latency.json"
        lat_meta = json.loads(lat_meta_path.read_text()) if lat_meta_path.exists() else meta
        parts.append("### Latency (batch 1, k = 1, 10 warm-up runs, device-synchronised)\n\n"
                     + latency_table(lat, lat_meta["device_name"])
                     + f"\n\nFeatures in float32. n = {int(lat.n.iloc[0])} test images: up to 50 per category "
                       "(fewer where the test set is smaller), one memory bank per category. Path (a) is the "
                       "model work; (b) adds PNG decoding and resizing on the CPU. The paper reports 60 ms (448) / "
                       "86 ms (672) per image on an NVIDIA A40 for its timer (decode → score, *without* heatmap "
                       "upsampling); no CUDA GPU was used here.")
    if (results / "calibration.csv").exists():
        parts.append(calibration_section(results))
    if (results / "defect_types.csv").exists():
        parts.append(failure_section(results, bench))
    if (results / "app_setup_cpu.csv").exists():
        parts.append(app_section(results))
    content = "\n\n".join(parts) + "\n"
    (results / "RESULTS.md").write_text("# Results\n\n" + content.replace("](results/", "]("))
    inject(readme, content)
    return content


def inject(readme: Path, content: str) -> None:
    text = readme.read_text()
    if START not in text or END not in text:
        raise ValueError(f"{readme} lacks {START} / {END} markers")
    new = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _: f"{START}\n{content}{END}", text, flags=re.S)
    readme.write_text(new)
    LOG.info("injected results into %s", readme)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Generate tables/figures from results and inject into README.")
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--paper", type=Path, default=Path("docs/paper_numbers.csv"))
    ap.add_argument("--readme", type=Path, default=Path("README.md"))
    args = ap.parse_args()
    build(args.results, args.paper, args.readme)


if __name__ == "__main__":
    main()
