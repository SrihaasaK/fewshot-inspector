"""Extract AnomalyDINO's published MVTec AD numbers from the arXiv LaTeX source.

Writes ``docs/paper_numbers.csv`` (long format) so that no paper number is ever
typed by hand. Source: arXiv 2405.14529v3 (WACV 2025 camera-ready), tables
``tables/MVTec_per_object.tex`` (Tab. "Detailed results on MVTec-AD",
AnomalyDINO-S (672), agnostic preprocessing) and ``tables/full_results_MVTec.tex``
(Tab. "Anomaly detection on MVTec-AD", macro means for 448 and 672).

Usage:
    python scripts/extract_paper_numbers.py [--tex-dir DIR] [--out docs/paper_numbers.csv]

If ``--tex-dir`` is omitted, the source tarball is downloaded from arXiv into a
temporary directory.
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import re
import tarfile
import tempfile
import urllib.request
from pathlib import Path

LOG = logging.getLogger("extract_paper_numbers")

ARXIV_SRC = "https://arxiv.org/e-print/2405.14529v3"
SHOTS = [1, 2, 4, 8, 16]
CATEGORY_MAP = {
    "Bottle": "bottle",
    "Cable": "cable",
    "Capsule": "capsule",
    "Carpet": "carpet",
    "Grid": "grid",
    "Hazelnut": "hazelnut",
    "Leather": "leather",
    "Metal nut": "metal_nut",
    "Pill": "pill",
    "Screw": "screw",
    "Tile": "tile",
    "Toothbrush": "toothbrush",
    "Transistor": "transistor",
    "Wood": "wood",
    "Zipper": "zipper",
    "Mean": "MEAN",
}
VALUE_RE = re.compile(r"(\d+\.\d)\\std\{(\d+\.\d)\}")


def _clean(cell: str) -> str:
    """Strip LaTeX decoration (bold, underline, cell colour) from a table row."""
    cell = re.sub(r"\\cellcolor\{[^}]*\}", "", cell)
    return cell.replace("\\textbf", "").replace("\\ul", "")


def parse_per_object(tex: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in tex.splitlines():
        if line.lstrip().startswith("%"):
            continue  # commented-out (superseded) rows
        name = line.split("&")[0].strip()
        if name not in CATEGORY_MAP:
            continue
        values = VALUE_RE.findall(_clean(line))
        if len(values) != 10:
            raise ValueError(f"expected 10 values in per-object row {name!r}, got {len(values)}")
        for i, k in enumerate(SHOTS):
            for j, metric in enumerate(["image_auroc", "pro"]):
                mean, std = values[2 * i + j]
                rows.append(
                    {
                        "source": "tab:MVTec-detailed-agnostic",
                        "resolution": 672,
                        "k": k,
                        "category": CATEGORY_MAP[name],
                        "metric": metric,
                        "mean": float(mean),
                        "std": float(std),
                    }
                )
    return rows


def parse_full_results(tex: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    shot: int | None = None
    metrics = ["image_auroc", "image_f1max", "image_ap", "pixel_auroc", "pixel_f1max", "pro"]
    for line in tex.splitlines():
        if line.lstrip().startswith("%"):
            continue
        m = re.search(r"\\textbf\{(\d+)-shot\}", line)
        if m:
            shot = int(m.group(1))
            continue
        m = re.match(r"\s*\\method\{\}-S \\scriptsize\{\((\d+)\)\}", line)
        if not m or shot is None:
            continue
        values = VALUE_RE.findall(_clean(line))
        if len(values) != 6:
            raise ValueError(f"expected 6 values for {shot}-shot S({m.group(1)}), got {len(values)}")
        for metric, (mean, std) in zip(metrics, values, strict=True):
            rows.append(
                {
                    "source": "tab:results_MVTec",
                    "resolution": int(m.group(1)),
                    "k": shot,
                    "category": "MEAN",
                    "metric": metric,
                    "mean": float(mean),
                    "std": float(std),
                }
            )
    return rows


def _download_source(dest: Path) -> Path:
    LOG.info("downloading %s", ARXIV_SRC)
    with urllib.request.urlopen(ARXIV_SRC, timeout=120) as resp:
        data = resp.read()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        members = [m for m in tar.getmembers() if m.name.startswith("tables/") and m.isfile()]
        tar.extractall(dest, members=members, filter="data")
    return dest


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Extract AnomalyDINO MVTec numbers from arXiv LaTeX.")
    ap.add_argument("--tex-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("docs/paper_numbers.csv"))
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        tex_dir = args.tex_dir or _download_source(Path(tmp))
        per_object = (tex_dir / "tables" / "MVTec_per_object.tex").read_text()
        full = (tex_dir / "tables" / "full_results_MVTec.tex").read_text()
    rows = parse_per_object(per_object) + parse_full_results(full)
    expected = 16 * 5 * 2 + 5 * 2 * 6  # per-object (15 cats + mean) + macro table (2 resolutions)
    if len(rows) != expected:
        raise ValueError(f"unexpected row count {len(rows)} (expected {expected})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["source", "resolution", "k", "category", "metric", "mean", "std"]
        )
        writer.writeheader()
        writer.writerows(rows)
    LOG.info("wrote %d rows to %s", len(rows), args.out)


if __name__ == "__main__":
    main()
