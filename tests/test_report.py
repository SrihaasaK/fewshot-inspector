"""Statistics and README injection in scripts/report.py (no data, no weights)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import report


def _bench() -> pd.DataFrame:
    rows = []
    vals = {("a", 0): 0.90, ("a", 1): 0.94, ("b", 0): 1.00, ("b", 1): 0.98}
    for (cat, seed), v in vals.items():
        rows.append({"category": cat, "k": 1, "seed": seed, "resolution": 672, "metric": "image_auroc", "value": v})
    return pd.DataFrame(rows)


def test_macro_is_per_seed_mean_then_population_std():
    m = report.macro(_bench(), "image_auroc").iloc[0]
    per_seed = np.array([(0.90 + 1.00) / 2, (0.94 + 0.98) / 2]) * 100
    assert m["mean"] == pytest.approx(per_seed.mean())
    assert m["std"] == pytest.approx(per_seed.std(ddof=0))


def test_per_category_mean_std():
    pc = report.per_category(_bench(), "image_auroc").set_index("category")
    assert pc.loc["a", "mean"] == pytest.approx(92.0)
    assert pc.loc["a", "std"] == pytest.approx(2.0)


def test_calibration_macro_subset():
    cal = pd.DataFrame([
        {"category": c, "k": 4, "seed": s, "policy": "p", "frr": f, "escape": e, "balanced_accuracy": 1 - (f + e) / 2}
        for c, s, f, e in [("x", 0, 0.1, 0.2), ("y", 0, 0.3, 0.0), ("x", 1, 0.1, 0.0), ("y", 1, 0.1, 0.2)]
    ])
    all_ = report.calibration_macro(cal).iloc[0]
    only_x = report.calibration_macro(cal, ["x"]).iloc[0]
    assert all_.frr == pytest.approx(15.0) and only_x.frr == pytest.approx(10.0)
    assert int(only_x.n_categories) == 1


def test_inject_replaces_only_between_markers(tmp_path: Path):
    readme = tmp_path / "README.md"
    readme.write_text(f"head\n{report.START}\nold\n{report.END}\ntail\n")
    report.inject(readme, "new $1 \\1 content\n")  # regex-special characters must survive verbatim
    text = readme.read_text()
    assert text.startswith("head\n") and text.endswith("tail\n")
    assert "old" not in text and "new $1 \\1 content" in text
    (tmp_path / "bad.md").write_text("no markers")
    with pytest.raises(ValueError, match="markers"):
        report.inject(tmp_path / "bad.md", "x")


def test_short_sha_keeps_dirty_marker():
    assert report.short_sha("0123456789abcdef0123") == "0123456789ab"
    assert report.short_sha("0123456789abcdef0123-dirty") == "0123456789ab-dirty"
