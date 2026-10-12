# Third-party notices

The code in this repository is MIT-licensed (see `LICENSE`) **except** for the portions listed below, which
are adapted from third-party code and remain under their original licences. Full licence texts are in
`LICENSES/`. Nothing is imported from these projects at runtime; DINOv2 is downloaded by `torch.hub` at
runtime and is not redistributed here.

## AnomalyDINO — Apache License 2.0

Damm, Laszkiewicz, Lederer, Fischer. <https://github.com/dammsi/AnomalyDINO>, commit
`b9d1c2648e3a5247437d4d953d907a8f3d994457`. Licence: `LICENSES/Apache-2.0-AnomalyDINO.txt`
(the project ships no NOTICE file).

| Our file | What | Relationship | Changes made |
|---|---|---|---|
| `inspector/metrics.py` — `aupro`, `_trapezoid` | PRO-curve / AUPRO computation and bounded trapezoid integration | **adapted** from `src/post_eval.py` (`compute_pro`, `trapezoid`), which itself builds on the MVTec evaluation code below | rewritten into one function over stacked arrays; renamed variables; removed printing / TIFF I/O; stable sort; returns the normalised AUPRO |
| `tests/test_components.py` — `ref_mean_top1p`, `ref_dists2map`, `ref_background_mask` (+ the inline rotation check) | reference behaviour used as **test oracles** | **copied with minimal edits** from `src/post_eval.py` (`mean_top1p`), `src/utils.py` (`dists2map`, `rotate_image`) and `src/backbones.py` (`compute_background_mask`) | wrapped as stand-alone functions; `svd_solver="full"` instead of `"randomized"` for determinism; type hints |
| `inspector/masking.py`, `inspector/inspector.py` (`rotate_image`, `top_pct_mean`, `patch_to_heatmap`), `inspector/backbone.py` (`Preprocessor`), `inspector/data.py` (category order, contiguous k-shot slicing) | the AnomalyDINO pipeline | **independent re-implementation** (no code copied) written to reproduce the reference behaviour exactly, including the flattened-morphology quirk; every default is cited in `docs/method_notes.md` | batched PyTorch implementation; exact torch kNN instead of faiss; SciPy-equivalent Gaussian in torch |

## MVTec AD evaluation code — BSD 3-Clause

Copyright 2022 MVTec Software GmbH. `mvtec_ad_evaluation.tar.xz` from the official MVTec AD page.
Licence (copyright notice, conditions and disclaimer reproduced in full):
`LICENSES/BSD-3-Clause-MVTec-AD-evaluation.txt`.

| Our file | Derived from |
|---|---|
| `inspector/metrics.py` — `aupro` | `pro_curve_util.py` → `compute_pro` (via AnomalyDINO `src/post_eval.py`) |
| `inspector/metrics.py` — `_trapezoid` | `generic_util.py` → `trapezoid` (via AnomalyDINO `src/post_eval.py`) |

## Data

The MVTec AD dataset is **not** included. The 20 downscaled demo images in `app/samples/` (with their own
notice, `app/samples/LICENSE.md`, and provenance manifest) and the figures under `docs/` and `results/`
that show MVTec AD images are derived from the dataset (© MVTec Software GmbH, CC BY-NC-SA 4.0) and are
shared under that licence.
