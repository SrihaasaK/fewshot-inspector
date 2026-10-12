# Synthetic-defect cutoff calibration — frozen protocol

Written **before any calibration result was computed** and committed before the full calibration run.
Every parameter below is fixed a priori; none is tuned on MVTec test labels (or on anything else). If a
later change is needed it is logged here with the reason, and all calibration results are re-run.

Disclosure: after this protocol was written and before it was committed, one smoke test (bottle + screw,
k = 1, seed 0) was run to check the code end to end; it surfaced only an output-directory bug. No
parameter was changed after it.

## Problem

AUROC is threshold-free, but a production line needs **one cutoff** (FAIL iff `score > cutoff`)
and, at setup time, has only the k good reference images — no defect photos. We choose the cutoff
from the references alone and then measure on the *real* MVTec test set:

- **FRR** (false-reject rate) = good parts flagged / good parts,
- **escape rate** = defective parts passed / defective parts,
- **balanced accuracy** = 1 − (FRR + escape) / 2.

## Calibration scores (from the k references only)

The inspector configuration is identical to the deployed one (AnomalyDINO-S, 672 px in the
benchmark, rotations + masking per `docs/method_notes.md`). Reference-view features are reused.

- **k ≥ 2 — leave-one-out.** For each reference *i*: memory bank = the other k − 1 references
  (with their rotation views); score *n* augmented good copies of reference *i* and *n* synthetic-defect
  copies of reference *i*.
- **k = 1.** Bank = the single reference; goods = augmented copies of it; defects = synthetic defects
  on augmented copies. Weaker by construction: the good copies are near-duplicates of a bank image.
- **Calibration-set size.** n = ⌈64 / k⌉ per reference, so ≈ 64 goods + ≈ 64 defects per (category, k, seed).
- **Pairing.** Defect copy *j* = synthetic defect applied to good copy *j* (same augmentation), so the
  two classes differ only by the defect.
- **Seeding.** `numpy.random.default_rng([seed, i, k])` per reference *i* — fully reproducible.

### Good-copy augmentation ("nominal variation" proxy)

| Op | Range | Note |
|---|---|---|
| brightness (multiplicative) | U(0.9, 1.1) | |
| contrast (about the image mean) | U(0.9, 1.1) | |
| Gaussian pixel noise | σ = 2 grey levels | |
| shift | U(−3 %, +3 %) of the shorter side, each axis | reflect-101 border |
| rotation | U(−10°, +10°) if the inspector rotates references (`agnostic`: all categories), else 0° | reflect-101 border |

### Synthetic defect generators (sizes relative to the shorter image side *s*)

All generators return `(image, defect_mask)` and are cycled in fixed order
`scratch, cutpaste, cutpaste_scar, stain` across the n copies. The defect centre is drawn from the
**placement region**: the inspector's own foreground mask of the (augmented) reference when masking
is on for the category, otherwise the whole image (textures and unmasked objects).

| Generator | Parameters |
|---|---|
| `scratch` | cubic Bézier, chord length U(0.08, 0.30)·s, control points offset ⟂ by U(−0.2, 0.2)·length; width U(1, 4) px at s = 1024 (≥ 1 px); colour = local mean ± U(40, 90) grey levels (darker or brighter, p = ½); Gaussian blur σ = max(0.5, width/3) on the stroke alpha |
| `cutpaste` | patch from the same image, area U(2 %, 15 %) of the image, aspect ratio log-U(0.3, 3.3); colour jitter (brightness ×U(0.9, 1.1), saturation ×U(0.9, 1.1), hue ± U(0, 5)°); rotation U(−45°, 45°); pasted at a new centre |
| `cutpaste_scar` | thin rectangle, width U(2, 16)/256·s, length U(10, 25)/256·s, same jitter/rotation as `cutpaste` |
| `stain` | irregular blob: radius U(0.03, 0.10)·s, smooth noise × radial fall-off thresholded; random RGB tint; opacity U(0.3, 0.7); feathered edge (Gaussian σ = 0.15·radius); mask = feathered alpha > ½ of its peak |

## Cutoff policies

Flag iff `score > cutoff`. All thresholds are computed on the calibration scores of the same
(category, k, seed) run.

| Policy | Rule | Deployable? |
|---|---|---|
| `good_only` | 99th percentile (linear interpolation) of synthetic-good scores × (1 + 0.10) | yes — needs no defects at all |
| `good_only_m0` | same with margin 0 (sensitivity variant, reported for transparency) | yes |
| `synth_youden` | maximise TPR − FPR on synthetic good vs synthetic defect; candidates = midpoints between consecutive distinct pooled scores (plus min − ε, max + ε); ties → median candidate | yes |
| `synth_escape5` | highest cutoff with ≤ 5 % synthetic escapes: with sorted defect scores d and m = ⌊0.05·n_d⌋ allowed escapes, cutoff = the float just below d₍ₘ₊₁₎ | yes |
| `oracle` | Youden on the **real test labels** (same tie rule) | **no — upper bound only** |

The `good_only` margin 0.10 is a conventional 10 % guard band chosen before seeing any result.

## Evaluation

- Real test scores come from the benchmark run (`results/raw/scores_672.csv`), i.e. the deployed
  inspector with all k references in the bank.
- Per (category, k, seed, policy): FRR, escape rate, balanced accuracy on the real test set.
- Macro mean over the 15 categories per seed, then mean ± std over seeds.
- Synthetic AUROC (synthetic good vs synthetic defect) per (category, k, seed), and the Spearman
  rank correlation across categories between synthetic and real image AUROC (per k, seed-averaged).
- **Gate**: `synth_youden` and `synth_escape5` beat `good_only` on macro balanced accuracy for every
  k ≥ 2. If not, the result is reported as-is with analysis.

## Amendment 1 (2026-10-12 15:30 local, during the full run) — choosing *one* deployable policy

Naming a single "best" synthetic policy by its MVTec test performance would select on test labels. Rule:
among `synth_youden` and `synth_escape5`, the **selected policy** is the one with the higher macro balanced
accuracy at k = 4 on the **3 dev categories `bottle`, `cable`, `capsule`** (the first three in MVTec order —
a mechanical choice). It is reported on the **12 held-out categories** (and, separately, on all 15), and it
becomes the demo app's default policy. All policies are still reported in full.

Disclosure: when this rule was written, the run had logged bottle k ∈ {1, 2, 4} (some seeds), and a headless
app smoke test on bottle had been run; no other category's calibration result had been seen. No protocol
parameter was changed.

## Known biases (reported in the README)

- **LOO bias**: calibration banks have k − 1 references, the deployed bank has k ⇒ real good scores
  are typically lower than calibration goods ⇒ cutoffs err towards lower FRR / higher escape.
- **Synthetic ≠ real**: generators approximate scratches, misplaced material and stains; logical
  defects (missing parts, wrong component, misplacement) are not modelled.
- **k = 1**: good copies are augmentations of the bank image itself ⇒ optimistic good scores ⇒
  cutoffs too low ⇒ higher FRR.
