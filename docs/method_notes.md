# AnomalyDINO — method notes (ground truth for this reimplementation)

Sources read on 2026-10-12:

- **Paper**: Damm, Laszkiewicz, Lederer, Fischer. *AnomalyDINO: Boosting Patch-based
  Few-shot Anomaly Detection with DINOv2*. WACV 2025. arXiv 2405.14529**v3** — cited by
  LaTeX source line, `paper:L<n>` = `arxiv_paper_v3.tex` line *n*; tables cited by file.
- **Reference code**: <https://github.com/dammsi/AnomalyDINO> @ `b9d1c2648e3a5247437d4d953d907a8f3d994457`
  (2025-12-17) — cited as `file:line`.
- **DINOv2 hub code**: `facebookresearch/dinov2` (torch.hub `main`), `dinov2/models/vision_transformer.py`.

`docs/paper_numbers.csv` is produced from the paper's LaTeX tables by
`scripts/extract_paper_numbers.py` (no number typed by hand).

## 1. Which configuration gives "96.6 % 1-shot"?

**AnomalyDINO-S at 672 px, `agnostic` preprocessing, 3 runs.** Not 448.

| Config | 1-shot image AUROC (MVTec, mean ± std over 3 runs) | 1-shot pixel AUROC | Source |
|---|---|---|---|
| S (448) | 96.5 ± 0.4 | 96.3 ± 0.1 | `tables/full_results_MVTec.tex` (Tab. `results_MVTec`, paper:L400–405) |
| **S (672)** | **96.6 ± 0.4** | **96.8 ± 0.1** | same |

The per-category table (`tables/MVTec_per_object.tex`, Tab. `MVTec-detailed-agnostic`,
caption paper:L642–647) is for **S (672)** and reports **image AUROC and PRO** per category for
k ∈ {1,2,4,8,16}. **The paper does not publish per-category pixel AUROC** — only the macro mean
(row above). An older per-category table is present but commented out in the source and is not used.

**Decision**: the benchmark harness runs at **672** so that the per-category gate (Phase 1) and
the mean gate (Phase 2) compare like with like. The library default (`InspectorConfig.resolution`)
stays **448** — the reference code default (`run_anomalydino.py:35`) and the paper's runtime
default (paper:L1022, "default setting is … resolution 448"); ~2× cheaper (32×32 vs 48×48 patches).

## 2. Defaults

| Item | Reference default | Citation |
|---|---|---|
| Backbone | `torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')`, eval mode, fp32 (`half_precision=False`) | `src/backbones.py:13,95-96`; `run_anomalydino.py:30` |
| Input resolution | smaller edge → **448** (code default); paper evaluates 448 and 672 | `run_anomalydino.py:35`; paper:L360 |
| Resize | `transforms.Resize(size=res, BICUBIC, antialias=True)` applied to a **PIL** image (aspect ratio kept) | `src/backbones.py:102-106,110-115` |
| Crop | crop bottom/right so H, W are multiples of patch size 14 (no centre crop) | `src/backbones.py:116-121` |
| Normalisation | ImageNet mean (0.485,0.456,0.406), std (0.229,0.224,0.225) | `src/backbones.py:105` |
| Colour | `cv2.imread(IMREAD_COLOR)` + BGR→RGB, i.e. grayscale images (grid, screw, zipper) are replicated to 3 channels | `src/detection.py:87,151` |
| Token output | `model.get_intermediate_layers(x)[0]` → defaults `n=1, norm=True, return_class_token=False`: **last block, after final LayerNorm, patch tokens only** (CLS + registers dropped), 384-d for ViT-S. Identical to `forward_features(x)["x_norm_patchtokens"]` | `src/backbones.py:132`; dinov2 `vision_transformer.py:270,301-320` |
| Feature normalisation | raw tokens are used for PCA masking; tokens are **L2-normalised only for kNN** (`faiss.normalize_L2`) | `src/detection.py:100,122-123,157,174` |
| Grid | 448 → 32×32 = 1024 patches; 672 → 48×48 = 2304 patches (MVTec images are square) | paper:L252,L361 |
| Reference sampling | `sorted(os.listdir(train/good))[seed*k:(seed+1)*k]` — **contiguous blocks, not random**: seeds 0/1/2 = first/second/third k train images | `src/detection.py:76`; paper:L375 (footnote) |
| Seeds / runs | 3 (`--num_seeds 3`, seeds 0,1,2); mean ± std over the 3 runs | README "default setting"; paper:L374 |
| Shots | k ∈ {1,2,4,8,16} | README; Tab. `results_MVTec` |
| Preprocessing mode | `agnostic` (default) | `run_anomalydino.py:33`; paper:L349,L363 |
| **Rotation augmentation** | `agnostic` ⇒ **all 15 MVTec categories, every k**. Angles **[0,45,90,135,180,225,270,315]** (8 copies incl. the original). Applied to the **original-resolution uint8 RGB** reference image *before* resize: `cv2.getRotationMatrix2D(center=(w/2,h/2), angle, 1.0)`, `cv2.warpAffine(..., (w,h), INTER_LINEAR, BORDER_DEFAULT=REFLECT_101)` (same output size; corners filled by reflection). Test images are never rotated. (`informed` would rotate only hazelnut + screw.) | `src/utils.py:7-22,119-137`; `src/detection.py:90-93`; paper:L349, Tab. `Preprocessing_Default` (paper:L870-919) |
| **Masking — which categories** | `capsule, hazelnut, pill, screw, toothbrush` = masked; textures (carpet, grid, leather, tile, wood) never masked "(T)"; bottle, cable, metal_nut, transistor, zipper failed the masking test "(MT)" | `src/utils.py:101-116`; Tab. `Preprocessing_Default` |
| **Masking — on what** | **test images only**; reference images are *not* masked (`mask_ref_images=False` ⇒ `masking_type=False` ⇒ all-ones mask) | `run_anomalydino.py:42`; `src/detection.py:100`; paper:L1007 ("we applied masking only to the test samples") |
| **Masking — procedure** | per image, on its own raw patch tokens: `PCA(n_components=1, svd_solver='randomized').fit_transform` → `mask = pc1 > 10`; centre check: take the centre crop `[0.2·H:0.8·H, 0.2·W:0.8·W]` of the grid; if ≤ 35 % of it is foreground, flip: `mask = -pc1 > 10`; then `cv2.dilate` 3×3 ones, then `cv2.morphologyEx(MORPH_CLOSE)` 3×3 ones. **Quirk**: `first_pc` has shape `(N, 1)`, so the dilation/closing run on the *flattened N×1 column* — a 1-D 3-tap operation along raster order (horizontal neighbours, wrapping across row ends), not 2-D morphology on the grid. Found by our unit test against a verbatim copy of the reference function; we reproduce it by default (`mask_morphology="flat"`, `"2d"` optional) | `src/backbones.py:159-174` (shape from `fit_transform` at :162); paper:L321-329, L1004-1007 |
| Masking decision rule ("masking test") | decided **once per category from the first reference sample**, by visual inspection of the PCA mask (Fig. masking test). The code hard-codes the resulting table; there is no automatic test in the code | paper:L326-327, L870-877; `src/utils.py:98-100` |
| NN search | exact `faiss.IndexFlatL2` (GPU or CPU) over **all** reference patches (no coreset) | `src/detection.py:111-124` |
| Distance | L2-normalise refs and queries, squared-L2 / 2 = **cosine distance 1 − cos**; `k_neighbors = 1` | `src/detection.py:122-123,173-178`; `run_anomalydino.py:36-37`; paper:L278-281 |
| Background patches | distance set to **0** for masked-out test patches (`output_distances[mask2] = distances`, zeros elsewhere) | `src/detection.py:180-182` |
| **Image score** | **mean of the top 1 % patch distances**, computed over the **full grid including the zeroed background patches**; count = `int(n·0.01)` (floor; 1024 → 10, 2304 → 23); if that is 0 → max. Paper: "empirical tail value at risk at 99 %". *(The brief's belief — mean of top-1 % — is confirmed.)* | `src/post_eval.py:293-297`; `src/detection.py:188`; paper:L291-298 |
| **Anomaly map** | patch grid → `cv2.resize(..., (W_orig, H_orig), INTER_LINEAR)` (**to the original image size**, e.g. 1024²) → `scipy.ndimage.gaussian_filter(sigma=4)` (scipy defaults: `truncate=4.0`, `mode='reflect'`) | `src/utils.py:25-30`; `src/detection.py:192-194`; paper:L304-306 ("bilinear upsampling and Gaussian smoothing (σ = 4.0)", following PatchCore) |
| Image AUROC | `roc_auc_score` per category; label = any GT pixel > 0 (equivalently: test folder ≠ `good`); MVTec macro mean over 15 categories | `src/post_eval.py:354-396,498-506` |
| **Pixel metrics resolution** | anomaly maps saved at **original image resolution** (`dists2map(d, image_test.shape)`), GT masks read as-is; pixel AUROC = `roc_auc_score` over **all pixels of all test images of a category pooled** (good images contribute all-zero GT) | `src/detection.py:192-194`; `src/post_eval.py:304-336` |
| AUPRO | MVTec official PRO curve, integrated to FPR 0.3 and normalised by 0.3 | `src/post_eval.py:169-290,321-328`; `run_anomalydino.py:172` |
| Latency protocol | batch 1, 25 warm-up iters, `torch.cuda.synchronize()`; timer covers load + preprocess + features + mask + kNN + score (not map upsampling). Paper: S/448 0.060 ± 0.012 s, S/672 0.086 ± 0.014 s per image on an NVIDIA A40 | `run_anomalydino.py:48,119-123`; `src/detection.py:147-186`; Tab. `runtime` (paper:L1018-1090) |

MVTec image sizes (relevant for pixel metrics and σ = 4 px at full resolution): bottle 900,
metal_nut 700, pill 800, tile 840, capsule 1000, others 1024 (all square).

## 3. Where this spec and the reference disagree — resolution

| Spec says | Reference does | We do |
|---|---|---|
| "fit: … keep only foreground patches when masking applies" | masks **test** images only (refs unmasked) | follow reference: `InspectorConfig.mask_references=False` by default (option to enable) |
| "seeded k-shot sampler" | deterministic contiguous slices of the sorted train list | `sample_references(..., strategy="contiguous")` default reproduces the paper; `strategy="random"` (seeded) available |
| "masking: auto" | per-category hard-coded table from a visual masking test | `auto` = paper table for MVTec category names; for unknown categories an *a-priori* automatic masking test on the first reference (documented in `inspector/masking.py`; never tuned on test labels; not used by the benchmark) |
| backbone returns L2-normalised features | PCA masking needs raw tokens; L2 only for kNN | backbone returns raw tokens; `PatchBank` L2-normalises refs + queries |
| "chunked GPU matmul (fp16 on CUDA)" | faiss fp32 exact L2 | exact max-cos via chunked matmul; fp16 on CUDA, fp32 on MPS/CPU (faiss backend optional) |

## 4. Gate targets taken from `docs/paper_numbers.csv`

- Phase 1 (k = 1, seeds 0,1,2, S/672): bottle 99.7 ± 0.2, screw 93.2 ± 0.3 (image AUROC).
- Phase 2: 1-shot macro image AUROC 96.6 ± 0.4 (S/672).
- The paper's std across 3 runs: ddof not stated (unverified); we report population std (ddof = 0).

## 5. MVTec AD download route (verified 2026-10-12, HEAD requests)

| Route | URL | Status |
|---|---|---|
| **Primary** — anomalib's URL (`open-edge-platform/anomalib` @ `dc087d5`, `src/anomalib/data/datamodules/image/mvtecad.py` L61-66; changed in PR #3673, 2026-07-17, because the old share `38536/…` now 404s) | `https://www.mydrive.ch/shares/150996/b52ecdcbf521176e9db9c731f2304b27/download/420938113-1629960298/mvtec_anomaly_detection.tar.xz` | 200, 5 264 982 680 bytes, `Accept-Ranges: bytes`; SHA-256 `cf4313b13603bec67abb49ca959488f7eedce2a9f7795ec54446c649ac98cd3d` |
| Fallback 1 — Hugging Face mirror of the identical tarball (LFS oid = same SHA-256) | `https://huggingface.co/datasets/micguida1/mvtech_anomaly_detection/resolve/eb692b36147a3b3f12c561b879a062104c1f20b8/mvtec_anomaly_detection.tar.xz` | 302 → 200, same size |
| Fallback 2 — Kaggle `ipythonx/mvtec-ad` (zip, original folder layout, 6644 files) | `kaggle datasets download ipythonx/mvtec-ad` | zip, different hash; verify by file counts |
| Official page (form-gated, licence point of record) | <https://www.mvtec.com/research-teaching/datasets/mvtec-ad> | CC BY-NC-SA 4.0; per-category tarballs no longer exist |

Licence: CC BY-NC-SA 4.0 (non-commercial). Cite Bergmann et al., CVPR 2019 and IJCV 2021.
The dataset is never committed (`data/` is gitignored).
