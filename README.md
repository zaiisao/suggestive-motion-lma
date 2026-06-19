# Suggestive-Motion-LMA

Reference implementation for the SIGGRAPH Posters '26 paper **"Appearance-Invariant
Detection of Suggestive Motion via Laban Movement Descriptors"** (Ahn, Kong, and
Jung 2026) · [doi:10.1145/3799825.3818709](https://doi.org/10.1145/3799825.3818709).

The pipeline classifies video motion fragments across a four-tier suggestiveness
taxonomy — *everyday → artistic → suggestive → explicit* — using only Laban Movement
Analysis (LMA) descriptors computed from SMPL skeletons. **No pixel-level information
ever reaches the classifier**, so moderation can run on avatar poses alone.

## Results (Table 1)

Source-video-grouped 5-fold cross-validation, class-balanced (1,027 per class; binary
1,507). Accuracy is out-of-fold balanced accuracy.

| | Binary | Three-way | Four-way |
|---|---:|---:|---:|
| *chance* | 0.50 | 0.33 | 0.25 |
| **LMA (logistic regression)** | 0.778 | 0.706 | 0.576 |
| **LMA (random forest)** | 0.809 | 0.708 | 0.578 |
| VideoMAE (appearance-free video) | 0.724 | 0.664 | 0.557 |
| *+ length-matched* — LMA (LR) | 0.678 | 0.635 | 0.520 |
| *+ length-matched* — LMA (RF) | 0.704 | 0.638 | 0.530 |
| *+ length-matched* — VideoMAE | 0.694 | 0.628 | 0.532 |

The hand-crafted LMA descriptor matches a video transformer fine-tuned on the same motion
re-rendered without appearance, once the clip-length confound is controlled — at no cost to
auditability. **[`reproduce/`](reproduce/) reproduces this entire table (and every other
number the paper draws from the motion data) from precomputed features, with no GPU.**

---

## Reproduce the paper (no video, no WHAM, no GPU)

```bash
conda env create -f environment.yml && conda activate wham
cd reproduce && jupyter notebook reproduce_paper.ipynb
```
The notebook needs only `numpy`, `pandas`, `scipy`, `scikit-learn` and the shipped
61-feature descriptors in [`reproduce/lma_data/`](reproduce/lma_data/). See
[reproduce/README.md](reproduce/README.md).

---

## Pipeline (video → features → classifier)

```
raw video ──► YOLO11x-pose filter (≥10 kpts @ conf≥0.5, mean conf≥0.5, segments ≥3s)
           ──► WHAM (SMPL mesh, camera frame)
           ──► SMPL-24 skeleton = J_regressor @ verts   (camera frame, no trans_world)
           ──► 61 LMA descriptors per frame  (Turab et al.'s LMA set, read literally → 61)
           ──► mean + std → 122-d per-fragment vector
           ──► logistic regression / random forest, source-video-grouped 5-fold CV
```

| Component | Source |
|---|---|
| Person filter | YOLO11x-pose (ultralytics) |
| 3D body reconstruction | [WHAM](https://github.com/yohanshin/WHAM), run from our [pinned fork](https://github.com/zaiisao/WHAM) @ `baca651` (YOLO26x detector, ViTPose++ 2D, OKS tracking, fp32 SLAM) |
| LMA descriptors | [dance-style-recognition](https://github.com/zaiisao/dance-style-recognition) (submodule) |
| Classifier | scikit-learn `LogisticRegression`, `RandomForestClassifier` |

The 61-feature schema and column order are documented in [docs/lma_features.md](docs/lma_features.md).

### Code map

| Path | Stage |
|---|---|
| [`core/wham_inference.py`](core/wham_inference.py) | WHAM worker: per-video SMPL reconstruction |
| [`scripts/run_wham_batch.py`](scripts/run_wham_batch.py) | YOLO-filtered batch driver: video → WHAM fragment `.npz` |
| [`scripts/extract_lma_features.py`](scripts/extract_lma_features.py) | WHAM fragment → `(T, 61)` LMA descriptor → 122-d vector |
| [`reproduce/`](reproduce/) | turn-key reproduction: notebook + 61-feature data |
| [`docs/wham_fork_vs_official_audit_2026-06-09.md`](docs/wham_fork_vs_official_audit_2026-06-09.md) | exactly how our WHAM fork deviates from upstream, and which deltas affect the numbers |
| [`docs/videomae_baseline.md`](docs/videomae_baseline.md) | how the appearance-free VideoMAE baseline (Table 1) was rendered, trained, and evaluated |

> **On the LMA submodule's defaults.** [dance-style-recognition](https://github.com/zaiisao/dance-style-recognition)
> is a general LMA tool: from raw video it estimates 3D pose with NLF and the floor plane with
> MoGe. **This paper uses neither.** Our pipeline overrides both — feeding WHAM-reconstructed
> SMPL-24 joints (camera frame) and a flat floor into the *same* 61-feature descriptor. The
> submodule exposes pose and floor as pluggable estimators, so the WHAM path installs no NLF or MoGe.

---

## Installation (full pipeline)

```bash
conda env create -f environment.yml && conda activate wham   # Python 3.9, PyTorch 2.1 / CUDA 11.8

# external dependencies (not vendored)
git submodule update --init                                  # dance-style-recognition (LMA extractor)

# WHAM: clone our PINNED FORK (not official upstream) — it carries the YOLO26x detector,
# ViTPose++ 2D, OKS association and fp32 SLAM the paper used. Official WHAM does not
# reproduce the paper's reconstruction; see docs/wham_fork_vs_official_audit_2026-06-09.md.
git clone https://github.com/zaiisao/WHAM.git external/WHAM
git -C external/WHAM checkout baca651
git -C external/WHAM submodule update --init --recursive     # DPVO + ViTPose at the pinned commits

wget -O yolo11x-pose.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-pose.pt
```
Follow WHAM's README to download its checkpoints and the SMPL body model into
`external/WHAM/dataset/body_models/`. Override the YOLO weights path with
`LMA_YOLO_MODEL_PATH` and the SMPL model path with `LMA_SMPL_MODEL_PATH` if needed.

### Run the pipeline

```bash
# 1. video -> WHAM fragment .npz  (GPU)
LMA_T3_DIR=/path/to/videos LMA_OUTPUT_DIR=output/wham \
    python scripts/run_wham_batch.py --start 0 --end 1000 --gpu-id 0

# 2. WHAM fragments -> 61-feature LMA descriptors  (CPU)
python scripts/extract_lma_features.py --wham-dir output/wham --out output/lma
```
Classification is then exactly the procedure in `reproduce/reproduce_paper.ipynb`
(balanced sampling at seed 42, source-video-grouped 5-fold CV).

---

## Dataset

Four tiers of motion fragments (7,251 renderable fragments, 1,273 source videos):

| Tier | Description | Source | Fragments | Videos |
|---|---|---|---:|---:|
| T0 | everyday (walking, sitting, dining) | Kinetics-700 | 1,027 | 515 |
| T1 | artistic (breakdance, ballet, capoeira, gymnastics) | Kinetics-700 | 1,889 | 412 |
| T2 | suggestive (twerk, perreo, sensual/heels/belly dance) | YouTube + TikTok | 1,715 | 108 |
| T3 | explicit | NPDI academic corpus | 2,620 | 238 |

Tier-2 acquisition queries/channels and the dataset construction are in Appendix C of
the paper. We do not redistribute source videos; the shipped data is the derived
61-feature LMA descriptors only.

---

## Citation

```bibtex
@inproceedings{ahn2026laban,
  title     = {Appearance-Invariant Detection of Suggestive Motion via Laban Movement Descriptors},
  author    = {Ahn, Jaehoon and Kong, Jeonghan and Jung, Moon-Ryul},
  booktitle = {SIGGRAPH Posters '26},
  year      = {2026},
  doi       = {10.1145/3799825.3818709}
}
```

The paper and figures are in [`paper/`](paper/). Licensed under CC BY 4.0 (see [LICENSE](LICENSE)).
