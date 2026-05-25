# Suggestive-Motion-LMA

Reference implementation for the SIGGRAPH Posters '26 paper **"Appearance-Invariant
Detection of Suggestive Motion via Laban Movement Descriptors on SMPL Skeletons"**
(Ahn, Kong, and Jung 2026).

The pipeline classifies video motion fragments across a four-tier suggestiveness
taxonomy — *everyday → artistic → suggestive → explicit* — using only Laban
Movement Analysis (LMA) descriptors computed from SMPL skeletons. No pixel-level
information ever reaches the classifier.

Headline results (5-fold CV, balanced 1,075 fragments per class, 20,514 total):

| Setting             | Accuracy | Chance |
|---------------------|---------:|-------:|
| 4-way (T0/T1/T2/T3) | **57.3%** | 25.0% |
| 3-way (drop T1)     | **72.1%** | 33.3% |
| Binary SFW/NSFW     | **78.7%** | 50.0% |

The 4-way confusion matrix (`results/fig_cm_4way.pdf`) is Figure 2 of the paper.

---

## Pipeline

```
raw video ──► YOLO11x-pose filter (≥10 kpts @ conf≥0.5, segments ≥3s)
           ──► ffmpeg clip extraction
           ──► WHAM (SMPL skeleton in world space)
           ──► 55 LMA descriptors per frame
           ──► mean + std → 110-dim per-fragment vector
           ──► logistic regression (or random forest)
```

| Component | Source |
|-----------|--------|
| Person filter | YOLO11x-pose (ultralytics) |
| 3D body reconstruction | [WHAM](https://github.com/yohanshin/WHAM), commit `baca6517` |
| LMA descriptors | [dance-style-recognition](https://github.com/zaiisao/dance-style-recognition), commit `baef1ee9` |
| Classifier | scikit-learn `LogisticRegression`, `RandomForestClassifier` |

---

## Installation

### 1. Conda environment

```bash
conda env create -f environment.yml
conda activate wham
```

This is the same environment used to produce the paper results. Pinned to
Python 3.9, PyTorch 2.1 + CUDA 11.8 wheels. Approx. 15 GB.

### 2. External dependencies

```bash
mkdir -p external
git clone https://github.com/yohanshin/WHAM.git external/WHAM
( cd external/WHAM && git checkout baca65177979c165c154992005124233843a048a )

git clone https://github.com/zaiisao/dance-style-recognition.git external/dance-style-recognition
( cd external/dance-style-recognition && git checkout baef1ee983e9d86449cad71f129474f2459955a5 )

# Apply the single-person-detector relaxation used in the paper
( cd external/WHAM && git apply ../../patches/wham_detector_relax_min_track.patch )
```

Follow WHAM's own README to download its model checkpoints (`checkpoints/`)
and the SMPL body model into `external/WHAM/dataset/body_models/`. WHAM
itself depends on ViTPose and (optionally) DPVO for SLAM; install them per
WHAM's instructions.

### 3. YOLO11x-pose

```bash
# ~110 MB
wget -O yolo11x-pose.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-pose.pt
```

Or override the path with `LMA_YOLO_MODEL_PATH=/path/to/yolo11x-pose.pt`.

### 4. The detector patch

The paper's batch pipeline relaxes WHAM's track-length thresholds so short
TikTok-style clips don't get dropped:

```diff
- MINIMUM_FRMAES = 30          # WHAM default
- MIN_TRACK_SECONDS = 2.5
+ MINIMUM_FRMAES = 15
+ MIN_TRACK_SECONDS = 1.0
```

`patches/wham_detector_relax_min_track.patch` is the diff to apply against
upstream WHAM `lib/models/preproc/detector.py`.

---

## Reproducing the paper

The pipeline has two stages, with a hard separation between feature extraction
(slow, GPU) and classification (fast, CPU). You only need to re-run stage 1
if you change the feature extractor.

### Stage 1: extract LMA features from video

**Tier 0 / 1 (Kinetics-700).** Point at a local Kinetics-700 train split and
list the classes:

```bash
export LMA_KINETICS_ROOT=/path/to/kinetics-700/train
export LMA_TIER1_OUTPUT_DIR=output/tier1_features

# Tier 1 — artistic motion (40 vids per class, one GPU shard)
CUDA_VISIBLE_DEVICES=0 python scripts/batch_tier1_kinetics.py \
    --gpu-id 0 \
    --classes breakdancing krumping capoeira gymnastics_tumbling ballet \
              salsa_dancing tap_dancing tango_dancing
```

For Tier 0 use the same script with everyday classes (walking, eating,
sitting, …) and change `--output` accordingly. The 19 / 16 specific Kinetics
classes used in the paper are listed in Section 2 of the poster.

**Tier 2 / 3 (web-scraped + NPDI).** Use the YOLO-filtered batch driver. It
expects a flat directory of video files:

```bash
export LMA_T3_DIR=/path/to/tier3_videos
export LMA_OUTPUT_DIR=output/tier3_processing

CUDA_VISIBLE_DEVICES=0 python scripts/batch_filtered_wham.py \
    --start 0 --end 1000 --gpu-id 0
```

Both batches write per-fragment files into
`output/<...>/<video_id>/lma_features_id{N}.npy`, each shaped `(T, 55)`.

### Stage 2: classify

```bash
# Defaults look in ./data/tier{0,1,2,3}/ — override with --tierN-dirs or env vars
python scripts/analyze_lma_tiers.py \
    --tier0-dirs output/tier0_features \
    --tier1-dirs output/tier1_features \
    --tier2-dirs output/tier2_features \
    --tier3-dirs output/tier3_features \
    --max-per-tier 1075 \
    --out-dir output/lma_analysis

# 3-way (paper's 72.1%)
python scripts/analyze_lma_tiers.py --drop-tier1 ...

# Binary SFW/NSFW (paper's 78.7%)
python scripts/analyze_lma_tiers.py --binary ...
```

Outputs in `--out-dir`:

- `summary.json` — accuracy + F1 per classifier
- `cm_logreg.png`, `cm_randomforest.png` — confusion matrices
- `kruskal_features.csv` — per-feature Kruskal–Wallis H ranking
- `rf_feature_importance.csv` — RandomForest importances
- `pca_2d.png`, `tsne_2d.png` — projections
- `per_tier_means.csv` — feature means per class

### Stage 2b: bit-exact paper reproduction (audit mode)

The three accuracies reported in the paper (4-way 57.3%, 3-way 72.1%, binary
78.7%) were each produced from a different snapshot of the still-growing
feature directories, on three different dates in April 2026. To make those
specific numbers auditable and re-runnable, this repo ships per-cutoff
manifests of every LMA feature file used:

```
data/manifest_paper_4way_2026-04-15_01-23.csv     # 20,534 files
data/manifest_paper_3way_2026-04-14_19-16.csv     # 17,880 files
data/manifest_paper_binary_2026-04-14_21-36.csv   # 18,358 files
```

Each row is `tier,path,mtime_utc,size_bytes,sha256`. To re-run the classifier
against the exact paper file set:

```bash
python scripts/analyze_lma_tiers.py \
    --manifest data/manifest_paper_4way_2026-04-15_01-23.csv \
    --max-per-tier 1075 \
    --out-dir output/audit_4way
```

This reproduces the published headline numbers **bit-exactly** (verified
2026-05-25):

| Setting | Paper | Reproduced |
|---|---|---|
| 4-way LogReg | 0.5728 | 0.5728 ✓ |
| 4-way RandomForest | 0.5742 | 0.5742 ✓ |
| 3-way LogReg | 0.7206 | 0.7206 ✓ |
| 3-way RandomForest | 0.7020 | 0.7020 ✓ |
| Binary LogReg | 0.7869 | 0.7869 ✓ |
| Binary RandomForest | 0.7824 | 0.7824 ✓ |

`scripts/verify_paper_reproduction.sh` runs all three configurations in
sequence. `scripts/check_manifest_hashes.py <manifest.csv>` validates that
the feature files referenced by a manifest still hash-match (use this before
publishing a result to catch silent file mutation).

`scripts/build_manifest.py` is the tool used to generate the manifests in
the first place. To produce one for a new analysis snapshot:

```bash
python scripts/build_manifest.py \
    --tier0-dirs <...> --tier1-dirs <...> --tier2-dirs <...> --tier3-dirs <...> \
    --cutoff '2026-04-15 01:23:30' \
    --out data/manifest_<your-run>.csv
```

The cutoff is needed because the feature directories may continue to grow
after a paper run; the manifest records the exact subset that was visible
to the classifier at run time.

### Stage 3: paper figure

`scripts/fig_confusion_matrix.py` regenerates Figure 2 from the published
counts (hard-coded — re-run after Stage 2 with your numbers if you want the
matrix from your own run):

```bash
python scripts/fig_confusion_matrix.py
# -> scripts/fig_cm_4way.pdf, scripts/fig_cm_4way.png
```

---

## Expected directory layout

```
suggestive-motion-lma/
├── core/
│   └── wham_inference.py         # per-video WHAM + LMA worker
├── scripts/
│   ├── batch_filtered_wham.py    # YOLO-filtered batch driver (tier 2/3)
│   ├── batch_tier1_kinetics.py   # Kinetics-700 batch driver (tier 0/1)
│   ├── analyze_lma_tiers.py      # classifiers + ablations
│   └── fig_confusion_matrix.py   # paper Figure 2
├── patches/
│   └── wham_detector_relax_min_track.patch
├── results/                      # paper figures
├── external/                     # WHAM, dance-style-recognition (cloned here)
├── environment.yml
└── README.md
```

---

## Notes

- The 55 LMA descriptors are computed by
  `process_lma_features.compute_lma_descriptor` from `dance-style-recognition`.
  `core/wham_inference.py` imports it directly from
  `external/dance-style-recognition/src/`.
- WHAM is run with the SLAM backend (DPVO) when available so that fragments
  carry world-space coordinates. If DPVO is not installed WHAM falls back to
  camera-space, which will degrade the Trajectory/Initiation features.
- `wham_inference.py` forces `multiprocessing` to `spawn` and pins
  `OMP_NUM_THREADS=1` to avoid a fork-time CUDA + DataLoader deadlock that
  otherwise leaves zombie WHAM workers running for hours.

---

## Citation

```bibtex
@inproceedings{ahn2026laban,
  title={Appearance-Invariant Detection of Suggestive Motion via Laban Movement
         Descriptors on SMPL Skeletons},
  author={Ahn, Jaehoon and Kong, Jeonghan and Jung, Moon-Ryul},
  booktitle={SIGGRAPH Posters '26},
  year={2026},
  doi={10.1145/3799825.3818709}
}
```
