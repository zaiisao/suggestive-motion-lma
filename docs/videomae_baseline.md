# The VideoMAE baseline (appearance-free video comparison)

Table 1 compares the hand-crafted LMA descriptor against a **learned video model**
(VideoMAE) fine-tuned on the *same motion re-rendered without appearance*. This note
documents exactly how that baseline was produced, so the VideoMAE half of Table 1 is
auditable even though — unlike the LMA half — it is not reproduced from scratch in this
repo.

## What ships here vs. what does not

- **Ships:** `reproduce/lma_data/videomae_preds.csv` — the per-fragment out-of-fold test
  predictions (`task, key, true, pred`; 10,203 rows = binary 2×1,507 + 3way_drop 3×1,027
  + 4way 4×1,027). `reproduce/reproduce_paper.ipynb` rebuilds the VideoMAE rows of Table 1
  from this file; the model is **not** retrained there.
- **Does not ship:** the training / rendering / split harness. It is research-grade,
  GPU-heavy, and needs the mesh renders (not shipped). It lives in the authors' development
  repository: `train_videomae/train_videomae_wham.py`, `scripts/archive/videomae/` (renders +
  splits + sweep), and `scripts/aligned_lma_vs_videomae.py` (the apples-to-apples LMA driver).

## Why it is a fair comparison (apples-to-apples)

The VideoMAE clips and the LMA fragments are the **same items under the same protocol**:

- **Same fragment identity.** Each render is named `sha1(lma_features_path)[:16]` — the
  *same key* used in `features.npz`. The notebook joins VideoMAE predictions to LMA
  fragments by this key.
- **Same sampling and folds.** The split builder mirrors the LMA pipeline exactly:
  `RandomState(42)` class-balancing (1,027/class for 4way & 3way_drop, 1,507/class for
  binary), then `StratifiedGroupKFold(5, shuffle, random_state=42)` grouped by source video
  (leak-free) — segments of one source video collapse to a single group, so no video
  straddles train/test. (An ungrouped `StratifiedKFold(5, seed 42)` mode also exists, used
  for the leaky/ablation rows.)
- **Appearance is removed, not the motion.** See below — the model sees only the moving
  mesh, so the comparison isolates *motion* from pixels.

## The inputs: appearance-free mesh renders

`scripts/archive/videomae/render_mesh_videos.py` drives WHAM's native global renderer
(checkerboard ground, subject-tracking camera) directly from the cached SMPL world vertices
in each `wham_fragment_id{N}.npz` — **no source video and no neural re-inference**, so the
renders are derivable purely from the shipped-pipeline WHAM fragments. The subject is a
**single uniform light-gray mesh** (`colors[..., :3] *= 0.9`, no texture, no skin, no
background), feet floored to `y=0`. Output is 256×256 (the model ingests 224). The only
visual signal is the motion and shape of a neutral body — this is what makes the baseline
**appearance-invariant**.

## The model and training

`train_videomae/train_videomae_wham.py` (runs in the `wham` env; a port of the original
`train_videomae.py` that swaps `pytorchvideo`→`torchvision.io` and `decord`→PyAV to avoid a
decord/CUDA segfault):

- **Model:** `MCG-NJU/videomae-base-finetuned-kinetics` (ViT-Base, Kinetics-400), with the
  1000-way K400 head replaced by an N-way head (`ignore_mismatched_sizes=True`).
- **Decode/augment:** uniform temporal subsample to `config.num_frames` (16), `/255`,
  per-channel normalize with the image-processor stats, resize to 224, random horizontal
  flip on train only.
- **Trainer:** HuggingFace `Trainer`, 30 epochs, batch 8, lr 5e-5, `warmup_ratio=0.1`,
  bf16, `seed=42`, `load_best_model_at_end` on validation accuracy.
- **Per fold:** trained on one `{task}_testfold{k}` split (`train/`/`val/`/`test/` of
  class-named subdirs of symlinked renders); `trainer.predict(test)` yields each test
  clip's prediction.

## From per-fold predictions to `videomae_preds.csv`

Across the 5 folds each fragment lands in the test set exactly once, so collecting every
fold's test predictions gives one out-of-fold `(key, true, pred)` per fragment. Stacking the
three tasks (`binary`, `3way_drop`, `4way`) yields `videomae_preds.csv`. The notebook then
computes balanced accuracy (and the length-matched, de-confounded variant) from these
predictions — the numbers in Table 1.

## Reproducing it (not turn-key)

Re-running the baseline needs a GPU and the mesh renders. The path is: render with
`render_mesh_videos.py` from the WHAM fragments → build splits with
`build_videomae_split_canonical.py --config {binary,3way_drop,4way} --group-by vid` → train
per fold with `train_videomae_wham.py` → collect the per-fold test predictions into the CSV.
The harness lives in the authors' development repository.

## Honest scope

The LMA half of Table 1 is reproduced end-to-end in this repo (the shipped `features.npz`'s
per-feature values are regenerated bit-for-bit by `scripts/extract_lma_features.py`, name-aligned).
The VideoMAE half is shipped
as predictions only: this document makes its **protocol** auditable — same fragments, same
seed-42 grouped folds, same keys, appearance-free inputs — but independently **re-running**
it requires the harness above plus a GPU and the renders.
