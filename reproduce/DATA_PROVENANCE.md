# How `lma_data/` was produced

The reproduction notebook needs only the precomputed descriptors in `lma_data/`; this
note records how they were generated from the full pipeline, so the shipped data is
auditable. None of these steps are required to reproduce the paper — they document the
path from raw video to the 122-d vectors the notebook consumes.

## `features.npz` — the 61-feature descriptors

For each motion fragment:

1. **Pose & reconstruction.** Raw video → YOLO11x-pose person/segment filter → WHAM (our
   pinned fork; see [`../docs/wham_fork_vs_official_audit_2026-06-09.md`](../docs/wham_fork_vs_official_audit_2026-06-09.md)),
   producing per-frame SMPL mesh vertices in the camera frame. This is the
   `core/wham_inference.py` + `scripts/run_wham_batch.py` stage.
2. **LMA descriptors.** Each WHAM fragment is turned into a `(T, 61)` per-frame Laban
   descriptor by `scripts/extract_lma_features.py` — the canonical SMPL-24 skeleton is
   regressed from the mesh vertices in the camera frame (`joints = J_regressor @ verts`,
   no `trans_world`), the per-frame body volume is the convex hull of the WHAM mesh, and the
   61 descriptors (a literal reading of Turab et al.'s LMA set — inter-joint distances and
   angles, dispersions, the four Effort factors, Initiation, trajectory, body volume) are
   computed over a causal 55-frame window; per-joint dynamics are lag-1 finite differences
   (`np.gradient`).
3. **Summary.** Each fragment's `(T, 61)` matrix is collapsed to a `122`-d vector by
   per-feature mean and standard deviation (`scripts/extract_lma_features.py:summary_vector`).
   Run on the same WHAM fragments, this driver reproduces every **per-feature value** in
   `features.npz` bit-for-bit — each column matches by name to float32-exact (`max|Δ| = 0`,
   short clips included). The shipped file's column ordering, its opaque `sha1` keys, and the
   anonymized `source_video` field are applied by the release-packaging step on top of those
   values (so a *naive* whole-array `np.array_equal` is not the test — the name-aligned values
   are what reproduce exactly).

`features.npz` holds, for the 7,251 renderable fragments: `keys` (a stable per-fragment
id), `X` (`N×122`), `length` (frame count), `source_video` (an **anonymized** opaque
grouping id for leak-free CV — a salted hash of the source; the raw source identifiers are
deliberately not published), and `feature_names`. Both `keys` and `source_video` are opaque
by design: the released features cannot be re-linked to a specific source video or person.

## `pools.json` — task pools

For each task (`binary`, `3way_drop`, `4way`), the list of candidate fragment keys per
class, *before* balancing. The notebook re-draws the paper's class-balanced sample from
these at seed 42. Tier membership of every fragment is recoverable from the `4way` pool
(its four class lists are tiers 0–3), which is also how the notebook reproduces the
per-tier dataset counts in the paper's *Data* paragraph.

## `videomae_preds.csv` — the learned-baseline predictions

Per-fragment predictions (`task, key, true, pred`) from VideoMAE fine-tuned on
appearance-free plain-mesh re-renders of the **same** fragments under the **same**
source-video-grouped folds. The video model is **not** retrained in the notebook; its
predictions are shipped so the comparison table can be rebuilt. The rendering + training
harness is not part of this repository (it is heavy and orthogonal to the LMA pipeline);
only its outputs are shipped. The full protocol — model, appearance-free renders, and the
seed-42 source-video-grouped folds (identical to the LMA side) — is documented in
[`../docs/videomae_baseline.md`](../docs/videomae_baseline.md).
