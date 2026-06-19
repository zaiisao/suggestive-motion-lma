# Reproducing the paper's results

Everything the paper derives from the **LMA motion data** is reproduced by
`reproduce_paper.ipynb` — no video, no WHAM, no GPU. Only the precomputed
61-feature Laban descriptors are needed.

## Run it
```bash
# from this directory (lma_data/ sits next to the notebook)
jupyter notebook reproduce_paper.ipynb     # or: jupyter lab
```
The notebook needs only `numpy`, `pandas`, `scipy`, and `scikit-learn` (the pinned
versions in the repo's `environment.yml`). Each section prints the reproduced numbers
and states the expected paper value above it.

## What's shipped (`lma_data/`)
| file | contents |
|---|---|
| `features.npz` | 7,251 renderable fragments × 122-d LMA vector (mean+std of 61 descriptors), clip length, source-video id, feature names |
| `pools.json`   | renderable fragment pool per task (binary / three-way / four-way) so the balanced samples can be re-drawn at seed 42, and the per-tier dataset counts |
| `videomae_preds.csv` | precomputed per-fragment VideoMAE predictions (the video model is **not** retrained here; [protocol](../docs/videomae_baseline.md)) |

## What it reproduces
- Dataset size (fragments / source videos per tier — the paper's *Data* paragraph)
- Leak-free LMA accuracy (LogReg & RF; binary / three-way / four-way) and the small grouping effect
- The clip-length confound and the length-matched (de-confounded) accuracies
- The full LMA-vs-VideoMAE comparison table (both panels of Table 1)
- Effort-factor decomposition (Kruskal–Wallis H ranks)
- Per-tier Directness and the four-way confusion matrix

The logistic-regression and VideoMAE rows reproduce Table 1 essentially exactly (VideoMAE
is a precomputed prediction file; LMA-LR lands within ≤0.2 pp). The random-forest rows carry
seed/BLAS nondeterminism across machines — up to ~1 pp, mostly on the length-matched panel
(e.g. length-matched RF can print ~0.71 against the paper's 0.704) — so they match the paper's
two-decimal values to within ~1 pp rather than exactly. Every qualitative claim holds on every
run: binary ≈0.78/0.81, three-way ≈0.71, four-way ≈0.58, the clip-length confound, and
LMA↔VideoMAE parity once length is controlled.

`DATA_PROVENANCE.md` documents how `lma_data/` was generated from the full pipeline; it is
provenance only — reproducing the paper needs just `lma_data/` + the notebook.
