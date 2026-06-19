# WHAM fork vs. official upstream — audit (2026-06-09)

The 3D body reconstruction stage runs a **modified fork** of WHAM, not stock WHAM.
This note enumerates every deviation from upstream and states, for each, whether it
affects the numbers the paper reports. It is the reference for the
`git clone … zaiisao/WHAM @ baca651` instruction in the top-level README.

## Provenance

| | |
|---|---|
| Fork | [`zaiisao/WHAM`](https://github.com/zaiisao/WHAM), branch `main` |
| Pinned commit | `baca65177979c165c154992005124233843a048a` (`baca651`) |
| Official upstream | [`yohanshin/WHAM`](https://github.com/yohanshin/WHAM) `main` @ `2b54f77` |
| Relationship | the fork's merge-base with upstream **is** upstream's HEAD, so the fork is a clean **5-commit superset** of official WHAM — no upstream history was rewritten |

The five commits on top of upstream:

```
34b8f0a  Add .gitignore
94b812a  Update modules to 2026 SOTA          (YOLO26x, ViTPose++, OKS)
d546323  Improve tracking robustness / SLAM wiring / fragment viz
0cd655d  Make WHAM installable as a package
baca651  Improve crash-dump logging in DPVO
```

`git diff 2b54f77..baca651 --stat` touches **only** these paths:

```
.gitignore
demo.py
lib/models/preproc/detector.py
lib/models/preproc/slam.py
setup.py
third-party/DPVO       (submodule pointer)
third-party/ViTPose    (submodule pointer)
```

Everything else is byte-identical to upstream. In particular:

- **All model / config code is unchanged**: `configs/yamls/*.yaml`, `configs/config.py`,
  `configs/constants.py`, and every network / SMPL definition under `lib/models/`
  (except the two `preproc` files above) are byte-identical to `2b54f77`.
- **No weights are vendored**: zero `.pth/.pkl/.pt/.npz/.ckpt` binaries are tracked in
  the fork. The SMPL body model and WHAM/ViTPose/DPVO checkpoints are downloaded per
  WHAM's own README, exactly as upstream. So the network and SMPL **weights cannot
  differ** from official on the published surface.
- **Submodule pins** that travel with the fork: `third-party/DPVO` `b8ff810`
  (upstream `5833835`), `third-party/ViTPose` `831848a` (upstream `d521645`). Clone with
  `--recurse-submodules` (or run `git submodule update --init --recursive` after
  `checkout baca651`) so these come along.

## The track-length gate (what the paper actually used)

`lib/models/preproc/detector.py` at `baca651` commits:

```python
MINIMUM_FRMAES   = 30      # (upstream value; unchanged)
MIN_TRACK_SECONDS = 2.5    # (upstream value; unchanged)
```

The effective per-track gate (`detector.py:143`) is
`min_track_frames = max(MINIMUM_FRMAES, int(fps * MIN_TRACK_SECONDS)) = max(30, fps·2.5)`.
**The paper's data was produced with these committed values.**

A relaxed `15 / 1.0` gate exists **only** as (a) an *uncommitted* working-tree edit on
the local box (the `# was 30` / `# was 2.5` comments) and (b) the parent repo's old
`patches/wham_detector_relax_min_track.patch`. Neither is part of the fork at `baca651`,
and **neither produced the paper numbers**. That patch has been removed from this repo to
avoid implying otherwise; the parent pipeline applies its own, separate length floor in
`core/wham_inference.py` (`len(frames) < 15`), independent of WHAM's internal gate.

## Deviations from upstream

| # | Area | Upstream | Fork (`baca651`) | Files | Affects paper numbers? |
|---|---|---|---|---|---|
| 1 | Person detector | YOLOv8x | **YOLO26x** | `detector.py:35-37` | **Yes** — different boxes feed ViTPose+WHAM, so reconstructed joints (and all 61 LMA features) differ |
| 2 | 2D pose | ViTPose-Huge (COCO) | **ViTPose++ base** (multi-dataset, UDP) | `detector.py:30-33`; ViTPose pin `831848a` | **Yes** — the 2D keypoints are WHAM's primary observation |
| 3 | Cross-frame association | `use_oks=False` | **`use_oks=True`** | `detector.py:103` | **Yes** — changes track-ID assignment / which frames are stitched into one subject track |
| 4 | Detector thresholds | `BBOX_CONF=0.5`, `TRACKING_THR=0.1` | **`0.4`, `0.25`** | `detector.py:25-26` | **Yes** — gates which detections survive |
| 5 | SLAM (DPVO) | mixed precision | **full fp32** (`MIXED_PRECISION=False`) + 4-arg `slam()` plumbing | `slam.py:44,58,63`; DPVO pin `b8ff810` | World-frame only — the paper's extended 61-feature path uses **camera-frame** joints (`J = R·V`), so the effect on published numbers is negligible; it matters for any world-frame quantity |
| 6 | `demo.py` | plain demo | fragment stitcher + `np.float` shim + subprocess visualizer | `demo.py` | **No** — the paper path does not run `demo.py`; it drives the fork through `core/wham_inference.py` + `scripts/run_wham_batch.py` |
| 7 | Empty-detection guard | none (crashes on 0 detections) | early-return guard | `detector.py:127-130` | **No** — robustness only; no change for videos with a tracked subject |
| 8 | Packaging | absent | `setup.py`, `.gitignore` | — | **No** — build only |
| 9 | Dead code | — | `MAX_TRACK_AGE=60`, `self.track_memory={}` declared but never consumed | `detector.py:29,50` | **No** — inert |

## Bottom line

The fork is a strict superset of official WHAM `2b54f77` whose *numeric* deviations are
the swapped detector (YOLO26x), 2D pose model (ViTPose++), OKS association, and two
detection thresholds — all in `lib/models/preproc/detector.py`. Model weights and configs
are upstream-identical, and the SLAM/fp32 change is world-frame only and immaterial to the
camera-frame 61-feature descriptors the paper reports. To reproduce the paper's WHAM stage,
clone the fork at `baca651` with its submodule pins — **the official upstream plus the old
relax patch reproduces none of these and is not the configuration the paper used.**
