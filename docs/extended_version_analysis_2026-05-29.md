# Post-acceptance analysis for the extended version (2026-05-29)

Reference notes for the extended / arXiv version of the SIGGRAPH '26 poster.
Everything here is **additive** — the poster pipeline and its frozen artifacts are
untouched and still reproduce bit-exact (V0 below).

**TL;DR**
1. We found a real, previously-unknown **joint-convention bug** in the LMA pipeline:
   WHAM emits a 31-joint **COCO+SPIN** layout, but the LMA extractor reads the first
   24 of it as if they were **SMPL-24**. So every named LMA feature was computed on
   the wrong body parts.
2. Fixing it (regress true SMPL-24 from the intact `verts`) gives a **real but modest**
   gain: **+2.7–3.6 pp on 4-way**, ~flat on 3-way/binary. It is a correction, not a
   transformation — the conclusions stand.
3. Two "obvious" suspects are **non-issues**: the floor/Y-axis convention is a proven
   no-op for the feature vector, and the world-space "tipping" is irrelevant because
   the features are rotation-invariant.
4. The remaining ceiling is **representational**: the 55-feature set is positional-only
   (no angular/articulation features, no periodicity, hips unused), and "suggestiveness"
   is substantially non-kinematic.
5. **Temporal length** interacts with accuracy in a tier-dependent, opposing way:
   SFW-side classes improve with clip length; NSFW-side classes (T3/NSFW) collapse with
   length. Strong implication: the suggestive signal is **bursty**, not sustained.

---

## 1. The joint-convention bug

**Contract assumed by the extractor.** `LMAExtractor.IDX`
(`external/dance-style-recognition/src/utils/lma_extractor.py:19-44`) is standard
**SMPL-24**: PELVIS=0, L_HIP=1, …, R_HAND=23.

**What was actually fed.** `core/wham_inference.py:~333` does
`joints = data['joints_world'][:, :24, :]` and passes it straight to
`compute_lma_descriptor` — no reordering anywhere. But `joints_world` comes from
`J_regressor_wham` (31×6890; `external/WHAM/lib/models/smpl.py:116`), which is
**17 COCO joints + 14 SPIN-extra**, NOT SMPL-24.

**Evidence it's wrong.**
- Bone lengths from `cached[:24]` under the SMPL-24 IDX are anatomically impossible:
  L_HIP→L_KNEE ≈ 15 cm, NECK→HEAD ≈ 62 cm, asymmetric L/R.
- Regressing SMPL-24 from the (intact) `verts` gives textbook bones on every tier:
  thighs 37–39 cm, shins 40–42 cm, shoulders 35 cm — symmetric.
- Direct same-frame matching: 10 of the 24 SMPL joints have an exact (0 mm) twin in the
  cached array, but at WHAM indices **17–28** (the COCO/extra block), not at the index
  the IDX map names. So named features used the wrong joint
  (e.g. `IDX["PELVIS"]=0` → COCO nose; `IDX["L_HIP"]=1` → left eye).
- Feature-level: `Dist_Hand_Shoulder_L` = 0.68 m on cached joints vs 0.14 m on correct
  joints; `Dispersion_Head` 0.54 vs 0.19. Not invariant — the values genuinely change.

**Red-team.** An adversarial agent tried five falsification attacks (hidden remap;
wrong reference; single-fragment artifact; feature no-op; "WHAM-24 == SMPL-24"). All
five failed to break it; the claim survived across 10 fragments spanning all tiers.

**The fix.** `joints = SMPL.J_regressor (24×6890) @ verts`. Works for **all** tiers
including T0, because it needs only `verts` (which every fragment has) and the `verts`
are a valid SMPL mesh — only the global root orientation is broken, not the surface.

**Integration status (now merged):** the fix is applied in the pipeline at
`core/wham_inference.py` (the LMA call site now regresses SMPL-24 from `verts` via
`processor.network.smpl.J_regressor` instead of slicing `joints_world[:, :24]`). It is
forward-only: existing cached `lma_features_*.npy` (the poster artifacts) are untouched,
so the poster numbers still reproduce bit-exact (verified: 4-way LogReg 0.5728 / RF
0.5742). A regression test `tests/test_joint_convention.py` pins the fixed contract
(joints fed to LMA are anatomically-valid SMPL-24) and is green.

### Two ruled-out red herrings
- **Floor / Y-axis.** `_normalize_pose_to_floor` assumes Y-down + floor=0, but WHAM is
  Y-up with feet ungrounded. **Proven a no-op:** all 55 features are invariant to a
  constant Y-offset and a Y sign-flip (they are within-pose distances or temporal
  derivative magnitudes). No absolute/signed-height feature exists.
- **World-space tipping.** WHAM's global/SLAM stage fails on much of this footage
  (handheld/crowd/social) → bodies are tipped/horizontal in world coords (~60–80% of
  sampled frames > 45° from vertical). But the LMA features are rotation-invariant, so
  this barely affects them. (It *does* wreck naïve world-space mesh rendering — see §5.)

---

## 2. Controlled experiment: V0 (paper) vs V_clean (joint fix only)

Single variable changed: joints come from `J_regressor @ verts` (true SMPL-24) instead
of `joints_world[:,:24]`. Same verts, volumes, floor, fps, classifier, splits, seed.
V0 reproduces the poster numbers bit-exact (4-way LogReg 0.5728 / RF 0.5742).

| Model | 4-way V0→V_clean | 3-way (drop) | Binary |
|---|---|---|---|
| LogReg | 0.5728 → 0.6002 (**+2.7**) | 0.7054 → 0.7070 (+0.2) | 0.7869 → 0.7920 (+0.5) |
| RandomForest | 0.5742 → 0.6030 (**+2.9**) | 0.6989 → 0.7060 (+0.7) | 0.7824 → 0.8041 (**+2.2**) |
| Temporal CNN | 0.6005 → 0.6353 (**+3.5**) | 0.7110 → 0.7296 (+1.9) | 0.8104 → 0.8146 (+0.4) |
| Transformer | 0.5888 → 0.6251 (**+3.6**) | 0.7250 → 0.7327 (+0.8) | 0.8254 → 0.8305 (+0.5) |

(3-way-merge: CNN +2.3, Transformer +0.6. sklearn MLPs: 4-way +1–2 pp, noisy — small-data
variance, not a config issue; trust LogReg/RF/temporal.)

**Verdict.** The bug capped performance, consistently and a bit more for the
trajectory-based deep nets (~+3.5 pp on 4-way), but it is a **correction, not a
transformation** (60 → 63.5%, not 60 → 80%). The lift concentrates on the fine-grained
4-way task and ~vanishes on gross binary — i.e. correct joints help where inter-tier
detail matters. Even mislabeled, the joints were *real, consistent* points, so the
features still carried genuine movement structure.

**Framing for the paper:** poster numbers stand and reproduce; report this as
"we identified and fixed a joint-convention bug; corrected numbers are modestly higher;
conclusions unchanged."

---

## 3. Why the ceiling is representational (feature-set limits)

The 55-feature descriptor (`lma_extractor.py:96-298`) decomposes as: 6 key-joint
velocities + 28 Effort (KE/Accel/Jerk/Directness per key joint + 4 global) + 8 Space
(5 dispersion + 3 pelvis-trajectory) + 1 body_volume + 12 Body (6 distances + 6
initiation). `KEY_JOINTS = {HEAD, PELVIS, L/R_WRIST, L/R_ANKLE}`.

- **Entirely positional**: Euclidean distances + 1st/2nd/3rd position derivatives.
  **No angular/articulation feature at all** (no joint angles, no segment orientations).
- **Pelvis is present** (a KEY_JOINT and the trajectory root) but only **as a translating
  point** — pelvic **tilt/rotation** is never measured.
- **Hips (L_HIP/R_HIP) are unused.** **No periodicity/rhythm** feature exists.
- The suggestiveness-relevant cues — pelvic tilt, hip isolation/circles, rhythm,
  orientation-to-viewer — are **angular/periodic** and thus absent, *even though the
  SMPL angles sit unused in `pose_world` (72-d)*. Cheap, well-defined future work:
  add pelvis/hip articulation + periodicity features sourced from `pose_world`.
- Deeper point: "sexual suggestiveness" lives substantially **off the skeleton**
  (appearance, context, camera, interaction) and single-person tracks lose interaction.
  Adjacent tiers (artistic T1 vs suggestive T2) are near-identical in Laban space — the
  boundary is semantic, not kinematic. Consistent with 3-way (drop T1) ≈ 0.71 ≫ 4-way ≈ 0.57.

---

## 4. Temporal-length analysis (TemporalCNN, V_clean)

Accuracy binned by fragment length (frames). Model truncates at `T_FIXED=256`, but
p95 length = 242, so ~95% of clips are never truncated — the trends are real.

**Overall by length** (4-way / 3-way / binary):

| length | 4-way | 3-way | binary |
|---|---|---|---|
| 15–29 | 0.717 | 0.763 | 1.000 |
| 30–59 | 0.661 | 0.780 | 0.940 |
| 60–119 | 0.581 | 0.685 | 0.774 |
| 120–239 | 0.651 | 0.727 | 0.762 |
| 240+ | 0.686 | 0.742 | 0.696 |
| Pearson(len,correct) | +0.02 | −0.04 | **−0.19** |

**Per-class by length (V_clean) — the key result, opposing trends:**

| | short → long |
|---|---|
| T0 (everyday) | 0.54 → **0.96** (rises) |
| T2 (suggestive) | 0.32 → 0.71 (rises, 4-way) |
| T1 (artistic) | 0.84 → **0.21** (collapses) |
| T3 (explicit) | 0.77 → **0.31** (collapses) |
| SFW (binary) | 0.88 → 0.99 (rises) |
| NSFW (binary) | 1.00 → **0.21** (collapses) |

Mean length of correct vs wrong confirms: NSFW correct = 61 frames, wrong = 161;
SFW correct = 135, wrong = 92 (same direction for T0/T2 vs T1/T3).

**Mechanism (well-supported):** the suggestive/explicit signal is **bursty and
localized**, not sustained. Long NSFW clips contain non-suggestive motion (setup,
transitions, talking) that **dilutes** the aggregate → it drifts toward SFW. SFW classes
instead benefit from more frames of mundane/structured motion. Binary shows it most
starkly (overall accuracy *decreases* with length; NSFW 1.00→0.21).

**Implication for the paper / ablation:** clip length is a confound correlated with
per-tier accuracy in opposite directions. **Fixed-length windows or segment-level (not
whole-clip) labeling** should help directly — the model should see the suggestive burst,
not a diluted average. (Open: confirm by sliding a window over long NSFW clips and showing
the NSFW score spikes on sub-segments.)

---

## 5. Mesh-video ablation (video-classifier baseline)

Goal: train a video model on rendered plain-mesh animations as a modality ablation.
Key facts learned:
- **Render from the canonical rebuild** (identity root + `pose_world` body pose), NOT the
  world `verts` (those are SLAM-tipped → bodies render lying down). Canonical = clean,
  upright, natural.
- **Original-camera view** is recoverable without re-running WHAM: PnP (canonical SMPL
  COCO joints via `J_regressor_coco` ↔ cached `keypoints_2d`) → per-frame camera, render
  via `cameras_from_opencv_projection`. Use **SQPNP** (ITERATIVE fails at high res →
  mesh behind camera). Reproj err ~50 px at 1080p. Per-frame PnP introduces mild camera
  jitter (fix: temporal-smooth rvec/tvec).
- **Chosen for the batch: custom follow-camera** (canonical mesh, body centered, no PnP →
  no jitter, consistent framing) — better for a classifier than variable original framing.
- **Coverage:** needs `pose_world` → ~18,961 fragments (T1-hailmary + T2 + T3). **T0
  (1,075) + old-T1 (498) lack `pose_world`/`keypoints_2d`** and need a WHAM re-run.
- **Throughput** (one A6000): 512² ≈ 7–10 h; 256² ≈ 3–4 h; native 1080p ≈ 40 h. Linear
  across GPUs (512² ≈ 2 h on 4 GPUs). Render is GPU-bound; npz I/O is negligible
  (only `pose_world`/`betas` are read, not `verts`).
- Overnight run launched 2026-05-29: 512², single GPU (tmux `meshrender`),
  output `/disk1/jaehoon/mesh_videos_followcam_512/T{tier}/<sha1>.mp4` + `manifest.csv`,
  resumable. Builder: `/tmp/render_followcam_batch.py`.

---

## 6. Open items / untested confounds
- **Label noise (untested, possibly larger than the joint fix):** some fragments track
  the wrong person (e.g. a breakdance fragment tracked a background spectator). Quantify
  prevalence; it caps accuracy independently of coordinates.
- A **de-tipped** (canonical-from-pose) feature variant would confirm tipping is harmless
  (available for ~92% of fragments with `pose_world`).
- Add pelvis/hip-articulation + periodicity features from `pose_world` and re-evaluate.
- Window-level NSFW scoring to confirm the bursty-signal mechanism.

## Artifacts / repro
- V0 baseline: `python scripts/analyze_lma_tiers.py --manifest data/manifest_paper_4way_*.csv --max-per-tier 1075` (+`--drop-tier1`, `--binary`).
- V_clean features: `/tmp/build_vclean.py` → `output/exp_clean/vclean_npy/` + `/tmp/vclean_4way.csv`, `/tmp/vclean_binary.csv`.
- V_clean classifiers: same `analyze_lma_tiers.py` on the vclean manifests; temporal nets `/tmp/tcnn_vclean.py`, `/tmp/ttrans_vclean.py` → `output/exp_clean/tcnn_vclean/`, `ttrans_vclean/`.
- (Note: `/tmp/*` is ephemeral — regenerate the builders/manifests if the box reboots.)
