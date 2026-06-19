# The 61 LMA features

A precise, code-verified reference for the 61 Laban Movement Analysis descriptors the
pipeline computes per frame. This complements Appendix A/B of the paper (which groups the
features by Laban category) by giving the exact key names and column order produced by the
extractor — the same order stored in `feature_names` inside `reproduce/lma_data/features.npz`.

The math lives in `external/dance-style-recognition/src/utils/lma_extractor.py`
(`extract_all_features`). Running `LMAExtractor(window_size=55, fps=60, short_window=5)` on a
`T`-frame, 24-joint SMPL sequence yields a per-frame `(T, 61)` stream with these keys.
`scripts/extract_lma_features.py` stacks them in **sorted-key** order and takes per-feature
mean and standard deviation to form the 122-d fragment vector.

Every feature is per-frame: the per-joint dynamics are the mean over a causal trailing
55-frame window of lag-1 (`np.gradient`) finite differences; the Space/Directness, Initiation,
and trajectory features are evaluated per frame over that window (Directness and Initiation use
the `short_window`=5 lag).

| # | keys | Laban family |
|---|---|---|
| 1–6 | `Dist_Hand_Shoulder_{L,R}`, `Dist_Ankle_Knee_{L,R}`, `Dist_Hands`, `Dist_Feet` | Body — inter-joint distances |
| 7–11 | `Dispersion_{Head,R_Wrist,L_Wrist,R_Ankle,L_Ankle}` | Space — limb dispersion |
| 12–17 | `Angle_{LArm,RArm,Shoulders,LKnee,RKnee,Hips}` | Body — inter-joint angles |
| 18 | `body_volume` | Shape — convex-hull volume of the mesh |
| 19–42 | `{HEAD,PELVIS,L_WRIST,R_WRIST,L_ANKLE,R_ANKLE}_{vel,KE,Accel,Jerk}` | Effort — velocity + Weight/Time/Flow per joint |
| 43–45 | `Effort_{Weight,Time,Flow}_Global` | Effort — globals |
| 46–51 | `{HEAD,PELVIS,L_WRIST,R_WRIST,L_ANKLE,R_ANKLE}_Directness` | Effort — Space (Directness) per joint |
| 52 | `Effort_Space_Global` | Effort — Space global |
| 53–58 | `Initiation_{HEAD,PELVIS,L_WRIST,R_WRIST,L_ANKLE,R_ANKLE}` | Body — movement initiation |
| 59–61 | `Traj_{Path_Length,Displacement,Curvature}` | Space — pelvis trajectory |

## Why 61 and not 55

Turab et al. state 54–55 features but publish no feature list and no code, so their exact set
is unrecoverable. Reading their *described* descriptor literally — inter-joint distances **and
angles**, dispersions, the four Effort factors, Initiation, trajectory, and body volume — sums
to **61**, not 55. We therefore keep all 61 rather than arbitrarily drop six to force their
stated count (paper Appendix B): a superset is safer than guessing which six to discard. The
six key joints are head, pelvis, both wrists, both ankles, in the standard SMPL-24 layout; the
skeleton is `J_regressor @ verts` in the camera frame.

## Per-feature pre-processing

Two transforms run before any feature, so every row inherits them:

1. **Missing-frame imputation** — per joint and channel, linear `np.interp` over the frames
   that have data (an all-empty clip returns zeros). An engineering necessity for WHAM
   dropouts; not in the source papers.
2. **Floor-relative height** — the vertical (Y) channel is replaced with `floor_y - y` from a
   per-frame fitted floor model; X and Z are untouched. This realizes the papers' floor-relative
   body normalization. (The WHAM suggestive-motion path uses a flat floor, since WHAM joints are
   already ground-aligned.)

Smoothing is off by default; the per-joint velocity/acceleration/jerk are lag-1 `np.gradient`
finite differences (the faithful reading of the papers, which name velocity/acceleration but
give no finite-difference stencil), averaged over the causal window.
