#!/usr/bin/env python
"""
Contract unit tests for the WHAM inference output npz.

Module under test:
    /home/sogang/jaehoon/KineGuard/core/wham_inference.py
    -> KineGuardWHAMProcessor.run_pipeline(...)  (the np.savez at L222-232)
    -> process_single_video(...)

The contract is DERIVED FROM THE CODE (file:line references below), then asserted
against two real outputs. We do NOT relax any check to make a sample pass; a
failure that reveals a real code/data mismatch is reported as a finding.

Derived contract (wham_inference.py):
  np.savez(out_npz, ...)  @ L222-232 writes these keys:
    joints       <- data['joints_world']      L197,L224 ; built L175 as
                    (smpl_output.joints + trans_world).squeeze(0)  -> (T, J, 3) float32
                    NOTE: L175 comment claims (T, 45, 3) but WHAM's J_regressor_wham
                    yields 31 joints in practice -> contract J == 31.
    keypoints_2d <- data['keypoints_2d']       L200-201,L225 ; ViTPose pixel coords
                    (T, 17, 3) -> x, y, conf   (comment L200)
    verts        <- data['verts_world']        L202,L226 ; (smpl_output.vertices +
                    trans_world).squeeze(0)    -> (T, 6890, 3) float32  (comment L176)
    pose_world   <- data['pose_world']         L186,L227 ; concat(root_world_aa(.,3),
                    body_aa(.,69)) -> (T, 72) axis-angle
    betas        <- data['betas']              L184,L228 ; pred['betas'].squeeze(0)
                    -> (T, 10)  (SMPL shape, 10 PCA coeffs)
    trans_world  <- data['trans_world']        L191,L229 ; (T, 3) world translation
    frame_ids    <- frames (== data['frame_ids']) L183,L211,L230 ; (T,) int frame idx
    fps          <- fps                        L231 ; scalar, positive

  Internal-consistency guarantees implied by the code:
    * Every per-frame array has identical leading dim T (all derived from the same
      WHAM pred of one track).
    * len(frame_ids) == T.
    * Track-length filter @ L214: tracks with len(frames) < 15 are skipped, so any
      written fragment MUST have T >= 15.
    * All values finite (no NaN/Inf) for a usable kinematics file.
    * verts form a plausible adult-sized human mesh: overall bbox diagonal ~1-2 m.
    * fps > 0.

  Older-schema files may contain only a subset of keys (observed: fps, frame_ids,
  joints, verts). Such files are tested for the keys they DO have, and the missing
  keys are reported as a schema-divergence finding rather than a hard failure.
"""

import os
import sys
import numpy as np

FULL_NPZ = "/home/sogang/jaehoon/KineGuard/output/tier2_features/7559829673161624840/_clip_seg0/wham_fragment_id0.npz"
OLD_NPZ = "/home/sogang/jaehoon/tier0/acting in play/0bdVrgImymc_000020_000030/wham_fragment_id0.npz"

# Contract constants derived from the code / SMPL convention.
EXPECTED_KEYS = {
    "joints", "keypoints_2d", "verts", "pose_world",
    "betas", "trans_world", "frame_ids", "fps",
}
J_JOINTS = 31          # WHAM J_regressor_wham output width (NOT the stale 45 in L175 comment)
N_KP = 17              # ViTPose COCO-17
N_VERTS = 6890         # SMPL mesh
POSE_W = 72            # 3 (root aa) + 69 (body aa)
BETAS_W = 10           # SMPL shape PCA
TRANS_W = 3
MIN_TRACK = 15         # L214 filter
MESH_DIAG_LO, MESH_DIAG_HI = 1.0, 2.0   # plausible adult mesh bbox diagonal (m)

# Per-frame arrays whose leading dim must equal T.
PER_FRAME_KEYS = ["joints", "keypoints_2d", "verts", "pose_world", "betas",
                  "trans_world", "frame_ids"]

_results = []  # (label, passed, message)


def check(label, cond, detail=""):
    passed = bool(cond)
    _results.append((label, passed, detail))
    status = "PASS" if passed else "FAIL"
    line = f"[{status}] {label}"
    if detail:
        line += f"  -- {detail}"
    print(line)
    return passed


def info(label, detail):
    """Non-pass/fail informational finding."""
    print(f"[INFO] {label}  -- {detail}")


def shape_check(npz, prefix, key, expected_tail, T):
    """Assert key exists with shape (T, *expected_tail)."""
    if key not in npz.files:
        check(f"{prefix}: '{key}' present", False, "key missing")
        return
    a = npz[key]
    want = (T,) + tuple(expected_tail)
    check(f"{prefix}: {key} shape {want}", a.shape == want, f"got {a.shape}")
    check(f"{prefix}: {key} float32", a.dtype == np.float32, f"got {a.dtype}")
    check(f"{prefix}: {key} finite", np.isfinite(a).all(), "contains NaN/Inf" if not np.isfinite(a).all() else "")


def test_npz(path, label, require_full_schema):
    print(f"\n===== {label} =====")
    print(f"path: {path}")
    if not os.path.exists(path):
        check(f"{label}: file exists", False, "FILE MISSING")
        return
    check(f"{label}: file exists", True)

    d = np.load(path, allow_pickle=True)
    keys = set(d.files)
    print(f"keys present: {sorted(keys)}")

    # --- Key set ---
    missing = EXPECTED_KEYS - keys
    extra = keys - EXPECTED_KEYS
    if require_full_schema:
        check(f"{label}: full key set == contract", missing == set() and extra == set(),
              f"missing={sorted(missing)} extra={sorted(extra)}")
    else:
        # Older schema: report divergence as a finding, do not hard-fail the run.
        if missing:
            info(f"{label}: SCHEMA DIVERGENCE", f"missing contract keys {sorted(missing)}")
        if extra:
            info(f"{label}: extra keys beyond contract", f"{sorted(extra)}")
        check(f"{label}: keys are a subset of contract (no foreign keys)", extra == set(),
              f"foreign keys: {sorted(extra)}")

    # --- Determine T from frame_ids (L211: frames = data['frame_ids']) ---
    if "frame_ids" not in keys:
        check(f"{label}: frame_ids present (needed to fix T)", False, "missing")
        return
    fid = d["frame_ids"]
    check(f"{label}: frame_ids is 1-D", fid.ndim == 1, f"ndim={fid.ndim}")
    check(f"{label}: frame_ids integer dtype",
          np.issubdtype(fid.dtype, np.integer), f"dtype={fid.dtype}")
    T = fid.shape[0]
    print(f"T (len frame_ids) = {T}")

    # --- Track-length filter guarantee (L214: <15 skipped) ---
    check(f"{label}: T >= {MIN_TRACK} (track-length filter L214)", T >= MIN_TRACK, f"T={T}")

    # --- frame_ids monotonic non-decreasing & non-negative ---
    check(f"{label}: frame_ids non-negative", (fid >= 0).all(), f"min={int(fid.min())}")
    check(f"{label}: frame_ids non-decreasing", bool(np.all(np.diff(fid) >= 0)), "not sorted")

    # --- Per-frame leading-dim consistency: every present per-frame array == T ---
    for k in PER_FRAME_KEYS:
        if k in keys:
            a = d[k]
            check(f"{label}: {k} leading dim == T({T})", a.shape[0] == T, f"got {a.shape[0]}")

    # --- Shapes / dtypes / finiteness for present keys ---
    if "joints" in keys:
        shape_check(d, label, "joints", (J_JOINTS, 3), T)
    if "verts" in keys:
        shape_check(d, label, "verts", (N_VERTS, 3), T)
    if "keypoints_2d" in keys:
        shape_check(d, label, "keypoints_2d", (N_KP, 3), T)
    if "pose_world" in keys:
        shape_check(d, label, "pose_world", (POSE_W,), T)
    if "betas" in keys:
        shape_check(d, label, "betas", (BETAS_W,), T)
    if "trans_world" in keys:
        shape_check(d, label, "trans_world", (TRANS_W,), T)

    # --- fps scalar, positive ---
    if "fps" in keys:
        fps = d["fps"]
        check(f"{label}: fps scalar", fps.ndim == 0, f"shape={fps.shape}")
        fpsv = float(fps)
        check(f"{label}: fps > 0", fpsv > 0, f"fps={fpsv}")

    # --- verts: plausible adult human mesh, per-frame bbox diagonal ~1-2 m ---
    if "verts" in keys and d["verts"].shape == (T, N_VERTS, 3):
        verts = d["verts"]
        ext = verts.max(axis=1) - verts.min(axis=1)        # (T,3)
        diag = np.linalg.norm(ext, axis=1)                  # (T,)
        med = float(np.median(diag))
        check(f"{label}: median verts bbox diag in [{MESH_DIAG_LO},{MESH_DIAG_HI}] m",
              MESH_DIAG_LO <= med <= MESH_DIAG_HI, f"median diag={med:.3f} m "
              f"(min={diag.min():.3f}, max={diag.max():.3f})")

    # --- keypoints_2d: conf channel present, pixel coords sane ---
    if "keypoints_2d" in keys and d["keypoints_2d"].shape == (T, N_KP, 3):
        kp = d["keypoints_2d"]
        check(f"{label}: keypoints_2d coords non-negative-ish (pixel space)",
              kp[..., :2].min() >= -50, f"min xy={kp[..., :2].min():.1f}")
        conf = kp[..., 2]
        check(f"{label}: keypoints_2d conf in [0, ~1.1]",
              conf.min() >= 0 and conf.max() <= 1.2, f"conf range [{conf.min():.3f},{conf.max():.3f}]")

    # --- betas: SMPL shape coeffs in a sane magnitude band ---
    if "betas" in keys and d["betas"].shape == (T, BETAS_W):
        betas = d["betas"]
        check(f"{label}: betas magnitude reasonable (|b|<10)",
              np.abs(betas).max() < 10, f"max|beta|={np.abs(betas).max():.3f}")


def main():
    print("WHAM inference output -- contract tests")
    print("module: /home/sogang/jaehoon/KineGuard/core/wham_inference.py (savez @ L222-232)")

    test_npz(FULL_NPZ, "FULL", require_full_schema=True)
    test_npz(OLD_NPZ, "OLD", require_full_schema=False)

    total = len(_results)
    passed = sum(1 for _, p, _ in _results if p)
    failed = total - passed
    print("\n========== SUMMARY ==========")
    print(f"checks: {total}  passed: {passed}  failed: {failed}")
    if failed:
        print("\nFAILURES:")
        for lbl, p, det in _results:
            if not p:
                print(f"  - {lbl}  ({det})")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
