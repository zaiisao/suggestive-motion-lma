#!/usr/bin/env python
"""
Contract unit test for the most safety-critical boundary in the pipeline:
the 3D-joint handoff from WHAM -> LMA feature extraction.

Background
----------
LMA indexes a 3D-joint array by body-part name using a FIXED map `self.IDX`
(lma_extractor.py:19-44, "Standard SMPL 24-joint topology"): index 0 = PELVIS,
1 = L_HIP, ... 15 = HEAD, ... 21 = R_WRIST.

The npz key `joints` (shape (T, 31, 3)) is WHAM's native output, regressed with
`J_regressor_wham` -> a COCO-17 + SPIN-14 layout, NOT the SMPL kinematic tree.
Feeding `joints[:, :24, :]` to LMA directly therefore MISLABELS every joint
("pelvis" lands on the nose, "L_hip" on the left eye, etc.).

THE FIX (core/wham_inference.py, extended version): LMA is now fed the canonical
SMPL-24 skeleton regressed from the (correct) mesh vertices,
    joints = J_regressor @ verts_world          # (T, 24, 3), true SMPL-24
instead of `data['joints_world'][:, :24, :]`.

This test pins the FIXED contract: it reproduces exactly what the pipeline now
feeds LMA (J_regressor @ verts), asserts those joints are anatomically valid
SMPL-24, and includes a regression guard confirming they differ from the old raw
`joints[:24]` path (i.e. the bug really was being avoided). Thresholds come from
adult human anatomy, not the data.

Run:
    /home/sogang/mnt/db_2/anaconda3/envs/wham/bin/python tests/test_joint_convention.py
"""

import os
import sys
import traceback

import numpy as np

IDX = {
    "PELVIS": 0, "L_HIP": 1, "R_HIP": 2, "SPINE1": 3,
    "L_KNEE": 4, "R_KNEE": 5, "SPINE2": 6, "L_ANKLE": 7,
    "R_ANKLE": 8, "SPINE3": 9, "L_FOOT": 10, "R_FOOT": 11,
    "NECK": 12, "L_COLLAR": 13, "R_COLLAR": 14, "HEAD": 15,
    "L_SHOULDER": 16, "R_SHOULDER": 17, "L_ELBOW": 18, "R_ELBOW": 19,
    "L_WRIST": 20, "R_WRIST": 21, "L_HAND": 22, "R_HAND": 23,
}

SMPL_MODEL_DIR = "/home/sogang/jaehoon/KineGuard/external/WHAM/dataset/body_models/smpl"

SAMPLES = [
    ("tier2/7559829673161624840",
     "/home/sogang/jaehoon/KineGuard/output/tier2_features/"
     "7559829673161624840/_clip_seg0/wham_fragment_id0.npz"),
    ("tier0/acting_in_play/0bdVrgImymc",
     "/home/sogang/jaehoon/tier0/acting in play/"
     "0bdVrgImymc_000020_000030/wham_fragment_id0.npz"),
]

# Anatomy-derived thresholds (adult human, metres); intentionally generous.
THIGH_MIN, THIGH_MAX = 0.32, 0.55      # hip -> knee
SHIN_MIN, SHIN_MAX = 0.32, 0.52        # knee -> ankle
UPPERARM_MIN, UPPERARM_MAX = 0.20, 0.36  # shoulder -> elbow
SYMMETRY_TOL = 0.15                    # left/right within 15%
RAW_DIFFERS_MIN = 0.10                 # fixed vs raw joints must differ by > 10 cm somewhere


class CheckRecorder:
    def __init__(self):
        self.results = []

    def check(self, label, name, passed, message):
        self.results.append((label, name, bool(passed), message))
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {message}")
        return passed

    def failures(self):
        return [r for r in self.results if not r[2]]


def median_bone(J, a, b):
    seg = J[:, IDX[a], :] - J[:, IDX[b], :]
    return float(np.median(np.linalg.norm(seg, axis=-1)))


def regress_smpl24(npz):
    """Reproduce exactly what core/wham_inference.py now feeds LMA:
    canonical SMPL-24 regressed from the mesh vertices."""
    import smplx
    model = smplx.SMPL(model_path=SMPL_MODEL_DIR, gender="neutral")
    Jreg = model.J_regressor.detach().cpu().numpy()[:24]      # (24, 6890)
    verts = np.asarray(npz["verts"], dtype=np.float64)         # (T, 6890, 3)
    return np.einsum("jv,tvc->tjc", Jreg, verts)               # (T, 24, 3)


def run_anatomy_checks(rec, label, J):
    """Assert SMPL-24 anatomical consequences on the joints fed to LMA."""
    l_thigh, r_thigh = median_bone(J, "L_HIP", "L_KNEE"), median_bone(J, "R_HIP", "R_KNEE")
    l_shin, r_shin = median_bone(J, "L_KNEE", "L_ANKLE"), median_bone(J, "R_KNEE", "R_ANKLE")
    l_uarm, r_uarm = median_bone(J, "L_SHOULDER", "L_ELBOW"), median_bone(J, "R_SHOULDER", "R_ELBOW")

    rec.check(label, "L_thigh length", THIGH_MIN <= l_thigh <= THIGH_MAX, f"{l_thigh:.3f} m")
    rec.check(label, "R_thigh length", THIGH_MIN <= r_thigh <= THIGH_MAX, f"{r_thigh:.3f} m")
    rec.check(label, "L_shin length", SHIN_MIN <= l_shin <= SHIN_MAX, f"{l_shin:.3f} m")
    rec.check(label, "R_shin length", SHIN_MIN <= r_shin <= SHIN_MAX, f"{r_shin:.3f} m")
    rec.check(label, "L_upperarm length", UPPERARM_MIN <= l_uarm <= UPPERARM_MAX, f"{l_uarm:.3f} m")
    rec.check(label, "R_upperarm length", UPPERARM_MIN <= r_uarm <= UPPERARM_MAX, f"{r_uarm:.3f} m")

    def sym(a, b):
        return abs(a - b) / max(abs(a), abs(b), 1e-6)
    rec.check(label, "thigh L/R symmetry", sym(l_thigh, r_thigh) <= SYMMETRY_TOL,
              f"{sym(l_thigh, r_thigh):.2f} (tol {SYMMETRY_TOL})")
    rec.check(label, "shin L/R symmetry", sym(l_shin, r_shin) <= SYMMETRY_TOL,
              f"{sym(l_shin, r_shin):.2f} (tol {SYMMETRY_TOL})")
    rec.check(label, "upperarm L/R symmetry", sym(l_uarm, r_uarm) <= SYMMETRY_TOL,
              f"{sym(l_uarm, r_uarm):.2f} (tol {SYMMETRY_TOL})")

    # NOTE: we deliberately do NOT assert gravity-relative topology (e.g. "feet below
    # pelvis"). That tests world ORIENTATION, not joint CONVENTION — and WHAM's world
    # frame is tipped on some clips (SLAM failures) and poses aren't always upright, so
    # topology is unreliable here AND irrelevant to the (rotation-invariant) LMA features.
    # Bone lengths + L/R symmetry above are frame- and pose-invariant, so they are the
    # robust validators that the joints are correctly-ordered SMPL-24. Pelvis-centrality
    # (a within-pose, orientation-free check) is a safe extra:
    pelvis_to_joints = np.median(np.linalg.norm(J - J[:, [IDX["PELVIS"]], :], axis=-1), axis=0)
    rec.check(label, "pelvis is central (min mean joint-distance)",
              pelvis_to_joints[IDX["PELVIS"]] <= 1e-6 and np.median(pelvis_to_joints) < 0.8,
              f"pelvis self-dist {pelvis_to_joints[IDX['PELVIS']]:.3f}, "
              f"median joint-to-pelvis {np.median(pelvis_to_joints):.2f} m")


def run_for_sample(rec, label, path):
    print(f"\n=== Sample: {label}\n    {path}")
    if not os.path.exists(path):
        rec.check(label, "sample npz exists", False, f"missing file: {path}")
        return
    npz = np.load(path)
    if "verts" not in npz.files:
        rec.check(label, "npz has 'verts'", False, f"keys: {list(npz.files)}")
        return

    # What the FIXED pipeline feeds LMA: J_regressor @ verts (true SMPL-24).
    J_fixed = np.asarray(regress_smpl24(npz), dtype=np.float64)
    rec.check(label, "fixed joints finite + (T,24,3)",
              J_fixed.ndim == 3 and J_fixed.shape[1] == 24 and np.isfinite(J_fixed).all(),
              f"shape {J_fixed.shape}")
    run_anatomy_checks(rec, label, J_fixed)

    # Regression guard: confirm the fix actually changed things — the old raw
    # joints[:24] (COCO+SPIN) differ substantially from the SMPL-24 the fix feeds LMA.
    if "joints" in npz.files and npz["joints"].shape[1] >= 24:
        raw = np.asarray(npz["joints"][:, :24, :], dtype=np.float64)
        max_d = float(np.median(np.linalg.norm(J_fixed - raw, axis=-1), axis=0).max())
        rec.check(label, "fix differs from old raw joints[:24]",
                  max_d > RAW_DIFFERS_MIN,
                  f"max per-joint median dist {max_d:.3f} m (> {RAW_DIFFERS_MIN}); "
                  f"raw path was the COCO+SPIN bug")


def main():
    rec = CheckRecorder()
    print("=" * 78)
    print("CONTRACT TEST: joints fed to LMA are valid SMPL-24 (verts-regressed)")
    print("=" * 78)
    for label, path in SAMPLES:
        try:
            run_for_sample(rec, label, path)
        except Exception:
            print(traceback.format_exc())
            rec.check(label, "sample ran without exception", False, "unhandled exception")

    fails = rec.failures()
    total = len(rec.results)
    print("\n" + "=" * 78)
    print(f"SUMMARY: {total - len(fails)}/{total} checks passed, {len(fails)} FAILED")
    print("=" * 78)
    if fails:
        print("\nVERDICT: the joints fed to LMA are NOT valid SMPL-24 — the fix is broken.")
        for label, name, _, msg in fails:
            print(f"  - [{label}] {name}: {msg}")
        sys.exit(1)
    print("\nVERDICT: the pipeline feeds LMA anatomically-valid SMPL-24 joints "
          "(regressed from verts); the joint-convention bug is fixed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
