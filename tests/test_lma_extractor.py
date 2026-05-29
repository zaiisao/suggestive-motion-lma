"""
Contract unit tests for the LMA (Laban Movement Analysis) feature extractor.

Module under test:
  external/dance-style-recognition/src/utils/lma_extractor.py
      class LMAExtractor.extract_all_features(all_joints, all_volumes, all_floor_models)
  external/dance-style-recognition/src/process_lma_features.py
      compute_lma_descriptor(...), IdentityFloor

Contract derived from the code (file:line references in comments below). These tests
do NOT modify pipeline code; they only assert the derived input/output contract and
known-input geometry. Run with the wham env python.

    /home/sogang/mnt/db_2/anaconda3/envs/wham/bin/python tests/test_lma_extractor.py
"""
import sys
import traceback

import numpy as np

SRC = "/home/sogang/jaehoon/KineGuard/external/dance-style-recognition/src"
sys.path.insert(0, SRC)

from utils.lma_extractor import LMAExtractor  # noqa: E402
from process_lma_features import compute_lma_descriptor, IdentityFloor  # noqa: E402

SAMPLE_NPY = (
    "/home/sogang/jaehoon/KineGuard/output/tier2_features/"
    "7559829673161624840/_clip_seg0/lma_features_id0.npy"
)

# ---------------------------------------------------------------------------
# Derived contract constants
# ---------------------------------------------------------------------------
# lma_extractor.py:18-44  self.IDX  -> SMPL 24-joint topology, ordered indices 0..23.
N_JOINTS = 24
IDX = {
    "PELVIS": 0, "L_HIP": 1, "R_HIP": 2, "SPINE1": 3, "L_KNEE": 4, "R_KNEE": 5,
    "SPINE2": 6, "L_ANKLE": 7, "R_ANKLE": 8, "SPINE3": 9, "L_FOOT": 10, "R_FOOT": 11,
    "NECK": 12, "L_COLLAR": 13, "R_COLLAR": 14, "HEAD": 15, "L_SHOULDER": 16,
    "R_SHOULDER": 17, "L_ELBOW": 18, "R_ELBOW": 19, "L_WRIST": 20, "R_WRIST": 21,
    "L_HAND": 22, "R_HAND": 23,
}
# process_lma_features.py:138-162 + extract_all_features docstring: exactly 55 features.
N_FEATURES = 55

# process_lma_features.py:274  feature_keys = sorted(lma_dict.keys())  -> column order is
# the alphabetical sort of the feature-name keys. Verified empirically against the module.
EXPECTED_SORTED_KEYS = [
    "Dispersion_Head", "Dispersion_L_Ankle", "Dispersion_L_Wrist", "Dispersion_R_Ankle",
    "Dispersion_R_Wrist", "Dist_Ankle_Knee_L", "Dist_Ankle_Knee_R", "Dist_Feet",
    "Dist_Hand_Shoulder_L", "Dist_Hand_Shoulder_R", "Dist_Hands", "Effort_Flow_Global",
    "Effort_Space_Global", "Effort_Time_Global", "Effort_Weight_Global", "HEAD_Accel",
    "HEAD_Directness", "HEAD_Jerk", "HEAD_KE", "HEAD_vel", "Initiation_HEAD",
    "Initiation_L_ANKLE", "Initiation_L_WRIST", "Initiation_PELVIS", "Initiation_R_ANKLE",
    "Initiation_R_WRIST", "L_ANKLE_Accel", "L_ANKLE_Directness", "L_ANKLE_Jerk",
    "L_ANKLE_KE", "L_ANKLE_vel", "L_WRIST_Accel", "L_WRIST_Directness", "L_WRIST_Jerk",
    "L_WRIST_KE", "L_WRIST_vel", "PELVIS_Accel", "PELVIS_Directness", "PELVIS_Jerk",
    "PELVIS_KE", "PELVIS_vel", "R_ANKLE_Accel", "R_ANKLE_Directness", "R_ANKLE_Jerk",
    "R_ANKLE_KE", "R_ANKLE_vel", "R_WRIST_Accel", "R_WRIST_Directness", "R_WRIST_Jerk",
    "R_WRIST_KE", "R_WRIST_vel", "Traj_Curvature", "Traj_Displacement",
    "Traj_Path_Length", "body_volume",
]


# ---------------------------------------------------------------------------
# Tiny assertion harness (prints PASS/FAIL, tallies results)
# ---------------------------------------------------------------------------
_PASS = 0
_FAIL = 0
_FAILURES = []


def check(name, cond, detail=""):
    global _PASS, _FAIL
    if cond:
        _PASS += 1
        print(f"  [PASS] {name}")
    else:
        _FAIL += 1
        _FAILURES.append((name, detail))
        print(f"  [FAIL] {name}" + (f"  -- {detail}" if detail else ""))


# ---------------------------------------------------------------------------
# Helpers to build controlled synthetic skeletons
# ---------------------------------------------------------------------------
def make_static_skeleton():
    """Build ONE static SMPL-24 pose with deliberately chosen segment lengths so
    that named Body-component distance features have analytically-known values.

    Coordinates are placed in the raw camera frame (Y is vertical). IdentityFloor
    negates Y in _normalize_pose_to_floor (lma_extractor.py:92), but Euclidean
    inter-joint distances are invariant to that axis flip, so the asserted
    distances below survive normalization unchanged.
    """
    sk = np.zeros((N_JOINTS, 3), dtype=np.float64)

    # Chosen ground-truth lengths (meters):
    SHIN = 0.42          # ankle <-> knee  (Dist_Ankle_Knee_*)
    UPPERARM = 0.31      # wrist <-> shoulder (Dist_Hand_Shoulder_*)
    HANDS_GAP = 0.55     # L_WRIST <-> R_WRIST (Dist_Hands)
    FEET_GAP = 0.18      # L_ANKLE <-> R_ANKLE (Dist_Feet)
    HEAD_SPINE2 = 0.50   # HEAD <-> SPINE2 (Dispersion_Head)

    # Anchor PELVIS at a non-trivial location to also exercise translation logic.
    sk[IDX["PELVIS"]] = [0.0, 1.00, 3.0]
    sk[IDX["SPINE2"]] = [0.0, 1.30, 3.0]

    # HEAD directly above SPINE2 by HEAD_SPINE2.
    sk[IDX["HEAD"]] = sk[IDX["SPINE2"]] + np.array([0.0, HEAD_SPINE2, 0.0])

    # Knees / ankles: place ankles, knees SHIN above each ankle (in Y).
    # Feet separated horizontally (X) by FEET_GAP, centered on pelvis X.
    half = FEET_GAP / 2.0
    sk[IDX["L_ANKLE"]] = [-half, 0.05, 3.0]
    sk[IDX["R_ANKLE"]] = [+half, 0.05, 3.0]
    sk[IDX["L_KNEE"]] = sk[IDX["L_ANKLE"]] + np.array([0.0, SHIN, 0.0])
    sk[IDX["R_KNEE"]] = sk[IDX["R_ANKLE"]] + np.array([0.0, SHIN, 0.0])

    # Shoulders; wrists UPPERARM straight below each shoulder (pure Y offset).
    sk[IDX["L_SHOULDER"]] = [-0.20, 1.25, 3.0]
    sk[IDX["R_SHOULDER"]] = [+0.20, 1.25, 3.0]
    sk[IDX["L_WRIST"]] = sk[IDX["L_SHOULDER"]] - np.array([0.0, UPPERARM, 0.0])
    sk[IDX["R_WRIST"]] = sk[IDX["R_SHOULDER"]] - np.array([0.0, UPPERARM, 0.0])

    # Force the wrist-wrist horizontal gap to exactly HANDS_GAP (override X).
    hw = HANDS_GAP / 2.0
    sk[IDX["L_WRIST"]][0] = -hw
    sk[IDX["R_WRIST"]][0] = +hw
    # And the upper-arm length must remain UPPERARM: recompute shoulder X = wrist X
    # so Dist_Hand_Shoulder stays a pure vertical UPPERARM.
    sk[IDX["L_SHOULDER"]][0] = -hw
    sk[IDX["R_SHOULDER"]][0] = +hw

    truth = dict(SHIN=SHIN, UPPERARM=UPPERARM, HANDS_GAP=HANDS_GAP,
                 FEET_GAP=FEET_GAP, HEAD_SPINE2=HEAD_SPINE2)
    return sk, truth


def run_extractor(seq, vols=None, floors=None, fps=30, window_size=10, short_window=3):
    n = len(seq)
    if vols is None:
        vols = [0.07] * n
    if floors is None:
        floors = [IdentityFloor() for _ in range(n)]
    return compute_lma_descriptor(
        list(seq), list(vols), list(floors),
        fps=fps, window_size=window_size, short_window=short_window,
    )


# ---------------------------------------------------------------------------
# TESTS
# ---------------------------------------------------------------------------
def test_idx_map_contract():
    """lma_extractor.py:18-44 -- IDX must be a 24-joint SMPL map, indices 0..23 unique."""
    print("\n[test] IDX joint-map contract (lma_extractor.py:18-44)")
    ex = LMAExtractor()
    check("IDX has 24 joints", len(ex.IDX) == N_JOINTS, f"got {len(ex.IDX)}")
    check("IDX indices are 0..23 unique",
          sorted(ex.IDX.values()) == list(range(N_JOINTS)),
          f"got {sorted(ex.IDX.values())}")
    check("IDX matches expected SMPL ordering", ex.IDX == IDX)
    check("KEY_JOINTS are the documented 6",
          ex.KEY_JOINTS == ["HEAD", "PELVIS", "L_WRIST", "R_WRIST", "L_ANKLE", "R_ANKLE"])


def test_output_shape_and_columns():
    """process_lma_features.py:274-275 -- matrix is (frames, 55), columns = sorted keys."""
    print("\n[test] Output shape / feature count / column order")
    N = 30
    rng = np.random.RandomState(1)
    seq = [rng.randn(N_JOINTS, 3) for _ in range(N)]
    d, m = run_extractor(seq)

    check("returns dict + matrix", isinstance(d, dict) and isinstance(m, np.ndarray))
    check("exactly 55 features in dict", len(d) == N_FEATURES, f"got {len(d)}")
    check("matrix shape == (frames, 55)", m.shape == (N, N_FEATURES), f"got {m.shape}")
    sorted_keys = sorted(d.keys())
    check("sorted keys == expected 55-name contract", sorted_keys == EXPECTED_SORTED_KEYS,
          f"diff={set(sorted_keys) ^ set(EXPECTED_SORTED_KEYS)}")
    # Every per-key vector must have length == n_frames.
    bad = [k for k in d if len(d[k]) != N]
    check("each feature vector length == n_frames", not bad, f"bad keys={bad[:3]}")
    # Matrix column c must equal dict[sorted_keys[c]].
    col_ok = all(np.array_equal(m[:, c], d[sorted_keys[c]]) for c in range(N_FEATURES))
    check("matrix columns follow sorted-key order", col_ok)
    check("output has no NaN/Inf", np.all(np.isfinite(m)))


def test_determinism():
    """Same input -> byte-identical output (no hidden randomness/state)."""
    print("\n[test] Determinism (identical input -> identical output)")
    N = 25
    rng = np.random.RandomState(7)
    seq = [rng.randn(N_JOINTS, 3) for _ in range(N)]
    vols = list(rng.rand(N) * 0.1)
    _, m1 = run_extractor([s.copy() for s in seq], vols=list(vols))
    _, m2 = run_extractor([s.copy() for s in seq], vols=list(vols))
    check("two runs produce byte-identical matrices",
          m1.shape == m2.shape and np.array_equal(m1, m2),
          f"max|diff|={np.abs(m1 - m2).max() if m1.shape == m2.shape else 'shape-mismatch'}")


def test_synthetic_skeleton_geometry():
    """KNOWN-INPUT SANITY: a static synthetic skeleton with chosen segment lengths must
    produce anatomically-correct Body-component distance features (lma_extractor.py:237-270).
    """
    print("\n[test] Synthetic-skeleton geometry sanity (lma_extractor.py:237-270)")
    sk, T = make_static_skeleton()
    N = 12
    seq = [sk.copy() for _ in range(N)]
    d, _ = run_extractor(seq, window_size=8, short_window=3)

    # Use a late frame (full causal window populated); distances are frame-constant anyway.
    t = N - 1
    tol = 1e-6

    def feat(name):
        return d[name][t]

    check("Dist_Ankle_Knee_L == shin length",
          abs(feat("Dist_Ankle_Knee_L") - T["SHIN"]) < tol,
          f"got {feat('Dist_Ankle_Knee_L'):.6f} vs {T['SHIN']}")
    check("Dist_Ankle_Knee_R == shin length",
          abs(feat("Dist_Ankle_Knee_R") - T["SHIN"]) < tol,
          f"got {feat('Dist_Ankle_Knee_R'):.6f} vs {T['SHIN']}")
    check("Dist_Hand_Shoulder_L == upper-arm length",
          abs(feat("Dist_Hand_Shoulder_L") - T["UPPERARM"]) < tol,
          f"got {feat('Dist_Hand_Shoulder_L'):.6f} vs {T['UPPERARM']}")
    check("Dist_Hand_Shoulder_R == upper-arm length",
          abs(feat("Dist_Hand_Shoulder_R") - T["UPPERARM"]) < tol,
          f"got {feat('Dist_Hand_Shoulder_R'):.6f} vs {T['UPPERARM']}")
    check("Dist_Hands == chosen wrist-wrist gap",
          abs(feat("Dist_Hands") - T["HANDS_GAP"]) < tol,
          f"got {feat('Dist_Hands'):.6f} vs {T['HANDS_GAP']}")
    check("Dist_Feet == chosen ankle-ankle gap",
          abs(feat("Dist_Feet") - T["FEET_GAP"]) < tol,
          f"got {feat('Dist_Feet'):.6f} vs {T['FEET_GAP']}")
    check("Dispersion_Head == HEAD<->SPINE2 length",
          abs(feat("Dispersion_Head") - T["HEAD_SPINE2"]) < tol,
          f"got {feat('Dispersion_Head'):.6f} vs {T['HEAD_SPINE2']}")

    # Static figure: velocity/accel/jerk features and trajectory path must be ~0.
    check("static figure -> PELVIS_vel ~ 0", abs(feat("PELVIS_vel")) < 1e-9,
          f"got {feat('PELVIS_vel'):.3e}")
    check("static figure -> Effort_Weight_Global ~ 0", abs(feat("Effort_Weight_Global")) < 1e-9,
          f"got {feat('Effort_Weight_Global'):.3e}")
    check("static figure -> Traj_Path_Length ~ 0", abs(feat("Traj_Path_Length")) < 1e-9,
          f"got {feat('Traj_Path_Length'):.3e}")
    check("body_volume passes through input volume",
          abs(feat("body_volume") - 0.07) < tol, f"got {feat('body_volume'):.6f}")


def test_translation_invariance():
    """Inter-joint distance features must be invariant to a constant spatial translation
    of the whole skeleton (math: dist(a-c, b-c) == dist(a,b)). Note IdentityFloor flips Y
    (lma_extractor.py:92) but a constant translation still cancels in pairwise distances.
    """
    print("\n[test] Translation invariance of distance features")
    sk, _ = make_static_skeleton()
    N = 12
    base = [sk.copy() for _ in range(N)]
    shift = np.array([0.37, -0.22, 0.91])
    shifted = [sk.copy() + shift for _ in range(N)]

    d0, _ = run_extractor(base, window_size=8, short_window=3)
    d1, _ = run_extractor(shifted, window_size=8, short_window=3)

    dist_keys = [k for k in EXPECTED_SORTED_KEYS
                 if k.startswith("Dist_") or k.startswith("Dispersion_")]
    t = N - 1
    worst = 0.0
    worst_key = None
    for k in dist_keys:
        diff = abs(d0[k][t] - d1[k][t])
        if diff > worst:
            worst, worst_key = diff, k
    check("all Dist_/Dispersion_ features invariant under translation", worst < 1e-9,
          f"worst={worst:.3e} at {worst_key}")


def test_velocity_invariance_constant_translation():
    """A skeleton undergoing the SAME constant per-frame translation as a moving baseline
    should yield identical velocity-derived Effort features regardless of starting offset
    (kinematics depend on np.gradient, which removes constant position offsets)."""
    print("\n[test] Velocity/Effort invariance to constant position offset")
    sk, _ = make_static_skeleton()
    N = 20
    vel_per_frame = np.array([0.01, 0.02, -0.005])
    seqA = [sk.copy() + vel_per_frame * i for i in range(N)]
    offset = np.array([5.0, -3.0, 2.0])
    seqB = [sk.copy() + offset + vel_per_frame * i for i in range(N)]

    dA, _ = run_extractor(seqA, window_size=8, short_window=3)
    dB, _ = run_extractor(seqB, window_size=8, short_window=3)
    t = N - 1
    kin_keys = [k for k in EXPECTED_SORTED_KEYS
                if k.endswith(("_vel", "_KE", "_Accel", "_Jerk"))
                or k.startswith("Effort_")]
    worst = max(abs(dA[k][t] - dB[k][t]) for k in kin_keys)
    check("velocity/effort features invariant to constant position offset",
          worst < 1e-9, f"worst|diff|={worst:.3e}")


def test_imputation_of_empty_frames():
    """lma_extractor.py:54-73 -- empty-list frames are linearly interpolated; an all-empty
    sequence yields zeros and must not crash."""
    print("\n[test] Missing-frame imputation contract (lma_extractor.py:54-73)")
    ex = LMAExtractor()
    sk, _ = make_static_skeleton()
    seq = [sk.copy(), [], [], sk.copy()]
    out = ex._impute_missing_data(seq)
    check("imputation output shape (n,24,3)", out.shape == (4, N_JOINTS, 3), f"got {out.shape}")
    # Endpoints preserved, middle interpolated to same constant pose.
    check("imputed gap matches surrounding constant pose",
          np.allclose(out[1], sk) and np.allclose(out[2], sk))
    allempty = ex._impute_missing_data([[], []])
    check("all-empty sequence -> zeros (n,24,3)",
          allempty.shape == (2, N_JOINTS, 3) and np.all(allempty == 0))


def test_real_sample_matches_contract():
    """The committed real output vector must match the (frames, 55) matrix contract."""
    print("\n[test] Real sample output matches contract")
    try:
        arr = np.load(SAMPLE_NPY, allow_pickle=True)
    except Exception as e:
        check("real sample loads", False, f"load error: {e}")
        return
    check("real sample is 2D ndarray", arr.ndim == 2, f"ndim={arr.ndim}")
    check("real sample has 55 feature columns", arr.shape[1] == N_FEATURES,
          f"shape={arr.shape}")
    check("real sample is finite", np.all(np.isfinite(arr)))


def main():
    tests = [
        test_idx_map_contract,
        test_output_shape_and_columns,
        test_determinism,
        test_synthetic_skeleton_geometry,
        test_translation_invariance,
        test_velocity_invariance_constant_translation,
        test_imputation_of_empty_frames,
        test_real_sample_matches_contract,
    ]
    for t in tests:
        try:
            t()
        except Exception:
            global _FAIL
            _FAIL += 1
            print(f"  [FAIL] {t.__name__} raised an exception:")
            traceback.print_exc()
            _FAILURES.append((t.__name__, "raised exception"))

    print("\n" + "=" * 60)
    print(f"RESULT: {_PASS} passed, {_FAIL} failed, {_PASS + _FAIL} checks total")
    print("=" * 60)
    if _FAILURES:
        print("FAILURES:")
        for name, detail in _FAILURES:
            print(f"  - {name}: {detail}")
    sys.exit(1 if _FAIL else 0)


if __name__ == "__main__":
    main()
