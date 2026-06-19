"""Known-answer contract tests for the 61-feature LMA extractor.

Builds synthetic skeletons whose feature values are analytically known and checks
the restored (data-producing) extractor reproduces them. Run with the wham env:
    python tests/test_lma_extractor.py

The extractor emits 61 per-frame features:
  per-joint {HEAD,PELVIS,L_WRIST,R_WRIST,L_ANKLE,R_ANKLE}_{vel,KE,Accel,Jerk,Directness} (30)
  Effort_{Weight,Time,Flow,Space}_Global (4)
  Dispersion_{Head,R_Wrist,L_Wrist,R_Ankle,L_Ankle} (5)
  Dist_{Hand_Shoulder_L,Hand_Shoulder_R,Ankle_Knee_L,Ankle_Knee_R,Hands,Feet} (6)
  Angle_{LArm,RArm,Shoulders,LKnee,RKnee,Hips} (6)
  Initiation_{HEAD,PELVIS,L_WRIST,R_WRIST,L_ANKLE,R_ANKLE} (6)
  Traj_{Path_Length,Displacement,Curvature} (3)
  body_volume (1)
"""
import os
import sys

import numpy as np

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "external", "dance-style-recognition", "src")
sys.path.insert(0, SRC)
from utils.lma_extractor import LMAExtractor  # noqa: E402
from lma_descriptor import IdentityFloor  # noqa: E402  (exercise the production flat floor)

IDX = LMAExtractor().IDX
PASS = FAIL = 0


def chk(name, cond, detail=""):
    global PASS, FAIL
    print(("  [PASS] " if cond else "  [FAIL] ") + name + ("" if cond else f"  -- {detail}"))
    PASS += bool(cond); FAIL += (not cond)



def skeleton():
    s = np.zeros((24, 3))
    s[IDX["PELVIS"]] = [0, 1.0, 0]; s[IDX["SPINE1"]] = [0, 1.2, 0]; s[IDX["HEAD"]] = [0, 1.7, 0]
    s[IDX["L_SHOULDER"]] = [-0.2, 1.4, 0]; s[IDX["R_SHOULDER"]] = [0.2, 1.4, 0]
    s[IDX["L_ANKLE"]] = [-0.1, 0.1, 0]; s[IDX["R_ANKLE"]] = [0.1, 0.1, 0]
    s[IDX["L_KNEE"]] = [-0.12, 0.5, 0]; s[IDX["R_KNEE"]] = [0.12, 0.5, 0]
    s[IDX["L_HIP"]] = [-0.1, 0.95, 0]; s[IDX["R_HIP"]] = [0.1, 0.95, 0]
    s[IDX["L_ELBOW"]] = [-0.25, 1.2, 0]; s[IDX["R_ELBOW"]] = [0.25, 1.2, 0]
    s[IDX["SPINE2"]] = [0, 1.3, -0.03]; s[IDX["SPINE3"]] = [0, 1.35, -0.03]; s[IDX["NECK"]] = [0, 1.55, 0]
    s[IDX["L_WRIST"]] = [-0.28, 1.05, 0.04]; s[IDX["R_WRIST"]] = [0.28, 1.05, 0.04]
    s[IDX["L_FOOT"]] = [-0.1, 0.0, 0.12]; s[IDX["R_FOOT"]] = [0.1, 0.0, 0.12]
    s[IDX["L_COLLAR"]] = [-0.08, 1.45, 0]; s[IDX["R_COLLAR"]] = [0.08, 1.45, 0]
    s[IDX["L_HAND"]] = [-0.3, 1.0, 0.05]; s[IDX["R_HAND"]] = [0.3, 1.0, 0.05]
    return s


def dist(s, a, b):
    return np.linalg.norm(s[IDX[a]] - s[IDX[b]])


EXPECTED_KEYS = sorted([
    "Angle_Hips", "Angle_LArm", "Angle_LKnee", "Angle_RArm", "Angle_RKnee", "Angle_Shoulders",
    "Dispersion_Head", "Dispersion_L_Ankle", "Dispersion_L_Wrist", "Dispersion_R_Ankle", "Dispersion_R_Wrist",
    "Dist_Ankle_Knee_L", "Dist_Ankle_Knee_R", "Dist_Feet", "Dist_Hand_Shoulder_L", "Dist_Hand_Shoulder_R", "Dist_Hands",
    "Effort_Flow_Global", "Effort_Space_Global", "Effort_Time_Global", "Effort_Weight_Global",
    "HEAD_Accel", "HEAD_Directness", "HEAD_Jerk", "HEAD_KE", "HEAD_vel",
    "Initiation_HEAD", "Initiation_L_ANKLE", "Initiation_L_WRIST", "Initiation_PELVIS", "Initiation_R_ANKLE", "Initiation_R_WRIST",
    "L_ANKLE_Accel", "L_ANKLE_Directness", "L_ANKLE_Jerk", "L_ANKLE_KE", "L_ANKLE_vel",
    "L_WRIST_Accel", "L_WRIST_Directness", "L_WRIST_Jerk", "L_WRIST_KE", "L_WRIST_vel",
    "PELVIS_Accel", "PELVIS_Directness", "PELVIS_Jerk", "PELVIS_KE", "PELVIS_vel",
    "R_ANKLE_Accel", "R_ANKLE_Directness", "R_ANKLE_Jerk", "R_ANKLE_KE", "R_ANKLE_vel",
    "R_WRIST_Accel", "R_WRIST_Directness", "R_WRIST_Jerk", "R_WRIST_KE", "R_WRIST_vel",
    "Traj_Curvature", "Traj_Displacement", "Traj_Path_Length", "body_volume",
])


def run(seq, volumes, w=10, fps=30.0):
    seq = np.asarray(seq, float); T = len(seq)
    return LMAExtractor(window_size=w, fps=fps, short_window=5).extract_all_features(
        [seq[t] for t in range(T)], volumes, [IdentityFloor()] * T)


print("[1] CONTRACT: 61 named per-frame features")
s = skeleton(); T = 80
static = [s.copy() for _ in range(T)]
vols = np.full(T, 0.0731)
d = run(static, vols)
ks = sorted(d)
chk("exactly 61 features", len(ks) == 61, f"n={len(ks)}")
chk("keys == 61-name contract", ks == EXPECTED_KEYS, f"diff={set(ks) ^ set(EXPECTED_KEYS)}")
chk("all six Angle_ present", all(f"Angle_{a}" in d for a in ["LArm", "RArm", "Shoulders", "LKnee", "RKnee", "Hips"]))
chk("each feature is a per-frame (T,) array", all(np.asarray(d[k]).shape == (T,) for k in ks))
chk("all finite", all(np.all(np.isfinite(np.asarray(d[k]))) for k in ks))

print("\n[2] STATIC GEOMETRY: distances / dispersions / volume vs hand-computed")
t = 1e-6
chk("Dist_Hands == |L_WRIST - R_WRIST|", abs(d["Dist_Hands"][0] - dist(s, "L_WRIST", "R_WRIST")) < t)
chk("Dist_Feet == |L_ANKLE - R_ANKLE|", abs(d["Dist_Feet"][0] - dist(s, "L_ANKLE", "R_ANKLE")) < t)
chk("Dist_Hand_Shoulder_L (uses WRIST)", abs(d["Dist_Hand_Shoulder_L"][0] - dist(s, "L_WRIST", "L_SHOULDER")) < t)
chk("Dispersion_Head == |HEAD - SPINE2|", abs(d["Dispersion_Head"][0] - dist(s, "HEAD", "SPINE2")) < t)
chk("body_volume == passed volume", abs(d["body_volume"][0] - 0.0731) < t)

print("\n[3] STATIC DYNAMICS: a motionless skeleton has zero velocity/accel/jerk")
for j in ["HEAD", "PELVIS", "L_WRIST"]:
    chk(f"{j}_vel == 0 (static)", abs(d[f"{j}_vel"][T // 2]) < 1e-6, f'{d[f"{j}_vel"][T//2]:.2e}')
    chk(f"{j}_Accel == 0 (static)", abs(d[f"{j}_Accel"][T // 2]) < 1e-6)
    chk(f"{j}_Jerk == 0 (static)", abs(d[f"{j}_Jerk"][T // 2]) < 1e-6)

print("\n[4] CONSTANT-VELOCITY TRANSLATION: vel = |dx|*fps, accel ~ 0, distances invariant")
dx, fps = 0.01, 30.0
moving = [s + np.array([i * dx, 0, 0]) for i in range(T)]
dm = run(moving, vols, fps=fps)
exp_vel = dx * fps
chk("HEAD_vel == |dx|*fps", abs(dm["HEAD_vel"][T // 2] - exp_vel) < 1e-4, f'{dm["HEAD_vel"][T//2]:.4f} vs {exp_vel:.4f}')
chk("HEAD_Accel ~ 0 (constant velocity)", abs(dm["HEAD_Accel"][T // 2]) < 1e-3)
chk("Dist_Hands invariant under translation", abs(dm["Dist_Hands"][T // 2] - dist(s, "L_WRIST", "R_WRIST")) < 1e-6)

print("\n" + "=" * 56 + f"\nRESULT: {PASS} passed, {FAIL} failed\n" + "=" * 56)
sys.exit(1 if FAIL else 0)
