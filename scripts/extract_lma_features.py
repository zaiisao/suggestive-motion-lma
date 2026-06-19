#!/usr/bin/env python3
"""Compute the paper's 61-feature LMA descriptors from WHAM fragments.

This is a thin driver over the **data-producing extractor** in the companion
submodule (``external/dance-style-recognition``). It does only the WHAM-specific
input preparation — regress the SMPL-24 skeleton from the posed mesh in the camera
frame and take the per-frame convex-hull body volume — and hands those to the
submodule's ``compute_lma_descriptor``, which is exactly the code that generated the
shipped ``reproduce/lma_data/features.npz``. Run on the same WHAM fragments, this driver
reproduces that file's **per-feature values** bit-for-bit — every column matches by name to
float32-exact (``max|Δ| = 0``, ``np.array_equal`` after aligning columns by ``feature_names``),
short clips included. The shipped file's fixed column order, its opaque ``sha1`` keys, and the
anonymized ``source_video`` grouping id are applied by the release-packaging step, not emitted
here (so a *naive* whole-array ``np.array_equal`` against the shipped file is not expected to
pass — only the name-aligned per-feature values are bit-identical). The LMA feature math is NOT
reimplemented here; it lives in ``external/dance-style-recognition/src/utils/lma_extractor.py``.

For each WHAM fragment ``.npz``:
  1. ``joints = J_regressor @ verts`` — SMPL-24 skeleton in the camera frame (no
     ``trans_world``); ``volume[t] = ConvexHull(verts[t]).volume`` — the mesh body volume;
  2. ``compute_lma_descriptor(joints, volumes, floors, fps)`` -> ``(T, 61)`` per-frame
     descriptor (the 55 of Turab et al. + six inter-joint angles, causal 55-frame window,
     short-window=5; no minimum-length guard, so short clips are handled);
  3. collapse to a 122-d vector by per-feature mean and standard deviation.

The turn-key reproduction of every paper number needs none of this — the precomputed
descriptors ship in ``reproduce/lma_data/`` and are consumed by
``reproduce/reproduce_paper.ipynb``. This is the auditable path from raw WHAM output.

Usage:
    python scripts/extract_lma_features.py --wham-dir output/wham --out output/lma
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import glob
import pickle
import sys
from multiprocessing import Pool

import numpy as np


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "external", "dance-style-recognition", "src")
DEFAULT_SMPL = os.path.join(
    REPO, "external", "WHAM", "dataset", "body_models", "smpl", "SMPL_NEUTRAL.pkl"
)
SMPL_PATH = os.environ.get("LMA_SMPL_MODEL_PATH", DEFAULT_SMPL)

sys.path.insert(0, SRC)
from lma_descriptor import compute_lma_descriptor, IdentityFloor, mesh_volume  # noqa: E402


def load_joint_regressor(smpl_path=SMPL_PATH):
    """The 24x6890 SMPL joint regressor (first 24 rows = the SMPL-24 skeleton)."""
    with open(smpl_path, "rb") as f:
        regressor = pickle.load(f, encoding="latin1")["J_regressor"]
    regressor = regressor.toarray() if hasattr(regressor, "toarray") else np.asarray(regressor)
    return regressor[:24].astype(np.float32)


def fragment_descriptor(verts, fps, joint_regressor):
    """A WHAM fragment's posed vertices -> (T, 61) per-frame LMA descriptor (sorted-key cols)."""
    verts = verts.astype(np.float32)
    joints = np.einsum("jk,tkc->tjc", joint_regressor, verts)            # (T,24,3) camera frame
    volumes = mesh_volume(verts)                                          # Shape feature (single source)
    floors = [IdentityFloor()] * len(joints)
    _, matrix = compute_lma_descriptor([joints[t] for t in range(len(joints))], volumes, floors, fps)
    return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)        # (T, 61)


def summary_vector(matrix):
    """(T, 61) per-frame matrix -> (122,) per-fragment vector = [mean(61), std(61)].

    The per-frame matrix is cast to float32 BEFORE the mean/std so the summary is computed
    in the same (float32) precision the shipped features.npz is stored in. This makes the
    per-feature *values* bit-for-bit identical to features.npz (rather than agreeing only to
    float32 round-off, which is what summarizing in float64 and casting at the end produces).
    The column order and keys here are this script's own; the shipped file's layout (column
    order, sha1 keys, anonymized source_video) is applied by the release-packaging step."""
    m = matrix.astype(np.float32)
    return np.concatenate([m.mean(0), m.std(0)])


def feature_names():
    """The 122 column names ([k_mean for k in sorted(61)] + [k_std ...]), matching summary_vector."""
    keys, _ = compute_lma_descriptor([np.zeros((24, 3))] * 60, np.zeros(60),
                                     [IdentityFloor()] * 60, 30.0)
    keys = sorted(keys)
    return [f"{k}_mean" for k in keys] + [f"{k}_std" for k in keys]


_REGRESSOR = None


def _init():
    global _REGRESSOR
    _REGRESSOR = load_joint_regressor()


def _work(npz_path):
    out_path = npz_path[:-4] + ".lma.npy"
    if os.path.exists(out_path):
        return ("skip", npz_path)
    try:
        with np.load(npz_path, allow_pickle=True) as data:
            if "verts" not in data.files:
                return ("no_verts", npz_path)
            verts = data["verts"]
            fps = float(data["fps"]) if "fps" in data.files else 30.0
        matrix = fragment_descriptor(verts, fps, _REGRESSOR)
        np.save(out_path, matrix)
        return ("ok", out_path, matrix.shape[0], matrix.shape[1])
    except Exception as exc:  # one bad fragment must not abort the batch
        return ("error", npz_path, f"{type(exc).__name__}: {exc}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wham-dir", required=True,
                    help="directory of WHAM fragment .npz files (searched recursively)")
    ap.add_argument("--out", required=True, help="output directory for the .npz feature bundle")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    fragments = sorted(glob.glob(os.path.join(args.wham_dir, "**", "*.npz"), recursive=True))
    if not fragments:
        sys.exit(f"no .npz fragments under {args.wham_dir}")
    print(f"[+] {len(fragments)} fragments  ->  {args.out}  (camera frame, 61-feature LMA extractor)")

    names = feature_names()
    keys, vectors, lengths, stats = [], [], [], {"ok": 0, "skip": 0, "no_verts": 0, "error": 0}
    runner = Pool(args.workers, initializer=_init) if args.workers > 1 else None
    results = (runner.imap_unordered(_work, fragments) if runner
               else (_init() or (_work(p) for p in fragments)))
    for res in results:
        stats[res[0]] += 1
        if res[0] == "ok":
            matrix = np.load(res[1])
            keys.append(os.path.relpath(res[1], args.out))
            vectors.append(summary_vector(matrix))
            lengths.append(matrix.shape[0])
        elif res[0] == "error":
            print(f"    ! {res[1]}: {res[2]}")
    if runner:
        runner.close(); runner.join()

    if vectors:
        np.savez_compressed(
            os.path.join(args.out, "features.npz"),
            keys=np.array(keys), X=np.stack(vectors),
            length=np.array(lengths, dtype=np.int32),
            feature_names=np.array(names),
        )
    print(f"[+] done: {stats}  ->  {os.path.join(args.out, 'features.npz')}")


if __name__ == "__main__":
    main()
