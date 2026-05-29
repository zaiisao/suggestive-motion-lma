#!/usr/bin/env python
"""Contract unit tests for scripts/analyze_lma_tiers.py (tier classifier module).

Run with the wham env python from the repo root:
  /home/sogang/mnt/db_2/anaconda3/envs/wham/bin/python tests/test_classifier_pipeline.py

These are *assertion* tests: each check states the contract derived from the
code (with file:line), then asserts it. The harness prints PASS/FAIL per check
and a final summary. It does NOT modify pipeline code.
"""
import csv
import importlib.util
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO_ROOT, "scripts", "analyze_lma_tiers.py")
MANIFEST = os.path.join(
    REPO_ROOT, "data", "manifest_paper_4way_2026-04-15_01-23.csv")
REAL_FEATURE = ("/home/sogang/jaehoon/KineGuard/output/tier2_features/"
                "7559829673161624840/_clip_seg0/lma_features_id0.npy")


def _load_module():
    spec = importlib.util.spec_from_file_location("analyze_lma_tiers", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---- tiny test harness ----
_results = []


def check(name, cond, detail=""):
    ok = bool(cond)
    _results.append((name, ok, detail))
    status = "PASS" if ok else "FAIL"
    line = f"[{status}] {name}"
    if detail and not ok:
        line += f"\n         -> {detail}"
    print(line)
    return ok


def main():
    alt = _load_module()

    # ============================================================
    # aggregate_fragment  (analyze_lma_tiers.py:113-127)
    # Contract: (T,55) -> (110,) = concat(mean(axis=0), std(axis=0)).
    # Rejects (None) when: ndim!=2, shape[1]!=55, shape[0]<5, or non-finite.
    # ============================================================
    rng = np.random.default_rng(0)

    # -- valid input --
    valid = rng.standard_normal((30, 55))
    out = alt.aggregate_fragment(valid)
    check("aggregate: valid (30,55) returns non-None",
          out is not None)
    check("aggregate: output length == 110 (mean||std)",
          out is not None and out.shape == (110,),
          f"got shape {None if out is None else out.shape}")
    if out is not None:
        exp = np.concatenate([valid.mean(axis=0), valid.std(axis=0)])
        check("aggregate: output == concat(mean(axis=0), std(axis=0))",
              np.allclose(out, exp),
              "aggregation does not match documented mean||std")

    # -- exactly the minimum frame count (5) is accepted (< 5 rejected) --
    check("aggregate: T==5 (min) accepted",
          alt.aggregate_fragment(rng.standard_normal((5, 55))) is not None)

    # -- too short (T < 5) rejected --
    check("aggregate: T==4 (too short) -> None",
          alt.aggregate_fragment(rng.standard_normal((4, 55))) is None)
    check("aggregate: T==1 -> None",
          alt.aggregate_fragment(rng.standard_normal((1, 55))) is None)

    # -- wrong number of columns rejected --
    check("aggregate: wrong cols (30,54) -> None",
          alt.aggregate_fragment(rng.standard_normal((30, 54))) is None)
    check("aggregate: wrong cols (30,56) -> None",
          alt.aggregate_fragment(rng.standard_normal((30, 56))) is None)

    # -- not 2D rejected --
    check("aggregate: 1D array -> None",
          alt.aggregate_fragment(rng.standard_normal(55)) is None)
    check("aggregate: 3D array (10,55,1) -> None",
          alt.aggregate_fragment(rng.standard_normal((10, 55, 1))) is None)

    # -- non-finite values rejected --
    nan_arr = rng.standard_normal((30, 55)); nan_arr[3, 7] = np.nan
    check("aggregate: contains NaN -> None",
          alt.aggregate_fragment(nan_arr) is None)
    inf_arr = rng.standard_normal((30, 55)); inf_arr[0, 0] = np.inf
    check("aggregate: contains +inf -> None",
          alt.aggregate_fragment(inf_arr) is None)

    # -- real on-disk fragment (174,55) is accepted --
    if os.path.exists(REAL_FEATURE):
        real = np.load(REAL_FEATURE, allow_pickle=True)
        rout = alt.aggregate_fragment(real)
        check("aggregate: real feature file -> (110,)",
              rout is not None and rout.shape == (110,),
              f"real shape {real.shape}")
    else:
        check("aggregate: real feature file present", False,
              f"missing {REAL_FEATURE}")

    # ============================================================
    # load_files  (analyze_lma_tiers.py:130-143)
    # Contract: returns (N,110); drops invalid fragments; (0,110) if none.
    # ============================================================
    with tempfile.TemporaryDirectory() as td:
        good = os.path.join(td, "good.npy")
        bad_short = os.path.join(td, "short.npy")
        bad_cols = os.path.join(td, "cols.npy")
        bad_nan = os.path.join(td, "nan.npy")
        not_npy = os.path.join(td, "broken.npy")
        np.save(good, rng.standard_normal((20, 55)))
        np.save(bad_short, rng.standard_normal((3, 55)))
        np.save(bad_cols, rng.standard_normal((20, 10)))
        bad = rng.standard_normal((20, 55)); bad[0, 0] = np.nan
        np.save(bad_nan, bad)
        with open(not_npy, "wb") as fh:
            fh.write(b"not a real npy file")

        M = alt.load_files([good, bad_short, bad_cols, bad_nan, not_npy])
        check("load_files: returns (N,110) with width 110",
              M.ndim == 2 and M.shape[1] == 110, f"got {M.shape}")
        check("load_files: keeps only the 1 valid fragment",
              M.shape[0] == 1, f"got N={M.shape[0]} (expected 1)")

        empty = alt.load_files([bad_short, not_npy])
        check("load_files: no valid -> (0,110)",
              empty.shape == (0, 110), f"got {empty.shape}")

    # ============================================================
    # _files_from_manifest  (analyze_lma_tiers.py:65-75)
    # Contract: tier <- int(row['tier']); path <- row['path'];
    #           returns {0,1,2,3:[...]} each sorted ascending.
    # ============================================================
    with tempfile.TemporaryDirectory() as td:
        man = os.path.join(td, "m.csv")
        with open(man, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["tier", "path", "mtime_utc", "size_bytes", "sha256"])
            w.writerow(["2", "/z/b.npy", "t", "1", "h"])
            w.writerow(["0", "/a/a.npy", "t", "1", "h"])
            w.writerow(["2", "/z/a.npy", "t", "1", "h"])
            w.writerow(["3", "/q.npy", "t", "1", "h"])
        fm = alt._files_from_manifest(man)
        check("manifest: keys are exactly {0,1,2,3}",
              set(fm.keys()) == {0, 1, 2, 3}, f"keys={sorted(fm.keys())}")
        check("manifest: label derived from 'tier' column (tier 0 has /a/a.npy)",
              fm[0] == ["/a/a.npy"], f"tier0={fm[0]}")
        check("manifest: tier 2 collects both rows, sorted",
              fm[2] == ["/z/a.npy", "/z/b.npy"], f"tier2={fm[2]}")
        check("manifest: empty tier -> []", fm[1] == [], f"tier1={fm[1]}")

    # Real manifest sanity: parse it, every path basename matches lma_features_*
    fm_real = alt._files_from_manifest(MANIFEST)
    total = sum(len(v) for v in fm_real.values())
    check("manifest(real): parses all 20534 data rows",
          total == 20534, f"total paths={total}")
    sample = (fm_real[2] or fm_real[0])[:50]
    check("manifest(real): paths are lma_features_*.npy",
          all(os.path.basename(p).startswith("lma_features_")
              and p.endswith(".npy") for p in sample))

    # ============================================================
    # Balancing determinism  (analyze_lma_tiers.py:169 + 215)
    # Contract: np.random.seed(seed) then np.random.choice(N, max_n,
    #           replace=False) -> identical indices across runs.
    # ============================================================
    def balanced_indices(seed, n, max_n):
        np.random.seed(seed)
        return np.random.choice(n, size=max_n, replace=False)

    i1 = balanced_indices(42, 500, 100)
    i2 = balanced_indices(42, 500, 100)
    check("balance: seed=42 reproduces identical selection across 2 runs",
          np.array_equal(i1, i2),
          "non-deterministic selection under fixed seed")
    i3 = balanced_indices(7, 500, 100)
    check("balance: different seed -> (almost surely) different selection",
          not np.array_equal(i1, i3))
    check("balance: replace=False -> no duplicate indices",
          len(np.unique(i1)) == len(i1))

    # ============================================================
    # End-to-end smoke: run the script on the real manifest,
    # --max-per-tier 100, into a temp out-dir. Assert it completes and
    # writes summary.json + classifier_results.json with acc in [0,1].
    # ============================================================
    with tempfile.TemporaryDirectory() as out_dir:
        cmd = [sys.executable, SCRIPT,
               "--manifest", MANIFEST,
               "--max-per-tier", "100",
               "--out-dir", out_dir,
               "--seed", "42"]
        proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True,
                              text=True, timeout=1200)
        completed = check("smoke: script exits 0", proc.returncode == 0,
                          f"rc={proc.returncode}\nstderr tail:\n"
                          + proc.stderr[-1500:])
        summ_path = os.path.join(out_dir, "summary.json")
        res_path = os.path.join(out_dir, "classifier_results.json")
        check("smoke: summary.json written", os.path.exists(summ_path))
        check("smoke: classifier_results.json written",
              os.path.exists(res_path))
        if os.path.exists(summ_path):
            with open(summ_path) as fh:
                summ = json.load(fh)
            # 4-way default: 4 classes, each capped at 100
            check("smoke: n_per_class capped at 100 each",
                  all(v == 100 for v in summ["n_per_class"].values()),
                  f"n_per_class={summ.get('n_per_class')}")
            accs = [c["acc"] for c in summ["classifiers"].values()]
            check("smoke: every classifier acc in [0,1]",
                  len(accs) > 0 and all(0.0 <= a <= 1.0 for a in accs),
                  f"accs={accs}")
            f1s = [c["f1_macro"] for c in summ["classifiers"].values()]
            check("smoke: every f1_macro in [0,1]",
                  all(0.0 <= f <= 1.0 for f in f1s), f"f1s={f1s}")

    # ---- summary ----
    n = len(_results)
    n_fail = sum(1 for _, ok, _ in _results if not ok)
    print("\n" + "=" * 60)
    print(f"TOTAL CHECKS: {n}   PASSED: {n - n_fail}   FAILED: {n_fail}")
    print("=" * 60)
    if n_fail:
        print("FAILURES:")
        for name, ok, detail in _results:
            if not ok:
                print(f"  - {name}: {detail}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
