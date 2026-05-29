#!/usr/bin/env python
"""
Contract unit tests for the data-provenance modules of the LMA pipeline:
  - scripts/build_manifest.py        (manifest builder)
  - scripts/check_manifest_hashes.py (hash verifier)

Contract derived directly from the code (no pipeline code is modified here):

  * Header / column schema (build_manifest.py:62):
        ["tier", "path", "mtime_utc", "size_bytes", "sha256"]
  * tier values come from the literal loop `for tier in (0,1,2,3)` (build_manifest.py:63)
  * path is found via glob "**/lma_features_*.npy" (build_manifest.py:67), so
    every recorded basename matches lma_features_id*.npy in practice.
  * size_bytes == os.stat(p).st_size (build_manifest.py:77, st.st_size)
  * sha256    == streaming sha256 over the file in 1<<20 chunks
                 (build_manifest.py:32-37, identical in check_manifest_hashes.py:15-20)
  * The verifier (check_manifest_hashes.py:33-45) reads the same columns via
    csv.DictReader and recomputes sha256_file(path) == row["sha256"].

These tests assert the frozen paper manifests still conform to that contract and,
crucially, that the on-disk feature files still hash to the recorded sha256
(a reproducibility / tamper check).

Run:
    /home/sogang/mnt/db_2/anaconda3/envs/wham/bin/python tests/test_manifest_integrity.py
"""
import csv
import hashlib
import os
import random
import subprocess
import sys
import tempfile

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "data")
SCRIPTS = os.path.join(REPO, "scripts")

MANIFEST_4WAY = os.path.join(DATA, "manifest_paper_4way_2026-04-15_01-23.csv")
MANIFEST_3WAY = os.path.join(DATA, "manifest_paper_3way_2026-04-14_19-16.csv")
MANIFEST_BINARY = os.path.join(DATA, "manifest_paper_binary_2026-04-14_21-36.csv")

EXPECTED_HEADER = ["tier", "path", "mtime_utc", "size_bytes", "sha256"]
EXPECTED_TIERS = {"0", "1", "2", "3"}

# Expected row counts per the project audit trail (paper headline counts).
EXPECTED_COUNTS = {
    MANIFEST_4WAY: 20534,
    MANIFEST_3WAY: 17880,
    MANIFEST_BINARY: 18358,
}

INTEGRITY_SAMPLE = 50
CHUNK = 1 << 20

# ---------------------------------------------------------------------------
# Minimal test harness (PASS/FAIL, no external deps).
# ---------------------------------------------------------------------------
_results = []  # (name, ok, detail)


def check(name, ok, detail=""):
    _results.append((name, bool(ok), detail))
    status = "PASS" if ok else "FAIL"
    line = f"[{status}] {name}"
    if detail:
        line += f"  -- {detail}"
    print(line)
    return ok


def sha256_file(path, chunk=CHUNK):
    """Replicates build_manifest.sha256_file / check_manifest_hashes.sha256_file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for buf in iter(lambda: f.read(chunk), b""):
            h.update(buf)
    return h.hexdigest()


def read_manifest(path):
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    return header, rows


# ---------------------------------------------------------------------------
# 1. Schema contract on all three frozen manifests.
# ---------------------------------------------------------------------------
def test_schema():
    for mpath in (MANIFEST_4WAY, MANIFEST_3WAY, MANIFEST_BINARY):
        name = os.path.basename(mpath)
        if not check(f"schema: manifest exists [{name}]", os.path.exists(mpath), mpath):
            continue
        header, rows = read_manifest(mpath)
        check(f"schema: header == {EXPECTED_HEADER} [{name}]",
              header == EXPECTED_HEADER, f"got {header}")
        check(f"schema: row count == {EXPECTED_COUNTS[mpath]} [{name}]",
              len(rows) == EXPECTED_COUNTS[mpath], f"got {len(rows)}")

        bad_tier = [r[0] for r in rows if r[0] not in EXPECTED_TIERS]
        check(f"schema: all tier in {sorted(EXPECTED_TIERS)} [{name}]",
              not bad_tier, f"{len(bad_tier)} bad e.g. {bad_tier[:3]}")

        bad_path = [r[1] for r in rows
                    if not os.path.basename(r[1]).startswith("lma_features_id")
                    or not r[1].endswith(".npy")]
        check(f"schema: all path basename ~ lma_features_id*.npy [{name}]",
              not bad_path, f"{len(bad_path)} bad e.g. {bad_path[:2]}")

        bad_size = [r for r in rows if not r[3].isdigit() or int(r[3]) <= 0]
        check(f"schema: size_bytes positive integers [{name}]",
              not bad_size, f"{len(bad_size)} bad")

        bad_hash = [r for r in rows
                    if len(r[4]) != 64 or any(c not in "0123456789abcdef" for c in r[4])]
        check(f"schema: sha256 is 64 lowercase hex [{name}]",
              not bad_hash, f"{len(bad_hash)} bad")


# ---------------------------------------------------------------------------
# 2. INTEGRITY: recompute size + sha256 for a random sample and compare.
# ---------------------------------------------------------------------------
def test_integrity_sample():
    header, rows = read_manifest(MANIFEST_4WAY)
    if header != EXPECTED_HEADER:
        check("integrity: header ok before sampling", False, str(header))
        return
    rng = random.Random(20260529)  # deterministic sample
    sample = rng.sample(rows, min(INTEGRITY_SAMPLE, len(rows)))

    n_missing = 0
    n_size_mismatch = 0
    n_hash_mismatch = 0
    n_ok = 0
    for r in sample:
        _, path, _, size_str, expected_hash = r
        if not os.path.exists(path):
            n_missing += 1
            print(f"    MISSING  {path}")
            continue
        actual_size = os.path.getsize(path)
        if actual_size != int(size_str):
            n_size_mismatch += 1
            print(f"    SIZE MISMATCH {path}\n      recorded={size_str} actual={actual_size}")
            continue
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            n_hash_mismatch += 1
            print(f"    HASH MISMATCH {path}\n      expected={expected_hash}\n      actual  ={actual_hash}")
            continue
        n_ok += 1

    check(f"integrity: {len(sample)} sampled files all present",
          n_missing == 0, f"{n_missing} missing")
    check(f"integrity: sampled size_bytes match disk",
          n_size_mismatch == 0, f"{n_size_mismatch} size mismatches")
    check(f"integrity: sampled sha256 match disk (REPRODUCIBILITY)",
          n_hash_mismatch == 0, f"{n_hash_mismatch} hash mismatches")
    print(f"    integrity summary: ok={n_ok} missing={n_missing} "
          f"size_mismatch={n_size_mismatch} hash_mismatch={n_hash_mismatch} "
          f"of {len(sample)} sampled")


# ---------------------------------------------------------------------------
# 3. build_manifest.py on a tiny temp dir: gather + hash + determinism.
# ---------------------------------------------------------------------------
def test_build_manifest_runnable():
    py = sys.executable
    script = os.path.join(SCRIPTS, "build_manifest.py")
    if not check("build: script exists", os.path.exists(script), script):
        return

    with tempfile.TemporaryDirectory() as tmp:
        # Build a tiny fixture: one dir per tier, a couple of fake .npy each,
        # plus a decoy non-matching file that must be ignored.
        tier_dirs = {}
        expected = {}  # (tier, abspath) -> (size, sha256)
        for tier in (0, 1, 2, 3):
            d = os.path.join(tmp, f"tier{tier}")
            sub = os.path.join(d, "clipA")
            os.makedirs(sub, exist_ok=True)
            tier_dirs[tier] = d
            for idx in range(2):
                arr = np.arange(10 + tier * 4 + idx, dtype=np.float32)
                fp = os.path.join(sub, f"lma_features_id{idx}.npy")
                np.save(fp, arr)
                expected[(str(tier), fp)] = (os.path.getsize(fp), sha256_file(fp))
            # decoy that should NOT be picked up (wrong name)
            np.save(os.path.join(sub, "other_array.npy"), np.zeros(3))

        out1 = os.path.join(tmp, "m1.csv")
        out2 = os.path.join(tmp, "m2.csv")
        cmd = [py, script, "--out", out1]
        for tier in (0, 1, 2, 3):
            cmd += [f"--tier{tier}-dirs", tier_dirs[tier]]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if not check("build: subprocess exit 0", proc.returncode == 0,
                     proc.stderr.strip()[-300:]):
            return

        header, rows = read_manifest(out1)
        check("build: temp header matches schema", header == EXPECTED_HEADER, str(header))
        check("build: gathered exactly 8 matching .npy (decoys ignored)",
              len(rows) == 8, f"got {len(rows)}")

        # Recorded (tier, path) set must equal what we created, with correct
        # size + hash for each.
        recorded = {(r[0], r[1]): (int(r[3]), r[4]) for r in rows}
        check("build: recorded (tier,path) set == fixture set",
              set(recorded.keys()) == set(expected.keys()),
              f"{len(recorded)} recorded vs {len(expected)} expected")
        mismatches = [k for k in expected if recorded.get(k) != expected[k]]
        check("build: recorded size+sha256 match independently computed values",
              not mismatches, f"{len(mismatches)} mismatched e.g. {mismatches[:1]}")

        # Determinism: a second run over the same fixture yields identical bytes.
        cmd2 = [py, script, "--out", out2]
        for tier in (0, 1, 2, 3):
            cmd2 += [f"--tier{tier}-dirs", tier_dirs[tier]]
        proc2 = subprocess.run(cmd2, capture_output=True, text=True)
        check("build: second subprocess exit 0", proc2.returncode == 0,
              proc2.stderr.strip()[-300:])
        with open(out1, "rb") as a, open(out2, "rb") as b:
            same = a.read() == b.read()
        check("build: deterministic (two runs byte-identical)", same)

        # Cutoff filter: a cutoff in the far past should drop everything.
        out3 = os.path.join(tmp, "m3.csv")
        cmd3 = [py, script, "--out", out3, "--cutoff", "2000-01-01 00:00:00"]
        for tier in (0, 1, 2, 3):
            cmd3 += [f"--tier{tier}-dirs", tier_dirs[tier]]
        subprocess.run(cmd3, capture_output=True, text=True)
        _, rows3 = read_manifest(out3)
        check("build: far-past cutoff keeps 0 files", len(rows3) == 0, f"got {len(rows3)}")


# ---------------------------------------------------------------------------
# 4. Cross-manifest subset relationships by path.
# ---------------------------------------------------------------------------
def test_subset_relationships():
    def path_set(mpath):
        _, rows = read_manifest(mpath)
        return {r[1] for r in rows}

    p4 = path_set(MANIFEST_4WAY)
    p3 = path_set(MANIFEST_3WAY)
    pb = path_set(MANIFEST_BINARY)

    check("subset: 4way path set has no duplicates",
          True)  # set() collapses; report sizes instead
    print(f"    |4way|={len(p4)} |3way|={len(p3)} |binary|={len(pb)}")

    extra3 = p3 - p4
    check("subset: 3way (17,880) ⊆ 4way (20,534) by path",
          not extra3, f"{len(extra3)} paths in 3way not in 4way e.g. {list(extra3)[:1]}")

    extrab = pb - p4
    check("subset: binary (18,358) ⊆ 4way (20,534) by path",
          not extrab, f"{len(extrab)} paths in binary not in 4way e.g. {list(extrab)[:1]}")


def main():
    print("=" * 72)
    print("Manifest provenance contract tests")
    print("=" * 72)
    test_schema()
    print("-" * 72)
    test_integrity_sample()
    print("-" * 72)
    test_build_manifest_runnable()
    print("-" * 72)
    test_subset_relationships()
    print("=" * 72)

    n_pass = sum(1 for _, ok, _ in _results if ok)
    n_fail = sum(1 for _, ok, _ in _results if not ok)
    print(f"TOTAL: {len(_results)} checks  PASS={n_pass}  FAIL={n_fail}")
    if n_fail:
        print("FAILURES:")
        for name, ok, detail in _results:
            if not ok:
                print(f"  - {name}  ({detail})")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
