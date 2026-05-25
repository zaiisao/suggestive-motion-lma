"""
Verify that every file listed in a manifest still has its recorded sha256.

Usage:
    python scripts/check_manifest_hashes.py data/manifest_paper_4way_2026-04-15_01-23.csv

Exits non-zero if any file is missing or its hash has drifted.
"""
import csv
import hashlib
import os
import sys


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for buf in iter(lambda: f.read(chunk), b""):
            h.update(buf)
    return h.hexdigest()


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    manifest_path = sys.argv[1]
    n = 0
    n_missing = 0
    n_mismatch = 0
    with open(manifest_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n += 1
            path = row["path"]
            expected = row["sha256"]
            if not os.path.exists(path):
                n_missing += 1
                print(f"  MISSING  {path}", file=sys.stderr)
                continue
            actual = sha256_file(path)
            if actual != expected:
                n_mismatch += 1
                print(f"  MISMATCH {path}\n    expected={expected}\n    actual  ={actual}",
                      file=sys.stderr)
    print(f"\n[+] Checked {n} files  missing={n_missing}  mismatch={n_mismatch}")
    sys.exit(0 if (n_missing == 0 and n_mismatch == 0) else 1)


if __name__ == "__main__":
    main()
