"""
Build a per-tier manifest of LMA feature files used to produce a given run of
analyze_lma_tiers.py — typically, the paper run.

Filters files by mtime <= --cutoff and records (path, mtime, size, sha256) so
the input set is auditable and reproducible. Output is a CSV with columns:

    tier,path,mtime_utc,size_bytes,sha256

Usage:
    python scripts/build_manifest.py \\
        --tier0-dirs /path/to/tier0_features \\
        --tier1-dirs /path/to/tier1a /path/to/tier1b \\
        --tier2-dirs /path/to/tier2_features \\
        --tier3-dirs /path/to/tier3_features \\
        --cutoff '2026-04-15 01:23:30' \\
        --out data/manifest_paper.csv
"""
import argparse
import csv
import datetime as dt
import glob
import hashlib
import os
import sys


def parse_cutoff(s):
    return dt.datetime.fromisoformat(s).timestamp()


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for buf in iter(lambda: f.read(chunk), b""):
            h.update(buf)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    for t in (0, 1, 2, 3):
        ap.add_argument(f"--tier{t}-dirs", nargs="+", required=True,
                        help=f"Directories holding Tier {t} lma_features_*.npy")
    ap.add_argument("--cutoff", default=None,
                    help="ISO timestamp (YYYY-MM-DD HH:MM:SS). Keep only files "
                         "with mtime <= cutoff. Omit to keep all.")
    ap.add_argument("--out", required=True, help="CSV path")
    args = ap.parse_args()

    cutoff_ts = parse_cutoff(args.cutoff) if args.cutoff else None
    if cutoff_ts:
        print(f"[*] Cutoff: {args.cutoff} (ts={cutoff_ts})", file=sys.stderr)
    else:
        print("[*] No cutoff — keeping all files", file=sys.stderr)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    n_total = 0
    n_kept = 0
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tier", "path", "mtime_utc", "size_bytes", "sha256"])
        for tier in (0, 1, 2, 3):
            dirs = getattr(args, f"tier{tier}_dirs")
            files = []
            for d in dirs:
                files.extend(glob.glob(os.path.join(d, "**", "lma_features_*.npy"),
                                       recursive=True))
            files = sorted(set(files))
            n_total += len(files)
            kept_this_tier = 0
            for p in files:
                st = os.stat(p)
                if cutoff_ts is not None and st.st_mtime > cutoff_ts:
                    continue
                mtime_iso = dt.datetime.fromtimestamp(st.st_mtime, dt.timezone.utc).isoformat()
                w.writerow([tier, p, mtime_iso, st.st_size, sha256_file(p)])
                kept_this_tier += 1
                n_kept += 1
            print(f"  Tier {tier}: {len(files):>6} found, {kept_this_tier:>6} kept",
                  file=sys.stderr)
    print(f"[+] Wrote {n_kept} / {n_total} files to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
