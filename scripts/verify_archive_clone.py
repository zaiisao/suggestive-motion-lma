"""
Cross-check the /disk1/jaehoon/suggestive-motion-lma-archive clone against
the original source dirs.

Checks per tier:
  - file count match
  - total bytes match
  - sha256 match for a random sample of N files (default 50)
  - every file listed in the shipped paper manifests resolves to an
    archive path with the recorded size

Exits non-zero on any mismatch.
"""
import csv
import hashlib
import os
import random
import sys

ARCHIVE = "/disk1/jaehoon/suggestive-motion-lma-archive"
REPO = "/home/sogang/jaehoon/suggestive-motion-lma"

PAIRS = [
    ("/home/sogang/jaehoon/tier0",                                     f"{ARCHIVE}/tier0_features"),
    ("/home/sogang/jaehoon/tier1",                                     f"{ARCHIVE}/tier1_features"),
    ("/home/sogang/jaehoon/KineGuard/output/tier1_features_hailmary", f"{ARCHIVE}/tier1_features_hailmary"),
    ("/home/sogang/jaehoon/KineGuard/output/tier2_features",          f"{ARCHIVE}/tier2_features"),
    ("/disk3/jaehoon/nsfw_datasets/tier3/NPDI_features_v2",           f"{ARCHIVE}/tier3_features"),
]

SAMPLE_SIZE = int(os.environ.get("VERIFY_SAMPLE", "50"))
RNG = random.Random(42)


def walk_files(root):
    out = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            out.append(os.path.join(dirpath, fn))
    return out


def sha256(path, block=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def check_pair(src, dst):
    print(f"\n[*] {src} -> {dst}")
    if not os.path.isdir(src):
        print(f"  [-] source dir not present, skipping")
        return True
    if not os.path.isdir(dst):
        print(f"  [!] archive dir missing")
        return False
    src_files = walk_files(src)
    dst_files = walk_files(dst)
    if len(src_files) != len(dst_files):
        print(f"  [!] file count mismatch: src={len(src_files)} dst={len(dst_files)}")
        return False
    src_bytes = sum(os.path.getsize(p) for p in src_files)
    dst_bytes = sum(os.path.getsize(p) for p in dst_files)
    if src_bytes != dst_bytes:
        print(f"  [!] byte total mismatch: src={src_bytes} dst={dst_bytes}")
        return False
    print(f"  [+] file count: {len(src_files)}, bytes: {src_bytes:,} (match)")

    # Random sample sha256 comparison
    sample = RNG.sample(src_files, min(SAMPLE_SIZE, len(src_files)))
    mismatches = 0
    for sp in sample:
        rel = os.path.relpath(sp, src)
        dp = os.path.join(dst, rel)
        if not os.path.isfile(dp):
            print(f"  [!] missing in archive: {rel}")
            mismatches += 1
            continue
        if sha256(sp) != sha256(dp):
            print(f"  [!] sha256 mismatch: {rel}")
            mismatches += 1
    if mismatches:
        print(f"  [!] sha256 sample: {mismatches}/{len(sample)} mismatches")
        return False
    print(f"  [+] sha256 sample of {len(sample)} files all match")
    return True


def check_paper_manifests():
    print("\n[*] Checking paper manifests resolve into the archive ...")
    ok = True
    for tag in ("4way", "3way", "binary"):
        candidates = [f for f in os.listdir(os.path.join(REPO, "data"))
                      if f.startswith(f"manifest_paper_{tag}_")]
        if not candidates:
            print(f"  [-] no manifest for {tag}")
            continue
        manifest = os.path.join(REPO, "data", candidates[0])
        n = 0
        n_missing = 0
        n_size_mismatch = 0
        with open(manifest) as f:
            reader = csv.DictReader(f)
            for row in reader:
                n += 1
                src_path = row["path"]
                size_recorded = int(row["size_bytes"])
                # Rewrite source path to archive equivalent
                arch_path = src_path
                for spath, apath in PAIRS:
                    if src_path.startswith(spath + "/"):
                        arch_path = apath + src_path[len(spath):]
                        break
                if not os.path.isfile(arch_path):
                    n_missing += 1
                    if n_missing <= 3:
                        print(f"  [!] not in archive: {arch_path}")
                elif os.path.getsize(arch_path) != size_recorded:
                    n_size_mismatch += 1
        status = "OK" if (n_missing == 0 and n_size_mismatch == 0) else "FAIL"
        print(f"  {tag:8s}: {n} entries, missing={n_missing}, size_mismatch={n_size_mismatch}  [{status}]")
        ok = ok and (n_missing == 0 and n_size_mismatch == 0)
    return ok


def main():
    ok = True
    for s, d in PAIRS:
        ok = check_pair(s, d) and ok
    ok = check_paper_manifests() and ok
    print(f"\n[{'+' if ok else '!'}] Overall: {'OK' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
