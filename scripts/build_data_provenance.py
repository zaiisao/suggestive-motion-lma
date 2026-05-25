"""
Build per-tier per-video provenance metadata for the suggestive-motion-lma data.

Existing manifests (data/manifest_paper_*.csv) record every LMA feature file
used by the paper. Those go down to (tier, fragment) granularity. This script
adds the level above that — (tier, source video) — so we can document where
each video originally came from, even when it's a TikTok / YouTube URL.

Outputs:
  <archive>/metadata/tier0_provenance.csv
  <archive>/metadata/tier1_provenance.csv
  <archive>/metadata/tier1_hailmary_provenance.csv
  <archive>/metadata/tier2_provenance.csv
  <archive>/metadata/tier3_provenance.csv
  <archive>/metadata/all_provenance.csv   (union, with `tier_origin` column)

Each row: video_id, source_kind, original_url, source_video_local_path,
features_local_path, n_fragments, in_paper_4way, in_paper_3way, in_paper_binary.
"""
import csv
import glob
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ARCHIVE = Path("/disk1/jaehoon/suggestive-motion-lma-archive")
METADATA_DIR = ARCHIVE / "metadata"

# Source-of-truth dirs (matches what the paper manifests reference)
T0_DIR = "/home/sogang/jaehoon/tier0"
T1_DIR = "/home/sogang/jaehoon/tier1"
T1_HAILMARY_DIR = "/home/sogang/jaehoon/KineGuard/output/tier1_features_hailmary"
T2_DIR = "/home/sogang/jaehoon/KineGuard/output/tier2_features"
T3_DIR = "/disk3/jaehoon/nsfw_datasets/tier3/NPDI_features_v2"

# Search roots for T2 source videos
T2_VIDEO_SEARCH_ROOTS = [
    "/disk1/kineguard_recon_yt",
    "/disk1/jaehoon/kineguard_recon_yt",
    "/home/sogang/jaehoon/KineGuard/kineguard_recon_tiktok",
    "/disk3/jaehoon/kineguard_tier2_crawl",
]
T2_VIDEO_EXTS = (".mp4", ".webm", ".mkv", ".avi", ".mov", ".m4v")

MANIFESTS = {
    "4way":   REPO / "data" / "manifest_paper_4way_2026-04-15_01-23.csv",
    "3way":   REPO / "data" / "manifest_paper_3way_2026-04-14_19-16.csv",
    "binary": REPO / "data" / "manifest_paper_binary_2026-04-14_21-36.csv",
}


def load_manifest_paths(csv_path):
    """Return the set of absolute paths in a manifest."""
    paths = set()
    with open(csv_path) as f:
        next(f)
        for line in f:
            parts = line.split(",", 2)
            if len(parts) >= 2:
                paths.add(parts[1])
    return paths


def classify_t2_source(video_id):
    """TikTok IDs are ~18-19 digit numerics. YouTube IDs are 11 chars alphanum + - _."""
    if video_id.isdigit() and len(video_id) >= 17:
        return "tiktok"
    return "youtube"


def t2_original_url(video_id, kind):
    if kind == "youtube":
        return f"https://www.youtube.com/watch?v={video_id}"
    elif kind == "tiktok":
        return f"https://www.tiktok.com/video/{video_id}"
    return ""


def index_t2_source_videos():
    """Build {video_id: [paths]} for all T2 source video files found on disk."""
    index = defaultdict(list)
    for root in T2_VIDEO_SEARCH_ROOTS:
        if not os.path.isdir(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                ext = os.path.splitext(fn)[1].lower()
                if ext in T2_VIDEO_EXTS:
                    base = os.path.splitext(fn)[0]
                    index[base].append(os.path.join(dirpath, fn))
    return index


# Kinetics-700 source root (where the raw videos live, used for T0/T1)
KINETICS_TRAIN_ROOT = "/disk4/kong/kinetics-dataset/k700-2020/train"

# Kinetics dir name = "<youtube_id>_<start_6digit>_<end_6digit>", e.g. "0bdVrgImymc_000020_000030"
KINETICS_DIRNAME_RE = re.compile(r"^(?P<yid>[A-Za-z0-9_\-]{11})_(?P<start>\d{6})_(?P<end>\d{6})$")


def kinetics_dir_to_url(dirname):
    """Convert a Kinetics dir name to its source YouTube URL with time range."""
    m = KINETICS_DIRNAME_RE.match(dirname)
    if not m:
        return None, None, None, None
    yid = m.group("yid")
    start = int(m.group("start"))
    end = int(m.group("end"))
    url = f"https://www.youtube.com/watch?v={yid}&t={start}s"
    return yid, start, end, url


def kinetics_local_path(klass, dirname):
    """Path to source mp4 in /disk4 Kinetics archive."""
    return os.path.join(KINETICS_TRAIN_ROOT, klass, dirname + ".mp4")


# ---- builders ---------------------------------------------------------------

def build_kinetics_tier(tier_label, tier_root, paper_paths, out_csv):
    """T0 / T1 / T1-hailmary layout: <tier_root>/<class>/<dirname>/[_clip_seg<N>/]lma_features_id*.npy.
    T0/T1 store features flat under <dirname>; T1-hailmary uses the YOLO-filtered
    _clip_seg<N> layout (same as T2). We accept both."""
    rows = []
    if not os.path.isdir(tier_root):
        print(f"[!] tier dir not present: {tier_root}", file=sys.stderr)
        return
    for klass in sorted(os.listdir(tier_root)):
        kpath = os.path.join(tier_root, klass)
        if not os.path.isdir(kpath):
            continue
        for dirname in sorted(os.listdir(kpath)):
            dpath = os.path.join(kpath, dirname)
            if not os.path.isdir(dpath):
                continue
            # First try flat, then YOLO-filtered _clip_seg layout
            features = sorted(glob.glob(os.path.join(dpath, "lma_features_id*.npy")))
            if not features:
                features = sorted(glob.glob(os.path.join(dpath, "_clip_seg*", "lma_features_id*.npy")))
            if not features:
                continue
            yid, t_start, t_end, url = kinetics_dir_to_url(dirname)
            in_runs = {tag: any(p in paper_paths[tag] for p in features)
                       for tag in paper_paths}
            rows.append({
                "tier_origin": tier_label,
                "video_id": yid or dirname,
                "source_kind": "youtube_kinetics700",
                "original_url": url or "",
                "kinetics_class": klass,
                "time_start_s": t_start if t_start is not None else "",
                "time_end_s":   t_end   if t_end   is not None else "",
                "source_video_local_path": kinetics_local_path(klass, dirname),
                "features_local_path": dpath,
                "n_fragments": len(features),
                "in_paper_4way":   in_runs.get("4way", False),
                "in_paper_3way":   in_runs.get("3way", False),
                "in_paper_binary": in_runs.get("binary", False),
            })
    write_csv(out_csv, rows)
    print(f"[+] {tier_label}: {len(rows)} videos -> {out_csv}")
    return rows


def build_t1_hailmary(paper_paths, out_csv):
    """T1 hailmary is class-organized like T0/T1 — reuse the kinetics builder
    but tag rows as tier1_hailmary so we can distinguish from base tier1."""
    rows = build_kinetics_tier("tier1_hailmary", T1_HAILMARY_DIR, paper_paths, out_csv)
    return rows


def build_t2(paper_paths, out_csv):
    rows = []
    video_index = index_t2_source_videos()
    for video_id in sorted(os.listdir(T2_DIR)):
        dpath = os.path.join(T2_DIR, video_id)
        if not os.path.isdir(dpath):
            continue
        # T2 has _clip_segN subdirs under each video dir
        features = sorted(glob.glob(os.path.join(dpath, "_clip_seg*", "lma_features_id*.npy")))
        if not features:
            continue
        kind = classify_t2_source(video_id)
        url = t2_original_url(video_id, kind)
        src_paths = video_index.get(video_id, [])
        in_runs = {tag: any(p in paper_paths[tag] for p in features) for tag in paper_paths}
        # Pull segment ranges out of filter_meta.json if it's there
        seg_info = ""
        fm = os.path.join(dpath, "filter_meta.json")
        if os.path.isfile(fm):
            try:
                with open(fm) as f:
                    j = json.load(f)
                segs = j.get("segments", [])
                seg_info = ";".join(f"{s['start']:.2f}-{s['end']:.2f}" for s in segs)
            except Exception:
                pass
        rows.append({
            "tier_origin": "tier2",
            "video_id": video_id,
            "source_kind": kind,
            "original_url": url,
            "kinetics_class": "",
            "time_start_s": "",
            "time_end_s": "",
            "source_video_local_path": src_paths[0] if src_paths else "",
            "features_local_path": dpath,
            "n_fragments": len(features),
            "yolo_filter_segments_s": seg_info,
            "in_paper_4way":   in_runs.get("4way", False),
            "in_paper_3way":   in_runs.get("3way", False),
            "in_paper_binary": in_runs.get("binary", False),
        })
    write_csv(out_csv, rows)
    print(f"[+] tier2: {len(rows)} videos -> {out_csv}")
    return rows


def build_t3(paper_paths, out_csv):
    """T3 layout: <T3_DIR>/<filename_stem>/lma_features_id*.npy (one dir per source mp4)."""
    rows = []
    if not os.path.isdir(T3_DIR):
        print(f"[!] dir not present: {T3_DIR}", file=sys.stderr)
        return
    for dirname in sorted(os.listdir(T3_DIR)):
        dpath = os.path.join(T3_DIR, dirname)
        if not os.path.isdir(dpath):
            continue
        features = sorted(glob.glob(os.path.join(dpath, "lma_features_id*.npy")))
        if not features:
            # Some entries may have features one level deeper
            features = sorted(glob.glob(os.path.join(dpath, "**", "lma_features_id*.npy"), recursive=True))
            if not features:
                continue
        in_runs = {tag: any(p in paper_paths[tag] for p in features) for tag in paper_paths}
        # NPDI source video location guess: corpus root, same filename + .mp4
        candidate = f"/disk3/jaehoon/nsfw_datasets/tier3/NPDI/{dirname}.mp4"
        src = candidate if os.path.isfile(candidate) else ""
        rows.append({
            "tier_origin": "tier3",
            "video_id": dirname,
            "source_kind": "NPDI_corpus",
            "original_url": "",  # NPDI is not URL-addressable; cite the corpus paper
            "kinetics_class": "",
            "time_start_s": "",
            "time_end_s": "",
            "source_video_local_path": src,
            "features_local_path": dpath,
            "n_fragments": len(features),
            "in_paper_4way":   in_runs.get("4way", False),
            "in_paper_3way":   in_runs.get("3way", False),
            "in_paper_binary": in_runs.get("binary", False),
        })
    write_csv(out_csv, rows)
    print(f"[+] tier3: {len(rows)} videos -> {out_csv}")
    return rows


def write_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = list(rows[0].keys())
    # Union of keys across rows (T2 has yolo_filter_segments_s; others don't)
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    os.makedirs(METADATA_DIR, exist_ok=True)
    paper_paths = {tag: load_manifest_paths(p) for tag, p in MANIFESTS.items()}
    for tag, paths in paper_paths.items():
        print(f"[*] manifest {tag}: {len(paths)} paths")

    t0 = build_kinetics_tier("tier0", T0_DIR, paper_paths,
                             str(METADATA_DIR / "tier0_provenance.csv"))
    t1 = build_kinetics_tier("tier1", T1_DIR, paper_paths,
                             str(METADATA_DIR / "tier1_provenance.csv"))
    t1hm = build_t1_hailmary(paper_paths,
                             str(METADATA_DIR / "tier1_hailmary_provenance.csv"))
    t2 = build_t2(paper_paths, str(METADATA_DIR / "tier2_provenance.csv"))
    t3 = build_t3(paper_paths, str(METADATA_DIR / "tier3_provenance.csv"))

    # Union CSV
    all_rows = []
    for batch in (t0, t1, t1hm, t2, t3):
        if batch:
            all_rows.extend(batch)
    if all_rows:
        write_csv(str(METADATA_DIR / "all_provenance.csv"), all_rows)
        print(f"[+] all_provenance.csv: {len(all_rows)} rows total")


if __name__ == "__main__":
    main()
