"""
Tier 1 batch: process N videos each from a list of artistic Kinetics-700
classes, sharded across GPUs.

Each GPU handles a subset of classes (e.g. 2 classes × 40 videos = 80 per shard).
Reads from a Kinetics-700 train split, writes features to
<out>/{class}/{video_id}/lma_features_id*.npy.

Usage:
    LMA_KINETICS_ROOT=/path/to/k700-2020/train \\
    python scripts/batch_tier1_kinetics.py --gpu-id 0 --classes breakdancing krumping
"""
import argparse
import glob
import os
import sys
import time
import json

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from scripts.batch_filtered_wham import (
    process_video, YOLO_MODEL_PATH, wait_for_cool,
)

KINETICS_ROOT = os.environ.get("LMA_KINETICS_ROOT", "data/kinetics-700/train")
OUT_ROOT_DEFAULT = os.environ.get("LMA_TIER1_OUTPUT_DIR", "output/tier1_features")
VIDEOS_PER_CLASS = 40


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-id", type=int, required=True)
    ap.add_argument("--classes", nargs="+", required=True,
                    help="Kinetics class names to process")
    ap.add_argument("--videos-per-class", type=int, default=VIDEOS_PER_CLASS)
    ap.add_argument("--output-dir", default=OUT_ROOT_DEFAULT)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    from ultralytics import YOLO
    model = YOLO(YOLO_MODEL_PATH)

    log_path = os.path.join(args.output_dir, f"gpu{args.gpu_id}_log.txt")

    def log(msg):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    log(f"Starting tier1 hailmary: GPU {args.gpu_id}, classes={args.classes}")

    videos = []
    for cls in args.classes:
        cls_dir = os.path.join(KINETICS_ROOT, cls)
        if not os.path.isdir(cls_dir):
            log(f"  [WARN] class dir not found: {cls_dir}")
            continue
        cls_videos = sorted(glob.glob(os.path.join(cls_dir, "*.mp4")))[:args.videos_per_class]
        log(f"  {cls}: {len(cls_videos)} videos")
        for v in cls_videos:
            videos.append((cls, v))

    log(f"Total videos: {len(videos)}")

    counts = {"processed": 0, "skipped": 0, "failed": 0, "filtered_out": 0}
    start_time = time.time()

    for i, (cls, video_path) in enumerate(videos):
        if i % 3 == 0:
            wait_for_cool(args.gpu_id, log=log)

        # Organize output by class
        cls_out = os.path.join(args.output_dir, cls)
        os.makedirs(cls_out, exist_ok=True)

        vid_id = os.path.splitext(os.path.basename(video_path))[0]
        elapsed = time.time() - start_time
        done = sum(counts.values())
        rate = done / (elapsed / 3600) if elapsed > 0 else 0
        remaining = (len(videos) - i) / rate if rate > 0 else 0
        log(f"[{i+1}/{len(videos)}] {cls}/{vid_id} (rate: {rate:.1f} vid/hr, ETA: {remaining:.1f} hr)")

        status, msg = process_video(video_path, cls_out, model, log=log)
        counts[status] += 1
        if status != "skipped":
            log(f"  -> {status}: {msg}")

    total_hours = (time.time() - start_time) / 3600
    log(f"\nBatch complete in {total_hours:.1f} hours:")
    for k, v in counts.items():
        log(f"  {k}: {v}")


if __name__ == "__main__":
    main()
