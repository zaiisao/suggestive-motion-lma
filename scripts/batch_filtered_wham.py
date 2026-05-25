#!/usr/bin/env python3
"""
Batch WHAM + LMA processing with YOLO pre-filtering.

For each video:
  1. YOLO11x-pose scans for continuous segments with a visible person
  2. Only those segments are extracted (ffmpeg clip) and passed to WHAM
  3. WHAM + LMA runs on the short, filtered clips

This avoids wasting GPU time on frames without visible humans (~77% of
pornographic content is filtered out on average).

Usage (in tmux):
    CUDA_VISIBLE_DEVICES=1 python scripts/batch_filtered_wham.py --start 0 --end 333 --gpu-id 1
    CUDA_VISIBLE_DEVICES=2 python scripts/batch_filtered_wham.py --start 333 --end 666 --gpu-id 2
    CUDA_VISIBLE_DEVICES=3 python scripts/batch_filtered_wham.py --start 666 --end 1000 --gpu-id 3
"""

import argparse
import os
import sys
import glob
import time
import json
import subprocess
import tempfile

import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

T3_DIR = os.environ.get("LMA_T3_DIR", "")
OUT_ROOT = os.environ.get("LMA_OUTPUT_DIR", "output/tier3_processing")
YOLO_MODEL_PATH = os.environ.get(
    "LMA_YOLO_MODEL_PATH", os.path.join(REPO, "yolo11x-pose.pt")
)

# Filter parameters
MIN_KEYPOINTS = 10
MIN_CONFIDENCE = 0.5
MIN_SEGMENT_SECONDS = 3.0
WHAM_TIMEOUT = 600  # 10 min max per segment


def get_gpu_temp(gpu_id):
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits",
             f"--id={gpu_id}"],
            capture_output=True, text=True, timeout=5,
        )
        return int(result.stdout.strip())
    except Exception:
        return 0


def wait_for_cool(gpu_id, threshold=82, target=75, log=print):
    temp = get_gpu_temp(gpu_id)
    if temp >= threshold:
        log(f"[THERMAL] GPU {gpu_id} at {temp}C — pausing...")
        while temp > target:
            time.sleep(60)
            temp = get_gpu_temp(gpu_id)
            log(f"[THERMAL] GPU {gpu_id} at {temp}C, waiting for {target}C...")
        log(f"[THERMAL] Cooled to {temp}C — resuming")


def yolo_filter(video_path, model, min_kpts=10, min_conf=0.5, min_seconds=3.0):
    """
    Run YOLO pose detection to find continuous segments with visible humans.
    Returns list of (start_sec, end_sec, duration_sec) tuples.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    required_frames = int(min_seconds * fps)
    results = model.predict(source=video_path, stream=True, conf=0.25, verbose=False)

    segments = []
    current_segment = []

    for frame_idx, r in enumerate(results):
        is_good = False
        if r.keypoints and len(r.keypoints.conf) > 0:
            for person_conf in r.keypoints.conf:
                visible_count = (person_conf > 0.4).sum().item()
                avg_conf = person_conf.mean().item()
                if visible_count >= min_kpts and avg_conf >= min_conf:
                    is_good = True
                    break

        if is_good:
            current_segment.append(frame_idx)
        else:
            if len(current_segment) >= required_frames:
                start_sec = current_segment[0] / fps
                end_sec = current_segment[-1] / fps
                segments.append((start_sec, end_sec, end_sec - start_sec))
            current_segment = []

    # Final segment
    if len(current_segment) >= required_frames:
        start_sec = current_segment[0] / fps
        end_sec = current_segment[-1] / fps
        segments.append((start_sec, end_sec, end_sec - start_sec))

    total_dur = total_frames / fps
    kept_dur = sum(s[2] for s in segments)
    return segments, total_dur, kept_dur


def extract_clip(video_path, start_sec, end_sec, output_path):
    """Extract a segment from a video using ffmpeg."""
    duration = end_sec - start_sec
    cmd = [
        "ffmpeg", "-y", "-ss", str(start_sec), "-t", str(duration),
        "-i", video_path, "-c:v", "libx264", "-preset", "ultrafast",
        "-crf", "23", "-an", output_path,
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


def run_wham_on_clip(clip_path, output_dir, timeout=600):
    """Run WHAM + LMA on a single clip.

    start_new_session=True puts the child in its own process group so we can
    killpg() the entire WHAM tree (main + DataLoader/SLAM workers) on timeout.
    stdout/stderr go to a log file rather than pipes — this avoids the
    pipe-buffer-fill deadlock where the child blocks on pipe_write and the
    parent's communicate() polls forever (the bug that left zombie children
    alive for 7+ hours).
    """
    import signal
    log_path = os.path.join(output_dir, "wham_inference.log")
    with open(log_path, "ab") as logf:
        logf.write(f"\n----- {time.strftime('%Y-%m-%d %H:%M:%S')} {clip_path} -----\n".encode())
        logf.flush()
        proc = subprocess.Popen(
            [sys.executable, os.path.join(REPO, "core", "wham_inference.py"),
             "--video", clip_path,
             "--output_dir", output_dir],
            stdout=logf, stderr=logf,
            start_new_session=True,
        )
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    return False  # give up; parent moves on to next video
    return proc.returncode == 0


def process_video(video_path, output_dir, model, log=print):
    """Full pipeline: YOLO filter → extract clips → WHAM + LMA."""
    vid_id = os.path.splitext(os.path.basename(video_path))[0]
    vid_out = os.path.join(output_dir, vid_id)

    # Skip if already processed (search recursively — WHAM saves into _clip_seg*/ subdirs)
    if glob.glob(os.path.join(vid_out, "**", "lma_dict_id*.npy"), recursive=True):
        return "skipped", "already processed"

    # Skip if previously failed
    fail_marker = os.path.join(vid_out, "_FAILED")
    if os.path.exists(fail_marker):
        return "skipped", "previously failed"

    os.makedirs(vid_out, exist_ok=True)

    # Step 1: YOLO filter
    try:
        segments, total_dur, kept_dur = yolo_filter(
            video_path, model, MIN_KEYPOINTS, MIN_CONFIDENCE, MIN_SEGMENT_SECONDS
        )
    except Exception as e:
        with open(fail_marker, "w") as f:
            f.write(f"YOLO filter failed: {e}")
        return "failed", f"YOLO error: {str(e)[:100]}"

    if not segments:
        with open(fail_marker, "w") as f:
            f.write("no segments with visible humans")
        # Save filter metadata even for skipped videos
        meta = {"total_duration": total_dur, "kept_duration": 0, "segments": 0, "reason": "no visible humans"}
        with open(os.path.join(vid_out, "filter_meta.json"), "w") as f:
            json.dump(meta, f)
        return "filtered_out", f"no visible humans in {total_dur:.0f}s video"

    ratio = kept_dur / total_dur if total_dur > 0 else 0
    log(f"  YOLO: {len(segments)} segments, {kept_dur:.0f}s / {total_dur:.0f}s kept ({ratio:.0%})")

    # Save filter metadata
    meta = {
        "total_duration": total_dur,
        "kept_duration": kept_dur,
        "filter_ratio": ratio,
        "segments": [{"start": s[0], "end": s[1], "duration": s[2]} for s in segments],
    }
    with open(os.path.join(vid_out, "filter_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Step 2: Extract and process each segment
    any_success = False
    for seg_idx, (start, end, dur) in enumerate(segments):
        clip_path = os.path.join(vid_out, f"_clip_seg{seg_idx}.mp4")
        try:
            extract_clip(video_path, start, end, clip_path)

            # Step 3: WHAM + LMA on the clip
            success = run_wham_on_clip(clip_path, vid_out, timeout=WHAM_TIMEOUT)
            if success:
                any_success = True
                log(f"  seg{seg_idx}: {dur:.0f}s OK")
            else:
                log(f"  seg{seg_idx}: {dur:.0f}s WHAM failed")

        except subprocess.TimeoutExpired:
            log(f"  seg{seg_idx}: {dur:.0f}s timeout")
        except Exception as e:
            log(f"  seg{seg_idx}: {dur:.0f}s error: {str(e)[:80]}")
        finally:
            # Clean up clip file to save disk space
            if os.path.exists(clip_path):
                os.remove(clip_path)

    if any_success and glob.glob(os.path.join(vid_out, "**", "lma_dict_id*.npy"), recursive=True):
        return "processed", f"{len(segments)} segs, {kept_dur:.0f}s kept"
    else:
        with open(fail_marker, "w") as f:
            f.write(f"WHAM failed on all {len(segments)} segments")
        return "failed", f"WHAM failed on all segments"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=1000)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--source-dir", type=str, nargs="+", default=[T3_DIR] if T3_DIR else [])
    parser.add_argument("--output-dir", type=str, default=OUT_ROOT)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load YOLO model
    from ultralytics import YOLO
    model = YOLO(YOLO_MODEL_PATH)

    # Get all video files (recursive, multiple extensions, across multiple source dirs)
    exts = ("*.mp4", "*.webm", "*.mkv", "*.avi", "*.mov")
    all_videos = []
    for src in args.source_dir:
        for ext in exts:
            all_videos.extend(glob.glob(os.path.join(src, "**", ext), recursive=True))
    all_videos = sorted(set(all_videos))
    selected = all_videos[args.start:args.end]

    log_path = os.path.join(args.output_dir, f"gpu{args.gpu_id}_filtered_log.txt")

    def log(msg):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    log(f"Starting filtered batch: videos {args.start}-{args.end} ({len(selected)} videos) on GPU {args.gpu_id}")

    counts = {"processed": 0, "skipped": 0, "failed": 0, "filtered_out": 0}
    start_time = time.time()

    for i, video_path in enumerate(selected):
        vid_id = os.path.splitext(os.path.basename(video_path))[0]

        # Thermal check every 3 videos
        if i % 3 == 0:
            wait_for_cool(args.gpu_id, log=log)

        elapsed = time.time() - start_time
        done = sum(counts.values())
        rate = done / (elapsed / 3600) if elapsed > 0 else 0
        remaining = (len(selected) - i) / rate if rate > 0 else 0
        log(f"[{i+1}/{len(selected)}] {vid_id} (rate: {rate:.1f} vid/hr, ETA: {remaining:.1f} hr)")

        status, msg = process_video(video_path, args.output_dir, model, log=log)
        counts[status] += 1
        if status != "skipped":
            log(f"  -> {status}: {msg}")

    total_time = (time.time() - start_time) / 3600
    log(f"\nBatch complete in {total_time:.1f} hours:")
    for k, v in counts.items():
        log(f"  {k}: {v}")


if __name__ == "__main__":
    main()
