#!/usr/bin/env python
"""
Contract unit tests for the batch-processing orchestration in
    /home/sogang/jaehoon/KineGuard/core/batch_processor.py

These are CONTRACT tests: they assert the behavior the production code actually
implements (derived from the source), using only temp dirs. No real outputs are
touched and no pipeline code is modified.

ISOLATION APPROACH
------------------
batch_processor.py imports `torch` at top level and run_multiprocess_pipeline()
does `from wham_inference import process_single_video` (heavy CUDA/model deps).
Importing the module top-level is therefore expensive/fragile. We do NOT import
it. Instead we re-create the *exact* pure-logic expressions from the source as
local functions (with file:line provenance in comments) and test those. Each
helper is a faithful transcription of the production expression; if the source
changes, these tests must be updated -- that is the point of a contract test.

DERIVED CONTRACT (batch_processor.py, current source)
-----------------------------------------------------
1. Resume / "already processed" rule  -- inlined at lines 54-62:
       video_name       = os.path.splitext(os.path.basename(vid))[0]   # L54
       video_output_dir = os.path.join(output_root, video_name)        # L55
       if os.path.exists(video_output_dir):                            # L59
           lma_files = glob(.../"lma_features_id*.npy")                # L60
           if len(lma_files) > 0:                                      # L61
               continue  # skip: already done                          # L62
   => "already processed" IFF dir exists AND >=1 file matching
      lma_features_id*.npy. summary.json alone does NOT count.
   Cross-checked against wham_inference.py: summary.json is written early
   (L322) but lma_features_id{_id}.npy is written LAST (L360), so the .npy
   sentinel is the correct "fully done" marker.

2. GPU queue assignment  -- lines 86-87:
       for i in range(num_processes):
           gpu_id_queue.put(i % num_gpus)
   => round-robin: worker i gets GPU (i % num_gpus).

3. Video gathering  -- lines 41-45:
       extensions = ['*.mp4','*.avi','*.mov','*.mkv']
       glob(input_folder/ext) + glob(input_folder/ext.upper())
   => only listed extensions, lower- and upper-case, from the single
      input_folder (non-recursive).

NOTE ON TASK SPEC MISMATCHES (reported, not silently accommodated):
  * There is NO function named `_is_already_processed` -- the rule is inlined.
  * There is NO `legacy_output_roots` feature in this module.
  * GPU code uses `i % num_gpus` (num_gpus = torch.cuda.device_count()),
    not a `gpu_ids[i % len(gpu_ids)]` list.
  * There is NO category-file gathering; gathering is by extension glob.
These are documented in the final report; the tests below cover the contract
the code ACTUALLY implements.
"""

import os
import sys
import tempfile
import shutil
from glob import glob

# ---------------------------------------------------------------------------
# Faithful transcriptions of the production pure-logic (see provenance above).
# ---------------------------------------------------------------------------

def video_output_dir_for(output_root, video_path):
    """Mirror of batch_processor.py L54-L55 (and wham_inference.py L291-292)."""
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(output_root, video_name)


def is_already_processed(video_output_dir):
    """Mirror of the inlined resume rule, batch_processor.py L59-L62.

    Returns True iff the directory exists AND contains >=1 file matching
    'lma_features_id*.npy'.
    """
    if os.path.exists(video_output_dir):
        lma_files = glob(os.path.join(video_output_dir, "lma_features_id*.npy"))
        if len(lma_files) > 0:
            return True
    return False


def build_gpu_queue(num_processes, num_gpus):
    """Mirror of batch_processor.py L86-L87 (round-robin GPU assignment)."""
    return [i % num_gpus for i in range(num_processes)]


def gather_videos(input_folder):
    """Mirror of batch_processor.py L41-L45 (extension-based gathering)."""
    extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
    all_videos = []
    for ext in extensions:
        all_videos.extend(glob(os.path.join(input_folder, ext)))
        all_videos.extend(glob(os.path.join(input_folder, ext.upper())))
    return all_videos


# ---------------------------------------------------------------------------
# Tiny test harness (no pytest dependency).
# ---------------------------------------------------------------------------

_PASS = 0
_FAIL = 0
_FAILURES = []


def check(name, condition, evidence=""):
    global _PASS, _FAIL
    if condition:
        _PASS += 1
        print(f"  PASS: {name}")
    else:
        _FAIL += 1
        _FAILURES.append((name, evidence))
        print(f"  FAIL: {name}" + (f"  [{evidence}]" if evidence else ""))


def touch(path, content="x"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_is_already_processed():
    print("\n[1] _is_already_processed / inlined resume rule (L59-62)")
    tmp = tempfile.mkdtemp(prefix="batchtest_")
    try:
        # (a) directory does NOT exist
        d_missing = os.path.join(tmp, "does_not_exist")
        check("(a) non-existent dir -> not processed",
              is_already_processed(d_missing) is False,
              f"got {is_already_processed(d_missing)}")

        # (b) directory exists but empty
        d_empty = os.path.join(tmp, "empty")
        os.makedirs(d_empty)
        check("(b) empty dir -> not processed",
              is_already_processed(d_empty) is False,
              f"got {is_already_processed(d_empty)}")

        # (c) contains lma_features_id0.npy -> processed
        d_done = os.path.join(tmp, "done")
        os.makedirs(d_done)
        touch(os.path.join(d_done, "lma_features_id0.npy"))
        check("(c) has lma_features_id0.npy -> processed",
              is_already_processed(d_done) is True,
              f"got {is_already_processed(d_done)}")

        # (c2) multiple ids still processed
        d_done2 = os.path.join(tmp, "done2")
        os.makedirs(d_done2)
        touch(os.path.join(d_done2, "lma_features_id0.npy"))
        touch(os.path.join(d_done2, "lma_features_id3.npy"))
        check("(c2) multiple lma_features_id*.npy -> processed",
              is_already_processed(d_done2) is True,
              f"got {is_already_processed(d_done2)}")

        # (d) contains ONLY summary.json -> NOT processed (half-done crash case)
        d_summary = os.path.join(tmp, "summary_only")
        os.makedirs(d_summary)
        touch(os.path.join(d_summary, "summary.json"), '{"ok": true}')
        check("(d) summary.json only -> NOT processed (crash-safe resume)",
              is_already_processed(d_summary) is False,
              f"got {is_already_processed(d_summary)} -- WOULD SILENTLY SKIP A HALF-DONE VIDEO")

        # (d2) summary.json + wham fragment but NO lma .npy -> NOT processed
        d_partial = os.path.join(tmp, "partial")
        os.makedirs(d_partial)
        touch(os.path.join(d_partial, "summary.json"))
        touch(os.path.join(d_partial, "wham_fragment_id0.npz"))
        check("(d2) wham fragment but no LMA .npy -> NOT processed",
              is_already_processed(d_partial) is False,
              f"got {is_already_processed(d_partial)}")

        # (e) unrelated file only -> NOT processed
        d_unrelated = os.path.join(tmp, "unrelated")
        os.makedirs(d_unrelated)
        touch(os.path.join(d_unrelated, "notes.txt"))
        check("(e) unrelated file only -> NOT processed",
              is_already_processed(d_unrelated) is False,
              f"got {is_already_processed(d_unrelated)}")

        # (e2) a similarly-named but non-matching .npy -> NOT processed
        # The glob pattern is lma_features_id*.npy; lma_dict_id*.npy must NOT count.
        d_dictonly = os.path.join(tmp, "dict_only")
        os.makedirs(d_dictonly)
        touch(os.path.join(d_dictonly, "lma_dict_id0.npy"))
        check("(e2) lma_dict_id0.npy only (not 'features') -> NOT processed",
              is_already_processed(d_dictonly) is False,
              f"got {is_already_processed(d_dictonly)} -- pattern would over-match")

        # video-name derivation matches process_single_video convention
        vod = video_output_dir_for("/out", "/in/My Clip.001.mp4")
        check("video_output_dir uses splitext(basename) -> 'My Clip.001'",
              vod == os.path.join("/out", "My Clip.001"),
              f"got {vod}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_resume_legacy_roots():
    print("\n[2] Resume across legacy_output_roots")
    # The current source has NO legacy_output_roots support: resume only checks
    # the single active output_root (L55). A video already finished in a
    # *legacy* root but absent from the active root would be RE-PROCESSED.
    # We assert the actual behavior and flag the missing feature.
    tmp = tempfile.mkdtemp(prefix="batchtest_legacy_")
    try:
        active_root = os.path.join(tmp, "active")
        legacy_root = os.path.join(tmp, "legacy")
        os.makedirs(active_root)
        # Video finished in legacy root only.
        legacy_done = os.path.join(legacy_root, "clipA")
        os.makedirs(legacy_done)
        touch(os.path.join(legacy_done, "lma_features_id0.npy"))

        # Active-root check (what the code actually does):
        active_vod = video_output_dir_for(active_root, "/src/clipA.mp4")
        active_skip = is_already_processed(active_vod)
        check("active-root resume: legacy-only video is NOT skipped (re-processed)",
              active_skip is False,
              f"active_skip={active_skip}")

        # Document: IF the contract included legacy roots, this would be True.
        legacy_vod = video_output_dir_for(legacy_root, "/src/clipA.mp4")
        check("(context) the file genuinely exists in the legacy root",
              is_already_processed(legacy_vod) is True,
              f"got {is_already_processed(legacy_vod)}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_gpu_queue_assignment():
    print("\n[3] GPU round-robin queue (L86-87: i % num_gpus)")
    # 4 GPUs, 6 workers -> 0,1,2,3,0,1
    check("4 gpus / 6 workers -> [0,1,2,3,0,1]",
          build_gpu_queue(6, 4) == [0, 1, 2, 3, 0, 1],
          f"got {build_gpu_queue(6, 4)}")
    # 1 GPU, 3 workers -> all 0
    check("1 gpu / 3 workers -> [0,0,0]",
          build_gpu_queue(3, 1) == [0, 0, 0],
          f"got {build_gpu_queue(3, 1)}")
    # 2 GPUs, 2 workers -> [0,1]
    check("2 gpus / 2 workers -> [0,1]",
          build_gpu_queue(2, 2) == [0, 1],
          f"got {build_gpu_queue(2, 2)}")
    # 0 workers -> empty queue
    check("0 workers -> []",
          build_gpu_queue(0, 4) == [],
          f"got {build_gpu_queue(0, 4)}")
    # balanced load: with N divisible by G each GPU used N/G times
    q = build_gpu_queue(8, 4)
    counts = [q.count(g) for g in range(4)]
    check("8 workers / 4 gpus -> each gpu used exactly twice",
          counts == [2, 2, 2, 2],
          f"counts={counts}")
    # every assigned id is a valid GPU index (0 <= id < num_gpus)
    q2 = build_gpu_queue(10, 3)
    check("all assigned ids within [0, num_gpus)",
          all(0 <= g < 3 for g in q2),
          f"q={q2}")


def test_video_gathering():
    print("\n[4] Video gathering by extension (L41-45)")
    tmp = tempfile.mkdtemp(prefix="batchtest_gather_")
    try:
        # listed extensions (lower + upper case)
        for fn in ["a.mp4", "b.MP4", "c.avi", "d.mov", "e.mkv", "f.MKV"]:
            touch(os.path.join(tmp, fn))
        # files that must be ignored
        for fn in ["g.txt", "h.json", "i.webm", "j.npy", "k.mp4.bak"]:
            touch(os.path.join(tmp, fn))
        # a subdirectory file that must NOT be picked up (non-recursive glob)
        touch(os.path.join(tmp, "sub", "deep.mp4"))

        found = sorted(os.path.basename(p) for p in gather_videos(tmp))
        expected = sorted(["a.mp4", "b.MP4", "c.avi", "d.mov", "e.mkv", "f.MKV"])
        check("gathers exactly listed video extensions (case-insensitive)",
              found == expected,
              f"found={found} expected={expected}")
        check("ignores non-video extensions (.txt/.json/.webm/.npy/.bak)",
              all(f not in found for f in ["g.txt", "h.json", "i.webm", "j.npy", "k.mp4.bak"]),
              f"found={found}")
        check("non-recursive: ignores videos in subdirectories",
              "deep.mp4" not in found,
              f"found={found}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_end_to_end_filtering():
    print("\n[5] Integrated resume filter over a gathered folder")
    tmp = tempfile.mkdtemp(prefix="batchtest_e2e_")
    try:
        input_folder = os.path.join(tmp, "in")
        output_root = os.path.join(tmp, "out")
        os.makedirs(input_folder)
        os.makedirs(output_root)
        for fn in ["done.mp4", "partial.mp4", "fresh.mp4"]:
            touch(os.path.join(input_folder, fn))

        # 'done' fully processed
        d = os.path.join(output_root, "done"); os.makedirs(d)
        touch(os.path.join(d, "lma_features_id0.npy"))
        # 'partial' has summary.json but no LMA features
        p = os.path.join(output_root, "partial"); os.makedirs(p)
        touch(os.path.join(p, "summary.json"))
        # 'fresh' has no output dir at all

        all_videos = gather_videos(input_folder)
        remaining = []
        for vid in all_videos:
            vod = video_output_dir_for(output_root, vid)
            if is_already_processed(vod):
                continue
            remaining.append(os.path.basename(vid))
        remaining = sorted(remaining)
        check("only 'done.mp4' is skipped; partial+fresh remain",
              remaining == ["fresh.mp4", "partial.mp4"],
              f"remaining={remaining}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    print("=" * 64)
    print("Contract tests for KineGuard/core/batch_processor.py orchestration")
    print("=" * 64)
    test_is_already_processed()
    test_resume_legacy_roots()
    test_gpu_queue_assignment()
    test_video_gathering()
    test_end_to_end_filtering()

    total = _PASS + _FAIL
    print("\n" + "=" * 64)
    print(f"RESULT: {_PASS}/{total} checks passed, {_FAIL} failed")
    print("=" * 64)
    if _FAILURES:
        print("FAILURES:")
        for name, ev in _FAILURES:
            print(f"  - {name}: {ev}")
    sys.exit(1 if _FAIL else 0)


if __name__ == "__main__":
    main()
