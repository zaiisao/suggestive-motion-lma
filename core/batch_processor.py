import multiprocessing as mp
import os
import torch
import argparse
from glob import glob
import multiprocessing.pool

class NoDaemonProcess(mp.Process):
    @property
    def daemon(self):
        return False
    @daemon.setter
    def daemon(self, value):
        pass

class NoDaemonPool(multiprocessing.pool.Pool):
    @staticmethod
    def Process(ctx, *args, **kwargs):
        return NoDaemonProcess(*args, **kwargs)

def init_worker(gpu_id_queue):
    """
    This runs once per worker process immediately after it is spawned.
    It locks each worker to a specific physical GPU and pre-loads the processor.
    """
    gpu_id = gpu_id_queue.get()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"[*] Worker {os.getpid()} assigned to GPU {gpu_id}, loading models...")

    import wham_inference
    wham_inference._shared_processor = wham_inference.KineGuardWHAMProcessor()
    print(f"[*] Worker {os.getpid()} models loaded.")

def _is_already_processed(video_output_dir):
    """A video is considered processed if LMA outputs exist OR summary.json exists
    (which is written even on no-person/no-LMA failures)."""
    if not os.path.exists(video_output_dir):
        return False
    if glob(os.path.join(video_output_dir, "lma_features_id*.npy")):
        return True
    if os.path.exists(os.path.join(video_output_dir, "summary.json")):
        return True
    return False


def run_multiprocess_pipeline(input_folder, output_root, num_processes, visualize, legacy_output_roots=None):
    from wham_inference import process_single_video

    # CRITICAL: Must use 'spawn' for CUDA applications
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # 1. GATHER VIDEOS from selected categories
    category_file = "selected_tier01_categories.txt"
    if not os.path.exists(category_file):
        print(f"[!] {category_file} not found.")
        return

    with open(category_file, 'r') as f:
        categories = [line.strip() for line in f if line.strip()]

    all_videos = []  # (video_path, category)
    for cat in categories:
        cat_dir = os.path.join(input_folder, cat)
        if os.path.isdir(cat_dir):
            for vid in sorted(glob(os.path.join(cat_dir, "*.mp4"))):
                all_videos.append((vid, cat))
        else:
            print(f"[!] Category folder not found: {cat_dir}")

    print(f"[*] Found {len(all_videos)} videos from {len(categories)} categories")

    # Filter out already-processed videos. Check the active output_root AND any
    # legacy_output_roots (e.g. an older disk that filled up). New writes still go
    # to output_root, but we won't redo work that already exists in legacy roots.
    legacy_output_roots = legacy_output_roots or []
    all_roots = [output_root] + legacy_output_roots
    video_list = []
    for vid, cat in all_videos:
        video_name = os.path.splitext(os.path.basename(vid))[0]

        already_done = False
        for root in all_roots:
            if _is_already_processed(os.path.join(root, cat, video_name)):
                already_done = True
                break
        if already_done:
            continue

        video_list.append((vid, cat))

    print(f"[*] {len(all_videos) - len(video_list)} already processed. {len(video_list)} remaining.")

    if not video_list:
        print("[*] All videos already processed. Exiting.")
        return

    print(f"[*] Starting pool with {num_processes} workers...")
    os.makedirs(output_root, exist_ok=True)

    # 2. PREPARE GPU QUEUE
    # Respect external CUDA_VISIBLE_DEVICES if set (e.g. "1" or "1,2")
    visible_env = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if visible_env:
        gpu_ids = [g.strip() for g in visible_env.split(",")]
    else:
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            raise RuntimeError("No CUDA GPUs available. This pipeline requires at least one CUDA device.")
        gpu_ids = [str(i) for i in range(num_gpus)]

    manager = mp.Manager()
    gpu_id_queue = manager.Queue()

    for i in range(num_processes):
        gpu_id_queue.put(gpu_ids[i % len(gpu_ids)])

    # 3. INITIALIZE POOL
    # initializer=init_worker ensures the GPU is isolated BEFORE wham_inference is imported
    with NoDaemonPool(processes=num_processes,
                 initializer=init_worker,
                 initargs=(gpu_id_queue,),
                 maxtasksperchild=50) as pool:
        
        # Prepare tasks — pass output_root/category so outputs are organized by category
        tasks = [(vid, os.path.join(output_root, cat), visualize) for vid, cat in video_list]
        results = pool.starmap(process_single_video, tasks)

    # 4. REPORT
    print("\n" + "="*30)
    print("FINAL REPORT")
    print("="*30)
    for success, msg in results:
        status = " [DONE]  " if success else " [FAILED] "
        print(f"{status} {msg}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KineGuard Multiprocessing Video Processor")
    
    parser.add_argument("--input", type=str, required=True, 
                        help="Path to folder containing video files")
    parser.add_argument("--output", type=str, default="output/multiprocess_results", 
                        help="Root directory for output fragments and features")
    parser.add_argument("--jobs", type=int, default=2, 
                        help="Number of parallel processes (workers) to run")
    parser.add_argument("--viz", action='store_true',
                        help="Enable 3D visualization rendering")
    parser.add_argument("--legacy-output", type=str, action='append', default=[],
                        help="Older output root(s) to check for resume only. New writes still go to --output. Pass multiple times for multiple legacy roots.")

    args = parser.parse_args()

    run_multiprocess_pipeline(
        input_folder=args.input,
        output_root=args.output,
        num_processes=args.jobs,
        visualize=args.viz,
        legacy_output_roots=args.legacy_output,
    )