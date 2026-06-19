"""WHAM inference driver for the suggestive-motion pipeline.

ATTRIBUTION. The per-video inference orchestration here — `preprocess_video()` and the
Phase-2 forward loop — is ADAPTED (a near line-for-line port) from WHAM's own `demo.py`
`run()` (yohanshin/WHAM, CC BY-NC research license); the detection / ViTPose / SLAM /
network-forward calls are WHAM library APIs used unchanged. FIRST-PARTY (ours): the
SMPL-24 `J_regressor @ verts` joint fix, the per-fragment `.npz` output + ffmpeg crops,
and the spawn / single-thread-BLAS / CUDA-recovery infrastructure.

WHAM is run from a MODIFIED FORK (zaiisao/WHAM @ baca651), not stock WHAM: YOLO26x
detector, ViTPose++ (base, MoE/UDP) 2D, OKS cross-frame association, and full-fp32 SLAM.
Every deviation from upstream is enumerated in docs/wham_fork_vs_official_audit_2026-06-09.md.

PIPELINE NOTE. This driver's job is to produce the per-fragment WHAM `.npz`. The canonical,
data-producing LMA features are computed by `scripts/extract_lma_features.py` from the
CAMERA-FRAME mesh vertices (`J_regressor @ verts`, no `trans_world`). The published
features.npz was built from a minimal camera-frame npz (`joints, verts, frame_ids, fps`);
this driver's job ends at the `.npz`; the LMA is computed downstream by
`scripts/extract_lma_features.py` (camera-frame `J_regressor @ verts` + mesh volumes), the
single canonical WHAM -> LMA path. (The richer/world-frame npz schema saved below is a
known follow-up to reconcile with that camera-frame stage; see the audit doc.)
"""
import sys
import os
import multiprocessing as _mp
# Force spawn so DataLoader / DPVO / multiprocessing workers don't inherit
# the parent's mid-init CUDA contexts (causes a futex/pipe deadlock where
# all workers park forever and the 10-min watchdog can't reach them).
try:
    _mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass
# Single-threaded BLAS keeps fork-safe libraries from spawning their own
# thread pools that race with PyTorch's CUDA initialization.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
np.float = float

sys.setrecursionlimit(5000)

# 1. Get Absolute Path to the project root
# This assumes wham_inference.py is in <repo>/core/
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))

# 2. Define external paths clearly
wham_root = os.path.join(project_root, "external/WHAM")
dpvo_path = os.path.join(wham_root, "third-party/DPVO")
lma_path = os.path.join(project_root, "external/dance-style-recognition/src")
vitpose_path = os.path.join(wham_root, "third-party/ViTPose")

# 3. Insert paths at INDEX 0 (Highest Priority)
# This forces Python to look in WHAM's folders first
for p in [wham_root, dpvo_path, lma_path, vitpose_path]:
    if p not in sys.path:
        sys.path.insert(0, p)

# NOTE: SLAMModel is intentionally NOT imported here. It is imported lazily in
# the try/except block below so a missing/broken DPVO build only disables global
# trajectory (_run_global=False) rather than crashing this module at import time.

import cv2
import torch
import joblib
import argparse
import json
import os.path as osp
from glob import glob
from collections import defaultdict
from progress.bar import Bar
from loguru import logger

from configs.config import get_cfg_defaults
from lib.data.datasets import CustomDataset
from lib.utils.imutils import avg_preds
from lib.utils.transforms import matrix_to_axis_angle
from lib.models import build_network, build_body_model
from lib.models.preproc.detector import DetectionModel
from lib.models.preproc.extractor import FeatureExtractor
from lib.models.smplify import TemporalSMPLify
from lib.vis.run_vis import run_vis_on_demo

import subprocess

try: 
    from lib.models.preproc.slam import SLAMModel
    _run_global = True
except ImportError: 
    logger.warning('DPVO (SLAM) is not installed. Global trajectory will default to local camera space!')
    _run_global = False

class WHAMLMAProcessor:
    def __init__(self, cfg_path='configs/yamls/demo.yaml'):
        print("[*] Initializing WHAM + LMA Processor...")
        self.cfg = get_cfg_defaults()
        self.cfg.DEVICE = f'cuda:0' if torch.cuda.is_available() else 'cpu'

        script_dir = os.path.dirname(os.path.abspath(__file__))
        wham_root = os.path.abspath(os.path.join(script_dir, '..', 'external', 'WHAM'))
        self.wham_root = wham_root
        full_cfg_path = os.path.join(wham_root, cfg_path)
        
        self.cfg.merge_from_file(full_cfg_path)
        
        original_cwd = os.getcwd()
        os.chdir(wham_root)
        try:
            # Build WHAM SMPL Model & Network
            smpl_batch_size = self.cfg.TRAIN.BATCH_SIZE * self.cfg.DATASET.SEQLEN
            self.smpl = build_body_model(self.cfg.DEVICE, smpl_batch_size)
            self.network = build_network(self.cfg, self.smpl)
            self.network.eval()
            
            # Detector & Extractor (Replaces YOLO & ViTPose)
            self.detector = DetectionModel(self.cfg.DEVICE)
            self.extractor = FeatureExtractor(self.cfg.DEVICE, self.cfg.FLIP_EVAL)
        finally:
            os.chdir(original_cwd)

    def preprocess_video(self, video_path, output_pth, calib=None, use_slam=True):
        """Replaces Phase 1: 2D Extraction."""

        # Reset detector tracking state from any previous video. Critical when a
        # single processor instance is reused across videos (e.g. batch mode),
        # otherwise tracks leak across videos and corrupt fragment IDs.
        self.detector.initialize_tracking()

        with torch.no_grad():
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            
            use_slam = use_slam and _run_global
            slam = SLAMModel(video_path, output_pth, width, height, calib, buffer=16384) if use_slam else None
            
            bar = Bar('Preprocessing: Tracking and SLAM', fill='#', max=length)
            while cap.isOpened():
                flag, img = cap.read()
                if not flag: break
                
                self.detector.track(img, fps, length)
                if slam is not None:
                    slam.track(video_path)
                bar.next()
            cap.release()

            tracking_results = self.detector.process(fps)
            if not tracking_results:
                print("[!] No valid tracking results after detection.")
                return None, fps

            slam_results = slam.process() if slam is not None else np.zeros((length, 7))
            if slam is None: slam_results[:, 3] = 1.0
            
            tracking_results = self.extractor.run(video_path, tracking_results)
            return CustomDataset(self.cfg, tracking_results, slam_results, width, height, fps), fps

    def run_pipeline(self, video_path, output_dir, visualize=False):
        """Replaces Phase 2: 3D Lifting (MotionBERT) -> Now using WHAM"""
        os.makedirs(output_dir, exist_ok=True)
        original_cwd = os.getcwd()
        os.chdir(self.wham_root)
        try:
            print("\n[*] Phase 1: Preprocessing Video & Extracting Features...")
            dataset, fps = self.preprocess_video(video_path, output_dir)
            if dataset is None:
                return {}, fps
            
            print("\n[*] Phase 2: WHAM 3D Inference & Global Optimization...")
            results = defaultdict(dict)
            n_subjs = len(dataset)
            
            for subj in range(n_subjs):
                with torch.no_grad():
                    batch = dataset.load_data(subj)
                    _id, x, inits, features, mask, init_root, cam_angvel, frame_id, kwargs = batch
                    
                    # WHAM Inference
                    pred = self.network(x, inits, features, mask=mask, init_root=init_root, cam_angvel=cam_angvel, return_y_up=True, **kwargs)
                    
                    # 1. Align temporal dimensions (T) and combine root + body poses
                    # Force shape to (Batch=1, Time=T, Joints, 3, 3)
                    root_pose = pred['poses_root_world'].reshape(1, -1, 1, 3, 3)
                    body_pose = pred['poses_body'].reshape(1, -1, 23, 3, 3)
                    poses_world_mat = torch.cat([root_pose, body_pose], dim=2)
                    
                    # 2. Extract the 6D rotation tensor AND preserve the 3D shape (1, T, 144)
                    pred_rot6d_world = poses_world_mat[..., :3, :2].contiguous().reshape(1, -1, 144)
                    
                    # 3. Call WHAM's custom SMPL wrapper with the exact arguments it requires
                    smpl_output = self.network.smpl(
                        pred_rot6d=pred_rot6d_world,
                        betas=pred['betas']
                    )
                    
                    # 4. Extract 3D data, apply world translation, and strip the dummy batch dimension
                    trans_world = pred['trans_world'].reshape(1, -1, 1, 3) # (1, T, 1, 3)
                    joints_world = (smpl_output.joints + trans_world).cpu().squeeze(0).numpy() # -> (T, 31, 3) COCO+SPIN (NOT SMPL-24; see LMA fix below)
                    verts_world = (smpl_output.vertices + trans_world).cpu().squeeze(0).numpy() # -> (T, 6890, 3)
                    
                    # 5. Restore all original WHAM dictionary keys for the visualizer
                    root_world_aa = matrix_to_axis_angle(pred['poses_root_world']).cpu().numpy().reshape(-1, 3)
                    root_cam_aa = matrix_to_axis_angle(pred['poses_root_cam']).cpu().numpy().reshape(-1, 3)
                    body_aa = matrix_to_axis_angle(pred['poses_body']).cpu().numpy().reshape(-1, 69)
                    
                    results[_id]['frame_ids'] = frame_id
                    results[_id]['betas'] = pred['betas'].cpu().squeeze(0).numpy()
                    results[_id]['pose'] = np.concatenate((root_cam_aa, body_aa), axis=-1)
                    results[_id]['pose_world'] = np.concatenate((root_world_aa, body_aa), axis=-1)
                    
                    # Foolproof trans and verts handling for the visualizer
                    trans_cam = pred['trans_cam'].cpu().squeeze(0).numpy()
                    results[_id]['trans'] = trans_cam - self.network.output.offset.cpu().numpy()
                    results[_id]['trans_world'] = pred['trans_world'].cpu().squeeze(0).numpy()
                    
                    verts_cam = pred['verts_cam'].cpu().squeeze(0).numpy()
                    results[_id]['verts'] = verts_cam + trans_cam[:, None, :] # Broadcast trans to (T, 1, 3)
                    
                    # 6. Store our LMA-specific parameters!
                    results[_id]['joints_world'] = joints_world

                    # 7. Original 2D keypoints from ViTPose (pixel coords, for overlay)
                    kp2d = dataset.tracking_results[_id]['keypoints']  # (T, 17, 3) — x, y, conf
                    results[_id]['keypoints_2d'] = kp2d
                    results[_id]['verts_world'] = verts_world

            if not results:
                print("[!] No subjects detected.")
                return None, fps
                
            processed_fragments = {}
            
            for _id, data in results.items():
                frames = data['frame_ids']
                
                # Optional: Skip noise/glitches (e.g., tracks shorter than 1 second)
                if len(frames) < 15:
                    print(f"[*] Skipping ID {_id} (Too short: {len(frames)} frames)")
                    continue
                    
                print(f"\n[*] Processing Fragment ID {_id} with {len(frames)} frames...")

                # 1. Save specific NPZ for this fragment
                out_npz = osp.join(output_dir, f"wham_fragment_id{_id}.npz")
                np.savez(
                    out_npz,
                    joints=data['joints_world'],
                    keypoints_2d=data['keypoints_2d'],
                    verts=data['verts_world'],
                    pose_world=data['pose_world'],
                    betas=data['betas'],
                    trans_world=data['trans_world'],
                    frame_ids=frames,
                    fps=fps
                )
                print(f"    -> Saved kinematics: {out_npz}")
                processed_fragments[_id] = data

                # 2. Render and Crop Video for this fragment
                if visualize:
                    # Create a temp folder for WHAM's native renderer
                    temp_dir = osp.join(output_dir, f"temp_vis_{_id}")
                    os.makedirs(temp_dir, exist_ok=True)
                    
                    # Render the full-length video but ONLY drawing this specific ID
                    run_vis_on_demo(self.cfg, video_path, {_id: data}, temp_dir, self.network.smpl, vis_global=_run_global)
                    
                    generated_videos = glob(osp.join(temp_dir, '*.mp4'))
                    if len(generated_videos) > 0:
                        raw_render = generated_videos[0]
                        final_cropped_video = osp.join(output_dir, f"preview_fragment_id{_id}.mp4")
                        
                        # Calculate timestamps based on frame indices
                        start_frame = int(np.min(frames))
                        end_frame = int(np.max(frames))
                        
                        start_time = start_frame / fps
                        duration = (end_frame - start_frame + 1) / fps
                        
                        print(f"    -> Cropping video from {start_time:.2f}s to {start_time+duration:.2f}s")
                        
                        # FFmpeg trims the dead space where the raw video was showing nothing
                        cmd = [
                            'ffmpeg', '-y', 
                            '-ss', str(start_time), 
                            '-t', str(duration),
                            '-i', raw_render, 
                            '-c:v', 'libx264', '-crf', '23', '-preset', 'fast', 
                            final_cropped_video
                        ]
                        
                        ffmpeg_proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                        if ffmpeg_proc.returncode == 0 and osp.exists(final_cropped_video):
                            os.remove(raw_render)
                            if osp.isdir(temp_dir) and len(os.listdir(temp_dir)) == 0:
                                os.rmdir(temp_dir)
                            print(f"    -> Saved preview video: {final_cropped_video}")
                        else:
                            print(f"[!] FFmpeg crop failed for Fragment {_id}; keeping raw render at {raw_render}")

            return processed_fragments, fps
        finally:
            os.chdir(original_cwd)

def process_single_video(video_path, output_root, visualize=False):
    """
    Worker function for multiprocessing. 
    Each process creates its own WHAMLMAProcessor instance.
    """

    video_path = os.path.abspath(video_path)
    output_root = os.path.abspath(output_root)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_dir = os.path.join(output_root, video_name)
    os.makedirs(video_output_dir, exist_ok=True)
    
    # Pass the device_id here
    processor = WHAMLMAProcessor()

    # Clear any GPU cache left from a previous video to curb memory growth.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        fragments, fps = processor.run_pipeline(video_path, video_output_dir, visualize=visualize)
    except RuntimeError as e:
        if 'CUDA' in str(e):
            # A CUDA error corrupts the whole GPU context — this worker can't be
            # trusted again. Exit hard so a Pool with maxtasksperchild respawns a
            # clean worker instead of cascading the failure onto later videos.
            import traceback
            print(f"[FATAL CUDA] Worker {os.getpid()} hit a CUDA error on {video_path}")
            traceback.print_exc()
            os._exit(1)
        import traceback
        print(f"⚠️ Video processing failed: {video_path}")
        print(f"Error: {e}")
        traceback.print_exc()
        return False, "Skipped due to RuntimeError"
    except Exception as e:
        import traceback
        print(f"⚠️ Video processing failed (likely exceeded frame buffer): {video_path}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error: {e}")
        traceback.print_exc()
        return False, "Skipped due to internal WHAM/DPVO error"

    summary = {
        'video_path': video_path,
        'video_output_dir': video_output_dir,
        'fps': float(fps) if fps is not None else None,
        'num_fragments': int(len(fragments)) if fragments else 0,
        'fragment_frame_counts': {},
        'written_files': [],
        'status': 'failed',
        'reason': ''
    }

    if fragments:
        for _id, data in fragments.items():
            summary['fragment_frame_counts'][str(_id)] = int(len(data['frame_ids']))

    if not fragments:
        summary['reason'] = 'No valid fragments'
        summary_path = osp.join(video_output_dir, 'summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        return False, f"No valid fragments: {video_path}"

    # This driver's output is the per-fragment WHAM .npz saved above. The LMA descriptors are
    # NOT computed here: that is the job of scripts/extract_lma_features.py, the single
    # canonical WHAM -> LMA path (camera-frame J_regressor@verts + mesh-hull volumes ->
    # the descriptor). Keeping it one place avoids duplicating (and drifting from) that code.
    print(f"\n[+] WHAM complete for {video_name}: {len(fragments)} fragment(s) saved.")
    summary['written_files'] = [f"wham_fragment_id{_id}.npz" for _id in fragments]

    summary['status'] = 'success'
    summary['reason'] = ''
    summary_path = osp.join(video_output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    return True, video_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output/wham_lma")
    parser.add_argument("--viz", action='store_true')
    opts = parser.parse_args()

    # Just call the worker function directly
    success, message = process_single_video(
        opts.video, 
        opts.output_dir, 
        visualize=opts.viz, 
    )
    
    if success:
        print(f"\n[SUCCESS] Pipeline complete for {message}")
    else:
        print(f"\n[ERROR] {message}")