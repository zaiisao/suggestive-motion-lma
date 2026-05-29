import sys
import os
import numpy as np
np.float = float

sys.setrecursionlimit(5000)

# 1. Get Absolute Path to the project root
# This assumes wham_inference.py is in KineGuard/core/
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

# NOTE: SLAMModel import is intentionally deferred to the try/except block
# below so a missing/broken DPVO build only disables global trajectory.

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
from scipy.spatial import ConvexHull

from configs.config import get_cfg_defaults
from lib.data.datasets import CustomDataset
from lib.utils.imutils import avg_preds
from lib.utils.transforms import matrix_to_axis_angle
from lib.models import build_network, build_body_model
from lib.models.preproc.detector import DetectionModel
from lib.models.preproc.extractor import FeatureExtractor
from lib.models.smplify import TemporalSMPLify
from lib.vis.run_vis import run_vis_on_demo

from process_lma_features import compute_lma_descriptor, IdentityFloor

import subprocess

try: 
    from lib.models.preproc.slam import SLAMModel
    _run_global = True
except ImportError: 
    logger.warning('DPVO (SLAM) is not installed. Global trajectory will default to local camera space!')
    _run_global = False

class KineGuardWHAMProcessor:
    def __init__(self, cfg_path='configs/yamls/demo.yaml'):
        print("[*] Initializing KineGuard WHAM Processor...")
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

        # Reset detector state from previous video (critical for shared processor)
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
                    joints_world = (smpl_output.joints + trans_world).cpu().squeeze(0).numpy() # -> (T, 45, 3)
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
                    results[_id]['verts_world'] = verts_world

            if not results:
                print("[!] No subjects detected.")
                return None, fps
                
            processed_fragments = {}
            
            for _id, data in results.items():
                frames = data['frame_ids']
                
                # Optional: Skip noise/glitches (e.g., tracks shorter than 1 second)
                if len(frames) < 30:
                    print(f"[*] Skipping ID {_id} (Too short: {len(frames)} frames)")
                    continue
                    
                print(f"\n[*] Processing Fragment ID {_id} with {len(frames)} frames...")

                # 1. Save specific NPZ for this fragment
                out_npz = osp.join(output_dir, f"wham_fragment_id{_id}.npz")
                np.savez(
                    out_npz,
                    joints=data['joints_world'],
                    verts=data['verts_world'],
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

_shared_processor = None  # Set by init_worker in batch_processor.py (once per worker)

def process_single_video(video_path, output_root, visualize=False):
    """
    Worker function for multiprocessing.
    Reuses _shared_processor created in init_worker (no repeated model loading).
    """
    global _shared_processor
    if _shared_processor is not None:
        processor = _shared_processor
    else:
        # Fallback: if not running via batch_processor pool, create one
        processor = KineGuardWHAMProcessor()

    # Clear GPU cache between videos to prevent memory accumulation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    video_path = os.path.abspath(video_path)
    output_root = os.path.abspath(output_root)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_dir = os.path.join(output_root, video_name)
    os.makedirs(video_output_dir, exist_ok=True)

    try:
        fragments, fps = processor.run_pipeline(video_path, video_output_dir, visualize=visualize)
    except RuntimeError as e:
        if 'CUDA' in str(e):
            # CUDA errors corrupt the entire GPU context — this worker must die.
            # maxtasksperchild in Pool will spawn a fresh worker automatically.
            import traceback
            print(f"[FATAL CUDA] Worker {os.getpid()} hit CUDA error on {video_path}")
            traceback.print_exc()
            # Force-kill this worker so Pool replaces it with a clean one
            os._exit(1)
        import traceback
        print(f"⚠️ Video processing failed: {video_path}")
        print(f"Error: {e}")
        traceback.print_exc()
        return False, "Skipped due to RuntimeError"
    except Exception as e:
        import traceback
        print(f"⚠️ Video processing failed: {video_path}")
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

    saved_lma_count = 0
    if fragments:
        print(f"\n[+] WHAM Complete for {video_name}. Starting LMA Integration...")
        for _id, data in fragments.items():
            print(f"[*] Extracting LMA features for Fragment {_id}...")
            
            joints = data['joints_world'][:, :24, :]
            verts_array = data['verts_world']
            
            # A. Calculate Volumes
            volumes = []
            last_v = 0.07 
            for verts in verts_array:
                try:
                    v = ConvexHull(verts).volume
                    volumes.append(v)
                    last_v = v
                except Exception:
                    volumes.append(last_v)
            
            floors = [IdentityFloor()] * len(joints)
            
            # B. Call LMA logic
            try:
                lma_dict, lma_matrix = compute_lma_descriptor(
                    joints=joints, 
                    volumes=volumes, 
                    floors=floors, 
                    fps=fps, 
                    window_size=55
                )
                
                # C. Save results
                np.save(osp.join(video_output_dir, f"lma_features_id{_id}.npy"), lma_matrix)
                np.save(osp.join(video_output_dir, f"lma_dict_id{_id}.npy"), lma_dict)
                summary['written_files'].append(f"lma_features_id{_id}.npy")
                summary['written_files'].append(f"lma_dict_id{_id}.npy")
                saved_lma_count += 1
            except Exception as exc:
                print(f"[!] Failed to compute/save LMA for Fragment {_id}: {exc}")

    if saved_lma_count == 0:
        summary['reason'] = 'No LMA outputs written'
        summary_path = osp.join(video_output_dir, 'summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        return False, f"No LMA outputs written: {video_path}"

    summary['status'] = 'success'
    summary['reason'] = ''
    summary_path = osp.join(video_output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    return True, video_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output/wham_kineguard")
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