import os
import sys
import cv2
import math
import time
import torch
import glob
import argparse
import numpy as np
import psutil
import gc

import torchvision.transforms as tfm
from PIL import Image
from omegaconf import OmegaConf
from hydra import initialize, compose
from natsort import natsorted

# Import MASt3R-Nav blocks
from libs.mapper.create_topomap import CostmapData
from libs.matcher.mast3r_matcher import Mast3rMatcher
from libs.localizer.loc_topo import LocalizeTopological
from libs.planner.plan_topo import PlanTopological
from libs.goal_generator.goal_gen import GoalGenerator
from libs.common.utils_sim import build_intrinsics
from libs.control.learnt_controller import ObjRelLearntController
from dust3r.inference import inference

def get_ram_usage():
    """Returns the current physical RAM (RSS) used by this process in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def get_pts3d_direct(matcher, rgb_img, device):
    """Memory-optimized forward pass extracting 3D points."""
    normalize = tfm.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    img_pil = Image.fromarray(rgb_img)
    img_tensor = tfm.ToTensor()(img_pil)
    img_tensor = tfm.Resize((matcher.resize_h, matcher.resize_w), antialias=True)(img_tensor)
    img_tensor = normalize(img_tensor).unsqueeze(0).to(device)
    
    img_pair = [
        {"img": img_tensor, "idx": 0, "instance": 0, "true_shape": np.int32([img_tensor.shape[-2:]])},
        {"img": img_tensor.clone(), "idx": 1, "instance": 1, "true_shape": np.int32([img_tensor.shape[-2:]])},
    ]
    with torch.no_grad():
        output = inference([tuple(img_pair)], matcher.model, device, batch_size=1, verbose=False)
    return output["pred1"]["pts3d"][0].cpu().numpy()

def main(input_dir):
    print("Loading configurations...")
    with initialize(version_base=None, config_path="configs"):
        cfg = compose(config_name="config")

    ctrl_cfg = OmegaConf.load("configs/controller/gnm_waypixel.yaml")
    ctrl_cfg.load_run = "checkpoints/gnm_mast3r_nav"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    img_w, img_h = int(cfg.matcher.resize_w), int(cfg.matcher.resize_h)

    print("Loading map...")
    costmap_path = "data/maps/traj_000/costmaps_320x240_EC_NONE_NC_NONE_NCF_10.npz"
    costmap_data = CostmapData.from_npz(costmap_path)
    map_img_paths = costmap_data.get_metadata()["image_paths"]

    print(f"Loading Models onto {device.upper()} (Base RAM: {get_ram_usage():.2f} MB)...")
    matcher = Mast3rMatcher(resize_w=img_w, resize_h=img_h, device=device)
    localizer = LocalizeTopological(map_img_paths, img_h, img_w, matcher, cfg.localizer)
    planner = PlanTopological(img_h, img_w, costmap_data, device, cfg.planner)
    goal_generator = GoalGenerator(img_h, img_w, localizer, planner, cfg)

    intrinsics = build_intrinsics(image_width=img_w, image_height=img_h, 
                                  field_of_view_radians_u=math.radians(79), device=device)

    controller = ObjRelLearntController(
        config=OmegaConf.to_container(ctrl_cfg, resolve=True),
        goal_source=cfg.goal_source,
        boost_final_goal=cfg.get("boost_final_goal", False),
    )

    # Prepare Image Feed
    if not os.path.isdir(input_dir):
        print(f"Error: {input_dir} is not a directory. Please provide a directory of images.")
        return

    image_files = []
    for ext in ('*.png', '*.jpg', '*.jpeg'):
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
    image_files = natsorted(image_files)

    print(f"\n--- Beginning 5 FPS Headless Inference on {len(image_files)} frames ---")
    
    TARGET_FPS = 5.0
    FRAME_TIME = 1.0 / TARGET_FPS
    dummy_depth = np.zeros((img_h, img_w), dtype=np.float32)

    for idx, img_path in enumerate(image_files):
        t_start = time.time()
        
        # 1. Read & Preprocess
        frame = cv2.imread(img_path)
        if frame is None: continue
        rgb = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (img_w, img_h))

        # 2. Pipeline Inference
        qry_pts3d = get_pts3d_direct(matcher, rgb, device)
        costmap = goal_generator.get_goal_mask(
            qry_img=rgb, qry_depth=dummy_depth, qry_pts3d=qry_pts3d, 
            intrinsics=intrinsics, candidate_img_indices=None, return_vis_data=False
        )

        v, w = 0.0, 0.0
        if costmap is not None:
            v, w, _ = controller.predict(rgb, costmap)

        # 3. Aggressive Memory Cleanup (Crucial for Jetson)
        del frame, rgb, qry_pts3d, costmap
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

        # 4. Measure & Print Metrics
        t_inference = time.time() - t_start
        ram_mb = get_ram_usage()
        
        # 5. Enforce 5 FPS (Sleep if inference was faster than 0.2s)
        sleep_time = max(0.0, FRAME_TIME - t_inference)
        time.sleep(sleep_time)
        
        t_total = time.time() - t_start
        actual_fps = 1.0 / t_total

        print(f"Frame {idx:04d} | RAM: {ram_mb:7.2f} MB | Inf Time: {t_inference:5.3f}s | FPS: {actual_fps:5.2f} | Cmd -> v: {v:5.2f}, w: {w:5.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str, help="Directory containing sequence of images")
    args = parser.parse_args()
    main(args.input_dir)