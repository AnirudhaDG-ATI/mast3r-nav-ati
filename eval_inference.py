import os
import sys
import cv2
import math
import torch
import glob
import argparse
import numpy as np

import matplotlib
matplotlib.use('Agg')  # CRITICAL: Prevents window pop-ups during batch processing
import matplotlib.pyplot as plt

import torchvision.transforms as tfm
from PIL import Image
from omegaconf import OmegaConf
from hydra import initialize, compose

# Import MASt3R-Nav blocks
from libs.mapper.create_topomap import CostmapData
from libs.matcher.mast3r_matcher import Mast3rMatcher
from libs.localizer.loc_topo import LocalizeTopological
from libs.planner.plan_topo import PlanTopological
from libs.goal_generator.goal_gen import GoalGenerator
from libs.common.utils_sim import build_intrinsics
from libs.control.learnt_controller import ObjRelLearntController
from dust3r.inference import inference

# Import notebook utilities
sys.path.append("notebooks")
import viz_utils

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

def process_single_image(image_path, out_dir, cfg, img_w, img_h, device, matcher, goal_generator, intrinsics, controller):
    """Processes a single image and saves the dashboard."""
    print(f"-> Processing: {os.path.basename(image_path)}")
    
    frame = cv2.imread(image_path)
    if frame is None: 
        print(f"   [Error] Could not read {image_path}. Skipping.")
        return

    rgb = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (img_w, img_h))

    # 1. Forward Pass
    qry_pts3d = get_pts3d_direct(matcher, rgb, device)

    # 2. Localize & Generate Costmap
    costmap = goal_generator.get_goal_mask(
        qry_img=rgb, qry_depth=None, qry_pts3d=qry_pts3d, intrinsics=intrinsics,
        candidate_img_indices=None, return_vis_data=False
    )
    if costmap is None: 
        print(f"   [Warning] Localization failed for {os.path.basename(image_path)}. Skipping.")
        return

    # 3. Predict Waypoints
    v, w, _ = controller.predict(rgb, costmap)

    # 4. Render Dashboard
    fig = plt.figure(figsize=(14, 12))
    fig.patch.set_facecolor('white')
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2])

    # Top: Given RGB input image spanning full width
    ax_top = fig.add_subplot(gs[0, :])
    ax_top.imshow(rgb)
    ax_top.set_title(f"Input: {os.path.basename(image_path)}", fontsize=14, pad=10)
    ax_top.axis('off')

    # Bottom Left: Predicted Trajectory Waypoints Map
    ax_left = fig.add_subplot(gs[1, 0])
    viz_utils.plot_traj(ax_left, controller.action_pred)
    ax_left.set_title(f"Predicted Waypoints (v:{v:.2f}, w:{w:.2f})", fontsize=14, pad=10)

    # Bottom Right: Dense WayPixel Costmap
    ax_right = fig.add_subplot(gs[1, 1])
    clean_costmap = np.array(costmap, dtype=np.float32)
    clean_costmap = np.where(clean_costmap >= 99.0, np.nan, clean_costmap)
    
    cmap_obj = plt.get_cmap('turbo').copy()
    cmap_obj.set_bad(color='white')
    
    vmin, vmax = np.nanmin(clean_costmap), np.nanmax(clean_costmap)
    im_cost = ax_right.imshow(clean_costmap, cmap=cmap_obj, vmin=vmin, vmax=vmax)
    ax_right.axis('off')
    ax_right.set_title(f"WayPixel Costmap", fontsize=14, pad=10)
    fig.colorbar(im_cost, ax=ax_right, fraction=0.046, pad=0.04, label='Distance Cost')

    plt.tight_layout()
    
    # Generate output filename
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    save_path = os.path.join(out_dir, f"{base_name}_viz.png")
    
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig) # CRITICAL: Prevents memory leak in loops!

def main(input_path):
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

    print("Loading Models (This takes a moment)...")
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

    # -------------------------------------------------------------------------
    # Batch Processing Logic
    # -------------------------------------------------------------------------
    out_dir = "inference_outputs"
    os.makedirs(out_dir, exist_ok=True)
    
    if os.path.isdir(input_path):
        # Find all images in the directory
        image_files = []
        for ext in ('*.png', '*.jpg', '*.jpeg'):
            image_files.extend(glob.glob(os.path.join(input_path, ext)))
            
        image_files = sorted(image_files) # Process in order
        print(f"\nFound {len(image_files)} images in directory. Outputting to '{out_dir}/'...")
        
        for img_path in image_files:
            process_single_image(img_path, out_dir, cfg, img_w, img_h, device, 
                                 matcher, goal_generator, intrinsics, controller)
    else:
        # Process single image
        print(f"\nProcessing single image. Outputting to '{out_dir}/'...")
        process_single_image(input_path, out_dir, cfg, img_w, img_h, device, 
                             matcher, goal_generator, intrinsics, controller)
        
    print("\n✅ All done!")