"""
server_inference.py — Real-time MASt3R-Nav closed-loop control server.
"""

import os
import sys
import cv2
import zmq
import time
import json
import math
import torch
import argparse
import numpy as np

# MASt3R-Nav imports
from omegaconf import OmegaConf
from hydra import initialize, compose
import torchvision.transforms as tfm
from PIL import Image

from libs.mapper.create_topomap import CostmapData
from libs.matcher.mast3r_matcher import Mast3rMatcher
from libs.localizer.loc_topo import LocalizeTopological
from libs.planner.plan_topo import PlanTopological
from libs.goal_generator.goal_gen import GoalGenerator
from libs.common.utils_sim import build_intrinsics
from libs.control.learnt_controller import ObjRelLearntController
from dust3r.inference import inference

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
FRAME_PORT = 3333
COMMAND_PORT = 3334
COMMAND_HZ = 10 

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

def get_opencv_costmap(costmap):
    """Converts the raw float costmap into a fast, colored OpenCV image."""
    clean_costmap = np.array(costmap, dtype=np.float32)
    valid_mask = clean_costmap < 99.0
    
    if not np.any(valid_mask):
        return np.full((clean_costmap.shape[0], clean_costmap.shape[1], 3), 255, dtype=np.uint8)

    vmin, vmax = np.nanmin(clean_costmap[valid_mask]), np.nanmax(clean_costmap[valid_mask])
    
    # Normalize to 0-255
    norm_costmap = np.zeros_like(clean_costmap, dtype=np.uint8)
    norm_costmap[valid_mask] = ((clean_costmap[valid_mask] - vmin) / (vmax - vmin + 1e-5) * 255).astype(np.uint8)
    
    # Apply Turbo colormap
    colored_costmap = cv2.applyColorMap(norm_costmap, cv2.COLORMAP_TURBO)
    
    # Paint unreachable/bad areas white
    colored_costmap[~valid_mask] = [255, 255, 255]
    return colored_costmap

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jetson-ip", type=str, required=True, help="Jetson IP")
    args = parser.parse_args()

    # --------------------------------------------------------------------------
    # 1. Initialize MASt3R-Nav Pipeline
    # --------------------------------------------------------------------------
    print("[Server] Loading MASt3R configs...")
    with initialize(version_base=None, config_path="configs"):
        cfg = compose(config_name="config")

    ctrl_cfg = OmegaConf.load("configs/controller/gnm_waypixel.yaml")
    ctrl_cfg.load_run = "checkpoints/gnm_mast3r_nav"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    img_w, img_h = int(cfg.matcher.resize_w), int(cfg.matcher.resize_h)

    print("[Server] Loading topological map...")
    costmap_path = "data/maps/traj_000/costmaps_320x240_EC_NONE_NC_NONE_NCF_10.npz"
    costmap_data = CostmapData.from_npz(costmap_path)
    map_img_paths = costmap_data.get_metadata()["image_paths"]

    print("[Server] Booting Neural Networks into VRAM (This takes a moment)...")
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

    # --------------------------------------------------------------------------
    # 2. Setup ZeroMQ Network
    # --------------------------------------------------------------------------
    context = zmq.Context()

    frame_sub = context.socket(zmq.SUB)
    frame_sub.setsockopt_string(zmq.SUBSCRIBE, "")
    frame_sub.setsockopt(zmq.RCVTIMEO, 200)
    frame_sub.connect(f"tcp://{args.jetson_ip}:{FRAME_PORT}")

    command_pub = context.socket(zmq.PUB)
    command_pub.setsockopt(zmq.SNDHWM, 1)
    command_pub.connect(f"tcp://{args.jetson_ip}:{COMMAND_PORT}")

    print(f"\n[Server] Connected! Ready to process stream.")
    cv2.namedWindow("Jetson Feed", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Live WayPixel Costmap", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("Localization Match", cv2.WINDOW_NORMAL)

    # Telemetry vars
    last_command_time = 0.0
    command_period = 1.0 / COMMAND_HZ
    stats_last_print = time.time()
    frames_in_window = 0
    total_lost = 0
    expected_id = -1
    v, w = 0.0, 0.0

    try:
        while True:
            # Drain socket to get absolute latest frame
            latest_msg = None
            while True:
                try:
                    latest_msg = frame_sub.recv_multipart(flags=zmq.NOBLOCK)
                except zmq.Again:
                    break

            if latest_msg is None:
                if cv2.waitKey(1) & 0xFF == ord('q'): break
                continue

            # Decode Network Data
            header = json.loads(latest_msg[0].decode("utf-8"))
            jpg_arr = np.frombuffer(latest_msg[1], dtype=np.uint8)
            frame = cv2.imdecode(jpg_arr, cv2.IMREAD_COLOR)

         
            # if vis_data:
            #         # Depending on your specific repo branch, this key might be "node_idx", 
            #         # "matched_node_idx", or "best_match". We safely fall back to grab the integer.
            #         match_idx = vis_data.get("matched_node_idx", vis_data.get("node_idx", -1))
                    
            #         if match_idx != -1 and 0 <= match_idx < len(map_img_paths):
            #             # Load the historical map image the robot matched with
            #             ref_img_path = map_img_paths[match_idx]
            #             ref_img = cv2.imread(ref_img_path)
                        
            #             if ref_img is not None:
            #                 # Label it and show it
            #                 cv2.putText(ref_img, f"Map Node: {match_idx}", (10, 30), 
            #                             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            #                 cv2.imshow("Localization Match", ref_img)   # --- Telemetry Updates ---
            current_id = header["frame_id"]
            if expected_id != -1 and current_id > expected_id:
                total_lost += (current_id - expected_id)
            expected_id = current_id + 1
            frames_in_window += 1
            net_latency = (time.time() - header["timestamp"]) * 1000

            # ------------------------------------------------------------------
            # 3. AI Inference (The Heavy Lifting)
            # ------------------------------------------------------------------
            inference_start = time.time()
            
            # Format image for model
            rgb = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (img_w, img_h))

            # Forward passes
            qry_pts3d = get_pts3d_direct(matcher, rgb, device)
            
            # Note: return_vis_data=True so we can pull localization info
            costmap, vis_data = goal_generator.get_goal_mask(
                qry_img=rgb, qry_depth=None, qry_pts3d=qry_pts3d, intrinsics=intrinsics,
                candidate_img_indices=None, return_vis_data=True
            )
            
            if costmap is not None:
                v, w, _ = controller.predict(rgb, costmap)
                
                # Render Costmap
                viz_costmap = get_opencv_costmap(costmap)
                cv2.imshow("Live WayPixel Costmap", viz_costmap)


            # if vis_data:
            #         # Depending on your specific repo branch, this key might be "node_idx", 
            #         # "matched_node_idx", or "best_match". We safely fall back to grab the integer.
            #         match_idx = vis_data.get("matched_node_idx", vis_data.get("node_idx", -1))
                    
            #         if match_idx != -1 and 0 <= match_idx < len(map_img_paths):
            #             # Load the historical map image the robot matched with
            #             ref_img_path = map_img_paths[match_idx]
            #             ref_img = cv2.imread(ref_img_path)
                        
            #             if ref_img is not None:
            #                 # Label it and show it
            #                 cv2.putText(ref_img, f"Map Node: {match_idx}", (10, 30), 
            #                             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            #                 cv2.imshow("Localization Match", ref_img)

            else:
                v, w = 0.0, 0.0 # Stop if localization fails

            inf_time = (time.time() - inference_start) * 1000
            
            # ------------------------------------------------------------------
            # 4. Visualization & Commands
            # ------------------------------------------------------------------
            # CLI Telemetry
            now = time.time()
            if now - stats_last_print >= 1.0:
                fps = frames_in_window / (now - stats_last_print)
                matched_node = vis_data.get("matched_node_idx", "N/A") if vis_data else "Failed"
                
                print(f"\r[Stats] Server FPS: {fps:.1f} | Net Latency: {net_latency:.0f}ms | Inf Time: {inf_time:.0f}ms | Lost: {total_lost} | Loc Node: {matched_node}", end="", flush=True)
                stats_last_print = now
                frames_in_window = 0

            # UI Text
            cv2.putText(frame, f"v: {v:.2f} w: {w:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Inf: {inf_time:.0f}ms", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
            cv2.imshow("Jetson Feed", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Send Command (throttled to COMMAND_HZ)
            if now - last_command_time >= command_period:
                command_pub.send_string(json.dumps({"v": float(v), "w": float(w), "timestamp": now}))
                last_command_time = now

    except KeyboardInterrupt:
        print("\n[Server] Shutting down...")
    finally:
        try: command_pub.send_string(json.dumps({"v": 0.0, "w": 0.0, "timestamp": time.time()}))
        except: pass
        cv2.destroyAllWindows()
        frame_sub.close()
        command_pub.close()
        context.term()

if __name__ == "__main__":
    main()