import sys
import time
import cv2
import math
import torch
import numpy as np
from omegaconf import OmegaConf
from hydra import initialize, compose # NEW: Added Hydra for config stitching

# Import the full MASt3R-Nav stack
from libs.mapper.create_topomap import CostmapData
from libs.matcher.mast3r_matcher import Mast3rMatcher
from libs.localizer.loc_topo import LocalizeTopological
from libs.planner.plan_topo import PlanTopological
from libs.goal_generator.goal_gen import GoalGenerator
from libs.common.utils_sim import build_intrinsics
from libs.control.learnt_controller import ObjRelLearntController, visualize_prediction

def main():
    print("Loading configurations...")
    
    # --- 1. PROPERLY LOAD CONFIGS WITH HYDRA ---
    with initialize(version_base=None, config_path="configs"):
        cfg = compose(config_name="config")
        
    ctrl_cfg = OmegaConf.load("configs/controller/gnm_waypixel.yaml")
    ctrl_cfg.load_run = "checkpoints/gnm_mast3r_nav"
    
    # --- 2. INITIALIZE MAPPING & LOCALIZATION STACK ---
    print("Loading Map Data...")
    # UPDATE THIS PATH to your exact generated .npz costmap file
    costmap_path = "data/maps/traj_000/costmaps_320x240_EC_NONE_NC_NONE_NCF_10.npz" 
    costmap_data = CostmapData.from_npz(costmap_path)
    map_img_paths = costmap_data.get_metadata()['image_paths']

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Initializing MASt3R Matcher...")
    matcher = Mast3rMatcher(
        resize_w=cfg.matcher.resize_w,
        resize_h=cfg.matcher.resize_h,
        geometric_verification=cfg.matcher.geometric_verification,
        subsample_or_initxy1=cfg.matcher.subsample_or_initxy1,
        device=device
    )

    print("Initializing Localizer...")
    localizer = LocalizeTopological(
        map_img_paths=map_img_paths,
        H=cfg.matcher.resize_h,
        W=cfg.matcher.resize_w,
        matcher=matcher,
        cfg=cfg.localizer
    )

    print("Initializing Planner...")
    planner = PlanTopological(
        H=cfg.matcher.resize_h,
        W=cfg.matcher.resize_w,
        costmap_data=costmap_data,
        device=device,
        cfg=cfg.planner
    )

    print("Initializing Goal Generator...")
    goal_generator = GoalGenerator(
        H=cfg.matcher.resize_h,
        W=cfg.matcher.resize_w,
        localizer=localizer,
        planner=planner,
        cfg=cfg
    )

    # The goal generator mathematically requires knowing the camera's field of view
    intrinsics = build_intrinsics(
        image_width=cfg.matcher.resize_w,
        image_height=cfg.matcher.resize_h,
        field_of_view_radians_u=math.radians(79), # 79 degrees is standard webcam FOV
        device=device
    )

    # --- 3. INITIALIZE NEURAL CONTROLLER ---
    print("Initializing Controller (PixelReact)...")
    controller = ObjRelLearntController(
        config=OmegaConf.to_container(ctrl_cfg, resolve=True),
        goal_source=cfg.goal_source,
        boost_final_goal=cfg.get("boost_final_goal", False)
    )

    # --- 4. START REAL WORLD LOOP ---
    print("Opening USB Camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        sys.exit(1)

    print("\n--- Starting Live Navigation Loop ---")
    print("Press 'q' in the video window to quit.")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
                
            # Convert BGR (OpenCV) to RGB and resize to match what the matcher expects (e.g., 320x240)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (cfg.matcher.resize_w, cfg.matcher.resize_h))

            t_start = time.time()

            # Localize and generate the dense WayPixel Costmap
            costmap = goal_generator.get_goal_mask(
                qry_img=rgb,
                qry_depth=None, # Pure RGB matching, no depth needed
                qry_pts3d=None,
                intrinsics=intrinsics,
                candidate_img_indices=None, # Lets the localizer figure out where you are automatically
                return_vis_data=False
            )
            
            if costmap is None:
                print("Could not localize... make sure camera sees a mapped area!")
                continue

            # Predict Trajectory & Waypoints
            v, w, _ = controller.predict(rgb, costmap)
            
            # Draw Trajectory 
            vis_img = visualize_prediction(
                rgb=rgb,
                pred_waypoints=controller.action_pred, 
                goal_mask_vis=controller.goal_mask_vis, 
                get_plot_img=True 
            )
            
            # Display Window
            if vis_img is not None:
                bgr_vis = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
                fps = 1.0 / (time.time() - t_start)
                cv2.putText(bgr_vis, f"Cmd V: {v:.2f} m/s | W: {w:.2f} rad/s | {fps:.1f} FPS", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("MASt3R-Nav Live Tracking", bgr_vis)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nManually interrupted.")
    finally:
        print("Cleaning up...")
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()