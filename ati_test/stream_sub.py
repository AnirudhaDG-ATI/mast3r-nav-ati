"""
server.py — Runs on your server/workstation.

- Connects a SUB socket to the Jetson's frame PUB socket, displays the feed
  with cv2.imshow.
- Connects a PUB socket to the Jetson's command SUB socket, sends (v, w)
  commands back.

You only need to set JETSON_IP below (Jetson has a static IP and binds
both sockets; the server just connects to it).

Requires: pip install pyzmq opencv-python numpy
"""

from email import header

import zmq
import cv2
import time
import json
import argparse
import numpy as np

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

FRAME_PORT = 3333
COMMAND_PORT = 3334

COMMAND_HZ = 10  # how often we send velocity commands

# --------------------------------------------------------------------------
# Placeholder control policy — replace with your mast3r-nav pipeline
# --------------------------------------------------------------------------

def compute_velocity_command(frame):
    """
    TODO: Replace this with a call into your actual navigation pipeline,
    e.g. episode.get_goal(...) + episode.goal_controller.predict(...)
    to get (v, w) from the frame instead of this placeholder.
    """
    v, w = 0.0, 0.0
    return v, w


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jetson-ip", type=str, required=True,
                         help="Static IP of the Jetson, e.g. 192.168.1.50")
    parser.add_argument("--manual", action="store_true",
                         help="Manual keyboard control (WASD) instead of the "
                              "compute_velocity_command placeholder")
    args = parser.parse_args()

    context = zmq.Context()

    # SUB socket for frames — connects to Jetson's PUB bind
    frame_sub = context.socket(zmq.SUB)
    #frame_sub.setsockopt(zmq.CONFLATE, 1)  # always show only the latest frame
    frame_sub.setsockopt_string(zmq.SUBSCRIBE, "")
    frame_sub.connect(f"tcp://{args.jetson_ip}:{FRAME_PORT}")

    # NOTE: CONFLATE only works cleanly with single-part messages, but we're
    # sending multipart [header, jpeg]. To keep CONFLATE's "latest only"
    # behavior, poll with a short timeout and drain the socket each loop
    # (see recv loop below) instead of relying purely on CONFLATE.
    frame_sub.setsockopt(zmq.RCVTIMEO, 200)  # ms

    # PUB socket for commands — connects to Jetson's SUB bind
    command_pub = context.socket(zmq.PUB)
    command_pub.setsockopt(zmq.SNDHWM, 1)
    command_pub.connect(f"tcp://{args.jetson_ip}:{COMMAND_PORT}")

    print(f"[Server] Frame SUB connected to tcp://{args.jetson_ip}:{FRAME_PORT}")
    print(f"[Server] Command PUB connected to tcp://{args.jetson_ip}:{COMMAND_PORT}")
    print("[Server] Press 'q' in the video window to quit.")

    cv2.namedWindow("Jetson Feed", cv2.WINDOW_NORMAL)

    last_command_time = 0.0
    command_period = 1.0 / COMMAND_HZ
    v, w = 1.0, 0.0
    
    stats_last_print_time = time.time()
    frames_received_in_window = 0
    total_lost_frames = 0
    expected_frame_id = -1

    try:
        while True:
            # --- Drain the frame socket, keep only the newest frame ---
            latest_msg = None
            while True:
                try:
                    latest_msg = frame_sub.recv_multipart(flags=zmq.NOBLOCK)
                except zmq.Again:
                    break

            if latest_msg is None:
                # No frame yet this iteration; brief wait then loop again
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            header_bytes, jpg_bytes = latest_msg
            header = json.loads(header_bytes.decode("utf-8"))

            jpg_arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
            frame = cv2.imdecode(jpg_arr, cv2.IMREAD_COLOR)

            current_frame_id = header["frame_id"]
            resolution = header.get("shape", "Unknown")
            latency_ms = (time.time() - header["timestamp"]) * 1000

            # 1. Detect lost packets by checking for skipped frame IDs
            if expected_frame_id != -1 and current_frame_id > expected_frame_id:
                dropped = current_frame_id - expected_frame_id
                total_lost_frames += dropped
            
            expected_frame_id = current_frame_id + 1
            frames_received_in_window += 1

            # 2. Print metrics to CLI every 1 second
            current_time = time.time()
            if current_time - stats_last_print_time >= 1.0:
                fps = frames_received_in_window / (current_time - stats_last_print_time)
                
                # Using \r and flush=True overwrites the same line in the terminal
                print(f"\r[Stream Stats] FPS: {fps:.1f} | Latency: {latency_ms:.0f}ms | Lost Pkts: {total_lost_frames} | Res: {resolution[1]}x{resolution[0]}", end="", flush=True)
                
                # Reset window
                stats_last_print_time = current_time
                frames_received_in_window = 0

            # latency = time.time() - header["timestamp"]
            # cv2.putText(frame, f"frame {header['frame_id']}  latency {latency*1000:.0f}ms",
            #             (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow("Jetson Feed", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            # --- Compute / update velocity command ---
            if args.manual:
                v, w = manual_control_from_key(key, v, w)
            else:
                v, w = compute_velocity_command(frame)

            # --- Send command at fixed rate ---
            now = time.time()
            if now - last_command_time >= command_period:
                command_pub.send_string(json.dumps({
                    "v": v,
                    "w": w,
                    "timestamp": now,
                }))
                last_command_time = now

    except KeyboardInterrupt:
        print("\n[Server] Shutting down...")
    finally:
        # send a final stop command before exiting
        try:
            command_pub.send_string(json.dumps({"v": 0.0, "w": 0.0, "timestamp": time.time()}))
        except Exception:
            pass
        cv2.destroyAllWindows()
        frame_sub.close()
        command_pub.close()
        context.term()


def manual_control_from_key(key, v, w, v_step=0.02, w_step=0.05, v_max=0.2, w_max=0.5):
    """Simple WASD manual override for testing the pipeline end to end."""
    if key == ord('w'):
        v = min(v + v_step, v_max)
    elif key == ord('s'):
        v = max(v - v_step, -v_max)
    elif key == ord('a'):
        w = max(w - w_step, -w_max)
    elif key == ord('d'):
        w = min(w + w_step, w_max)
    elif key == ord(' '):
        v, w = 0.0, 0.0
    return v, w


if __name__ == "__main__":
    main()
