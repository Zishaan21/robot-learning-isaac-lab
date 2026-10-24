"""Lesson 06, step 3: run the lesson-03 policy with EXACT vs CAMERA cube positions and compare. See lesson.md.

Run from Learning\\Robotics:
    .\\run.bat lesson_06_camera_lift\\evaluate.py --headless              # numbers + videos
    .\\run.bat lesson_06_camera_lift\\evaluate.py --num_envs 8            # also watch it live
    add --no_video to skip recording

Videos (env 0, both phases) go to lesson_06_camera_lift\\output\\videos\\:
    overview_exact.mp4 / overview_camera.mp4   - the arm, seen from the side
    camera_exact.mp4   / camera_camera.mp4     - what the front camera sees; green ring = true cube
                                                 centre, red cross = the network's estimate
"""
import h5py  # noqa: F401  (load before Isaac Sim - see isaaclab_run.py)

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--episodes", type=int, default=3, help="episodes per mode (5 s each)")
parser.add_argument("--no_video", action="store_true", help="don't record videos")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
simulation_app = AppLauncher(args).app

import imageio.v2 as imageio
import numpy as np
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import camera_lift  # noqa: E402
from camera_lift import (  # noqa: E402
    FrankaCubeLiftCameraEnvCfg, camera_images, get_estimator, goal_world, load_policy, object_world,
    use_camera_for_object_position, world_to_pixel,
)

LIFT_HEIGHT = 0.04  # same threshold as the task's lifting reward
GOAL_RADIUS = 0.10  # "carried to the goal" = within 10 cm
RECORD = not args.no_video
VIDEO_DIR = os.path.join(HERE, "output", "videos")
VIDEO_EVERY = 2  # policy runs at 50 Hz; every 2nd step -> 25 fps, i.e. real time
CAM_SCALE = 4  # upscale the 160x120 camera image to 640x480 for the video

env_cfg = FrankaCubeLiftCameraEnvCfg()
env_cfg.scene.num_envs = args.num_envs
use_camera_for_object_position(env_cfg)
# overview video: follow env 0 from the front-left side
env_cfg.viewer.origin_type = "env"
env_cfg.viewer.env_index = 0
env_cfg.viewer.eye = (1.6, 1.4, 0.9)
env_cfg.viewer.lookat = (0.4, 0.0, 0.2)
env_cfg.viewer.resolution = (960, 540)
env = ManagerBasedRLEnv(cfg=env_cfg, render_mode="rgb_array" if RECORD else None)
wrapped = RslRlVecEnvWrapper(env)
policy = load_policy(wrapped, env.device)
steps_per_episode = int(env.max_episode_length)
cam = env.scene["front_camera"]


def draw_ring(img, u, v, colour, r=10, thick=2):
    h, w = img.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    d = np.sqrt((xx - u) ** 2 + (yy - v) ** 2)
    img[(d >= r - thick) & (d <= r)] = colour


def draw_cross(img, u, v, colour, size=8, thick=2):
    u, v = int(round(u)), int(round(v))
    h, w = img.shape[:2]
    for k in range(-size, size + 1):
        for t in range(-thick // 2, thick // 2 + 1):
            for y, x in ((v + k, u + k + t), (v + k, u - k + t)):
                if 0 <= y < h and 0 <= x < w:
                    img[y, x] = colour


def camera_frame():
    """Env 0's front-camera image, upscaled, with the true (green) and estimated (red) cube centre."""
    rgb, _ = camera_images(cam)
    img = np.repeat(np.repeat(rgb[0].cpu().numpy(), CAM_SCALE, axis=0), CAM_SCALE, axis=1).copy()
    true_uv, _ = world_to_pixel(cam, object_world(env))
    draw_ring(img, *(true_uv[0].cpu().numpy() * CAM_SCALE), colour=(0, 255, 0))
    est = get_estimator(env.device).last_estimate_w
    if est is not None:
        est_uv, _ = world_to_pixel(cam, est)
        draw_cross(img, *(est_uv[0].cpu().numpy() * CAM_SCALE), colour=(255, 0, 0))
    return img


if RECORD:
    os.makedirs(VIDEO_DIR, exist_ok=True)

results = {}
for mode in ("exact", "camera"):
    camera_lift.MODE["use_camera"] = mode == "camera"
    lifted, goal_dist, est_err = [], [], []
    writers = {}
    if RECORD:
        fps = round(1 / (env.step_dt * VIDEO_EVERY))
        writers = {name: imageio.get_writer(os.path.join(VIDEO_DIR, f"{name}_{mode}.mp4"), fps=fps,
                                            macro_block_size=8)
                   for name in ("overview", "camera")}
    # everything (incl. reset) in inference mode: tensors created there can't be modified outside it
    with torch.inference_mode():
        obs, _ = wrapped.reset()
        for ep in range(args.episodes):
            for t in range(steps_per_episode):
                obs, _, _, _ = wrapped.step(policy(obs))
                obj = object_world(env)
                if t >= steps_per_episode // 2:  # judge the second half of each episode
                    lifted.append((obj[:, 2] > LIFT_HEIGHT).float().mean().item())
                    goal_dist.append(torch.norm(obj - goal_world(env), dim=1).cpu().numpy())
                est = get_estimator(env.device).last_estimate_w
                if est is not None and t > 5:
                    est_err.append(torch.norm(est - obj, dim=1).cpu().numpy())
                if RECORD and t % VIDEO_EVERY == 0:
                    writers["overview"].append_data(env.render())
                    writers["camera"].append_data(camera_frame())
            print(f"[{mode}] episode {ep + 1}/{args.episodes} done")
    for wtr in writers.values():
        wtr.close()
    if RECORD:
        print(f"[{mode}] videos saved: output\\videos\\overview_{mode}.mp4, output\\videos\\camera_{mode}.mp4")
    gd = np.concatenate(goal_dist)
    ee = np.concatenate(est_err)
    results[mode] = {
        "lifted": np.mean(lifted) * 100,
        "goal_dist": gd.mean() * 100,
        "at_goal": np.mean(gd < GOAL_RADIUS) * 100,
        "est_err": np.mean(ee) * 100,
        "est_err95": np.percentile(ee, 95) * 100,
    }

print("\n=== same policy, exact vs camera cube position (second half of each episode) ===")
print(f"{'':28s}{'exact':>10s}{'camera':>10s}")
rows = [("cube lifted (% of time)", "lifted", "{:9.1f}%"), ("cube-goal distance (cm)", "goal_dist", "{:10.1f}"),
        (f"within {GOAL_RADIUS * 100:.0f} cm of goal (%)", "at_goal", "{:9.1f}%"),
        ("camera estimate error (cm)", "est_err", "{:10.2f}"), ("  95th percentile (cm)", "est_err95", "{:10.2f}")]
for label, key, fmt in rows:
    print(f"{label:28s}" + "".join(fmt.format(results[m][key]) for m in ("exact", "camera")))

env.close()
simulation_app.close()
