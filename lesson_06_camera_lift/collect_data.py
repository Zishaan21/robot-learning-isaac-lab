"""Lesson 06, step 1: collect labelled front-camera images from the real lift scene. See lesson.md.

The trained lesson-03 policy (exact cube position) drives the arms, so the images show what the
camera will really see: cube on the table, cube in the gripper, cube lifted, arm in front of it.

Run from Learning\\Robotics:
    .\\run.bat lesson_06_camera_lift\\collect_data.py --headless
Output: lesson_06_camera_lift\\data\\lift_camera_dataset.npz and data\\label_check.png
"""
import h5py  # noqa: F401  (load before Isaac Sim - see isaaclab_run.py)

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=50)
parser.add_argument("--samples", type=int, default=4000, help="images to keep")
parser.add_argument("--every", type=int, default=6, help="record every N policy steps")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
simulation_app = AppLauncher(args).app

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from camera_lift import (  # noqa: E402
    CAM_H, CAM_W, FrankaCubeLiftCameraEnvCfg, camera_images, load_policy, object_world, sample_depth, world_to_pixel,
)

OUT_DIR = os.path.join(HERE, "data")
os.makedirs(OUT_DIR, exist_ok=True)
MARGIN = 3  # px

env_cfg = FrankaCubeLiftCameraEnvCfg()
env_cfg.scene.num_envs = args.num_envs
env = ManagerBasedRLEnv(cfg=env_cfg)
wrapped = RslRlVecEnvWrapper(env)
policy = load_policy(wrapped, env.device)
cam = env.scene["front_camera"]

rgbs, depths, pixels, centre_depths, surface_depths = [], [], [], [], []
obs = wrapped.get_observations()
step = 0
n = 0  # images collected so far
while n < args.samples and simulation_app.is_running():
    with torch.inference_mode():
        obs, _, _, _ = wrapped.step(policy(obs))
    step += 1
    if step < 5 or step % args.every:  # skip the first frames (renderer warm-up)
        continue
    with torch.inference_mode():
        rgb, depth = camera_images(cam)
        uv, zc = world_to_pixel(cam, object_world(env))  # labels: exact, from the simulator
        zs = sample_depth(depth, uv)  # what the depth camera measures at that pixel
    inside = (uv[:, 0] >= MARGIN) & (uv[:, 0] < CAM_W - MARGIN) & (uv[:, 1] >= MARGIN) & (uv[:, 1] < CAM_H - MARGIN)
    keep = inside.nonzero().squeeze(-1)
    rgbs.append(rgb[keep].cpu().numpy())
    depths.append(depth[keep].cpu().numpy().astype(np.float16))
    pixels.append(uv[keep].cpu().numpy())
    centre_depths.append(zc[keep].cpu().numpy())
    surface_depths.append(zs[keep].cpu().numpy())
    n += len(keep)
    if step % (args.every * 20) == 0:
        print(f"step {step}: {n}/{args.samples} images")

rgb = np.concatenate(rgbs)[: args.samples]
depth = np.concatenate(depths)[: args.samples]
pixel = np.concatenate(pixels)[: args.samples].astype(np.float32)
zc = np.concatenate(centre_depths)[: args.samples].astype(np.float32)
zs = np.concatenate(surface_depths)[: args.samples].astype(np.float32)

# Label sanity check (lesson 05!): does the depth camera see something at the labelled pixel about
# half a cube in front of the cube centre? If images lagged the labels, this would often be way off.
gap = zc - zs
print(f"\nlabel check: centre depth - measured depth at label pixel: median {np.median(gap) * 100:.1f} cm, "
      f"within 0-6 cm for {np.mean((gap > 0) & (gap < 0.06)) * 100:.0f}% of images")

K = cam.data.intrinsic_matrices[0].cpu().numpy()  # same lens for every env's camera
np.savez_compressed(os.path.join(OUT_DIR, "lift_camera_dataset.npz"), rgb=rgb, depth=depth, pixel=pixel,
                    centre_depth=zc, surface_depth=zs, K=K)

# draw labels on a few images - always look at your data
idx = np.linspace(0, len(rgb) - 1, 12).astype(int)
fig, axes = plt.subplots(2, 6, figsize=(16, 4.4))
for ax, i in zip(axes.flat, idx):
    ax.imshow(rgb[i])
    ax.plot(*pixel[i], "o", mfc="none", mec="lime", ms=12, mew=1.5)
    ax.set_title(f"#{i}  gap {gap[i] * 100:.1f} cm", fontsize=8)
    ax.axis("off")
fig.suptitle("green o = labelled cube centre (from the simulator)")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "label_check.png"), dpi=90)
print(f"DONE - saved {len(rgb)} images to data\\lift_camera_dataset.npz and data\\label_check.png")

env.close()
simulation_app.close()
