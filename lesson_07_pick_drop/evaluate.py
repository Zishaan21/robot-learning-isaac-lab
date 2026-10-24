"""Lesson 07: score a policy on pick-and-drop. See lesson.md.

Run from Learning\\Robotics:
    run.bat lesson_07_pick_drop\\evaluate.py --headless                          # v2 box task, newest checkpoint
    run.bat lesson_07_pick_drop\\evaluate.py --headless --variant table          # v1 table-spot task
    run.bat lesson_07_pick_drop\\evaluate.py --headless --checkpoint <path.pt>   # any checkpoint (e.g. lesson 03's)
    add --video to save output\\videos\\<variant>_<run>.mp4 (env 0)
"""
import h5py  # noqa: F401  (load before Isaac Sim - see isaaclab_run.py)

import argparse
import glob
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--variant", choices=["box", "table"], default="box")
parser.add_argument("--checkpoint", type=str, default=None, help="policy .pt (default: newest run of the variant)")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--episodes", type=int, default=3)
parser.add_argument("--video", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.video:
    args.enable_cameras = True
simulation_app = AppLauncher(args).app

import numpy as np
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

HERE = os.path.dirname(os.path.abspath(__file__))
ROBOTICS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import pick_drop_env_cfg as pd  # noqa: E402

if args.variant == "box":
    EnvCfg, RunnerCfg, experiment = pd.FrankaDropInBoxEnvCfg_PLAY, pd.DropInBoxPPORunnerCfg, "franka_drop_in_box"
    goal_reached, released_fn = pd._in_box, pd.released_in_box_reward
    goal_label = "cube in the box"
else:
    EnvCfg, RunnerCfg, experiment = pd.FrankaPickDropEnvCfg_PLAY, pd.PickDropPPORunnerCfg, "franka_pick_drop"
    goal_reached, released_fn = pd._placed, pd.released_reward
    goal_label = "cube placed on the spot"

if args.checkpoint is None:
    runs = sorted(glob.glob(os.path.join(ROBOTICS, "logs", "rsl_rl", experiment, "*")))
    runs = [r for r in runs if "smoketest" not in r]
    ckpts = sorted(glob.glob(os.path.join(runs[-1], "model_*.pt")), key=lambda p: int(p.split("_")[-1][:-3]))
    args.checkpoint = ckpts[-1]
print(f"[INFO] variant: {args.variant}   policy: {args.checkpoint}")

env_cfg = EnvCfg()
env_cfg.scene.num_envs = args.num_envs
if args.video:
    env_cfg.viewer.origin_type, env_cfg.viewer.env_index = "env", 0
    env_cfg.viewer.eye, env_cfg.viewer.lookat, env_cfg.viewer.resolution = (1.6, 1.4, 0.9), (0.45, 0, 0.1), (960, 540)
env = ManagerBasedRLEnv(cfg=env_cfg, render_mode="rgb_array" if args.video else None)
wrapped = RslRlVecEnvWrapper(env)
runner = OnPolicyRunner(wrapped, RunnerCfg().to_dict(), log_dir=None, device=env.device)
runner.load(args.checkpoint)
policy = runner.get_inference_policy(device=env.device)
T = int(env.max_episode_length)
# outside the reward manager we must resolve the finger joints ourselves
gripper_cfg = SceneEntityCfg("robot", joint_names=["panda_finger.*"])
gripper_cfg.resolve(env.scene)

writer = None
if args.video:
    import imageio.v2 as imageio

    os.makedirs(os.path.join(HERE, "output", "videos"), exist_ok=True)
    name = os.path.basename(os.path.dirname(args.checkpoint))
    writer = imageio.get_writer(os.path.join(HERE, "output", "videos", f"{args.variant}_{name}.mp4"),
                                fps=round(1 / (2 * env.step_dt)), macro_block_size=8)

carried, reached, released, final_dist, final_height = [], [], [], [], []
with torch.inference_mode():
    obs, _ = wrapped.reset()
    for ep in range(args.episodes):
        ever_carried = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        for t in range(T):
            obs, _, _, _ = wrapped.step(policy(obs))
            ever_carried |= env.scene["object"].data.root_pos_w[:, 2] > pd.CARRIED_HEIGHT
            if writer is not None and t % 2 == 0:
                writer.append_data(env.render())
            if t == T - 2:  # judge the state at the end of the episode (the env resets after the last step)
                carried.append(ever_carried.float().mean().item())
                reached.append(goal_reached(env, "object_pose").float().mean().item())
                released.append(released_fn(env, "object_pose", gripper_cfg).mean().item())
                final_dist.append(pd._goal_xy_dist(env, "object_pose").cpu().numpy())
                final_height.append(env.scene["object"].data.root_pos_w[:, 2].cpu().numpy())
        print(f"episode {ep + 1}/{args.episodes} done")
if writer is not None:
    writer.close()

d = np.concatenate(final_dist) * 100
z = np.concatenate(final_height) * 100
print(f"\n=== {args.variant} task, state at the end of each episode ===")
print(f"cube was carried (> {pd.CARRIED_HEIGHT * 100:.0f} cm)       : {np.mean(carried) * 100:5.1f}%")
print(f"{goal_label:32s}: {np.mean(reached) * 100:5.1f}%   (after being carried)")
print(f"... and released (SUCCESS)       : {np.mean(released) * 100:5.1f}%   (gripper open, hand > 5 cm above)")
print(f"cube-to-target distance          : median {np.median(d):.1f} cm (horizontal)")
print(f"cube height (centre)             : median {np.median(z):.1f} cm, 10-90%: {np.percentile(z, 10):.1f}-"
      f"{np.percentile(z, 90):.1f} cm   (resting on the table = ~2.4 cm)")
env.close()
simulation_app.close()
