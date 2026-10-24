"""Lesson 08, step 1: inspect the two 6-joint arms we could use. See lesson.md.

Spawns each arm at the lift task's table (same layout as lesson 03), holds its default pose for 2 s,
then prints its joints, bodies and where its hand ends up, and saves a picture.

Run from Learning\\Robotics:
    .\\run.bat lesson_08_six_joint_arm\\inspect_arms.py --headless
Output: lesson_08_six_joint_arm\\output\\inspect_<arm>.png
"""
import h5py  # noqa: F401  (load before Isaac Sim - see isaaclab_run.py)

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--arms", nargs="+", default=["kinova_j2n6s300", "ur10e_robotiq"],
                    help="which arms to inspect (first run downloads each robot model)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
simulation_app = AppLauncher(args).app

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import isaacsim.core.utils.prims as prim_utils
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_assets import KINOVA_JACO2_N6S300_CFG, UR10e_ROBOTIQ_GRIPPER_CFG

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
os.makedirs(OUT, exist_ok=True)

ALL_ARMS = {"ur10e_robotiq": UR10e_ROBOTIQ_GRIPPER_CFG, "kinova_j2n6s300": KINOVA_JACO2_N6S300_CFG}
ARMS = {name: ALL_ARMS[name] for name in args.arms}
SPACING = 3.0

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device=args.device))
sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg(), translation=(0, 0, -1.05))
sim_utils.DomeLightCfg(intensity=3000.0).func("/World/light", sim_utils.DomeLightCfg(intensity=3000.0))

robots, cams = {}, {}
for i, (name, cfg) in enumerate(ARMS.items()):
    origin = (0.0, i * SPACING, 0.0)
    prim_utils.create_prim(f"/World/{name}", "Xform", translation=origin)
    # same table as the lift task (lift_env_cfg.py): top at z = 0, centred 0.5 m in front of the robot
    table = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")
    table.func(f"/World/{name}/Table", table, translation=(0.5, 0, 0), orientation=(0.707, 0, 0, 0.707))
    robots[name] = Articulation(cfg.replace(prim_path=f"/World/{name}/Robot"))
    cams[name] = Camera(CameraCfg(
        prim_path=f"/World/{name}/Cam", width=640, height=480, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.05, 20.0)),
    ))

sim.reset()
for i, name in enumerate(ARMS):
    # side view of each robot + table
    eye = torch.tensor([[1.4, i * SPACING - 1.6, 0.9]], device=sim.device)
    target = torch.tensor([[0.35, i * SPACING, 0.25]], device=sim.device)
    cams[name].set_world_poses_from_view(eye, target)

for _ in range(200):  # hold the default pose for 2 s - weak/heavy arms sag
    for robot in robots.values():
        robot.set_joint_position_target(robot.data.default_joint_pos)
        robot.write_data_to_sim()
    sim.step()
    for robot in robots.values():
        robot.update(sim.get_physics_dt())
    for cam in cams.values():
        cam.update(sim.get_physics_dt())

for i, (name, robot) in enumerate(robots.items()):
    base = robot.data.root_pos_w[0]
    print(f"\n=================== {name} ===================")
    print(f"joints ({robot.num_joints}): {robot.joint_names}")
    print(f"actuated joint groups: { {k: v.joint_names for k, v in robot.actuators.items()} }")
    print(f"bodies ({robot.num_bodies}): {robot.body_names}")
    print("body positions relative to the robot base (m), default pose after 2 s:")
    for b, bname in enumerate(robot.body_names):
        p = (robot.data.body_pos_w[0, b] - base).cpu().numpy()
        print(f"   {bname:35s} x={p[0]:6.3f} y={p[1]:6.3f} z={p[2]:6.3f}")
    err = (robot.data.joint_pos[0] - robot.data.default_joint_pos[0]).abs().max().item()
    print(f"largest joint deviation from the default pose after 2 s: {err:.3f} rad (sag / weak gains)")

    img = cams[name].data.output["rgb"][0, ..., :3].cpu().numpy()
    plt.imsave(os.path.join(OUT, f"inspect_{name}.png"), img)
    print(f"saved output\\inspect_{name}.png")

simulation_app.close()
