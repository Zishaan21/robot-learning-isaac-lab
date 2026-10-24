"""Lesson 06 shared code: the lift task + a camera, pixel <-> 3D maths, and the camera-based observation.

Import this only AFTER the Isaac Sim app has been launched (it imports isaaclab modules).
"""
import os
import sys

import torch

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import combine_frame_transforms, quat_apply, quat_inv, subtract_frame_transforms
from isaaclab_tasks.manager_based.manipulation.lift.config.franka.joint_pos_env_cfg import FrankaCubeLiftEnvCfg_PLAY

HERE = os.path.dirname(os.path.abspath(__file__))
ROBOTICS = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROBOTICS, "lesson_05_find_cube"))
from cube_finder import CubeFinder  # noqa: E402  (lesson 05's network)

# the lesson-03 policy trained with 4096 envs (lifts + carries to the goal)
POLICY_CHECKPOINT = os.path.join(ROBOTICS, "logs", "rsl_rl", "franka_lift", "2026-10-24_17-36-25", "model_1499.pt")
DETECTOR_PATH = os.path.join(HERE, "models", "lift_cube_detector.pt")

# Front camera: in front of the table, looking back at the robot, tilted ~11 degrees down.
# "world" convention = the camera looks along its +X. The rotation is yaw 180 deg (turn to face -X)
# followed by pitch 11 deg (tilt down): quaternion (w, x, y, z) = (0, -sin(5.5), 0, cos(5.5)).
# Placed low and far so the cube stays in view both on the table and when lifted up to 0.5 m.
CAM_POS = (1.5, 0.0, 0.45)
CAM_ROT = (0.0, -0.0958, 0.0, 0.9954)
CAM_W, CAM_H = 160, 120


@configclass
class FrankaCubeLiftCameraEnvCfg(FrankaCubeLiftEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.scene.front_camera = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/FrontCamera",
            offset=TiledCameraCfg.OffsetCfg(pos=CAM_POS, rot=CAM_ROT, convention="world"),
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=18.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.05, 10.0)
            ),
            width=CAM_W,
            height=CAM_H,
        )
        # hide the goal markers - the camera network should only see the cube
        self.commands.object_pose.debug_vis = False


# ---------------- pixel <-> 3D (lesson 04 maths, batched on the GPU) ----------------
def world_to_pixel(cam, p_w):
    """World points (N, 3) -> pixel (N, 2) and depth along the camera's viewing axis (N,)."""
    p_cam = quat_apply(quat_inv(cam.data.quat_w_ros), p_w - cam.data.pos_w)  # ROS axes: z forward
    uvw = torch.bmm(cam.data.intrinsic_matrices, p_cam.unsqueeze(-1)).squeeze(-1)
    return uvw[:, :2] / uvw[:, 2:3], p_cam[:, 2]


def pixel_to_world(cam, uv, z):
    """Pixel (N, 2) + depth along the viewing axis (N,) -> world points (N, 3)."""
    ones = torch.ones_like(uv[:, :1])
    rays = torch.bmm(torch.linalg.inv(cam.data.intrinsic_matrices), torch.cat([uv, ones], 1).unsqueeze(-1)).squeeze(-1)
    return cam.data.pos_w + quat_apply(cam.data.quat_w_ros, rays * z.unsqueeze(-1))


def sample_depth(depth, uv):
    """Median of the 3x3 depth pixels around each (u, v). depth: (N, H, W)."""
    n, h, w = depth.shape
    c = uv[:, 0].round().long().clamp(1, w - 2)
    r = uv[:, 1].round().long().clamp(1, h - 2)
    offs = torch.tensor([-1, 0, 1], device=depth.device)
    rr = (r[:, None, None] + offs[None, :, None]).expand(n, 3, 3)
    cc = (c[:, None, None] + offs[None, None, :]).expand(n, 3, 3)
    vals = depth[torch.arange(n, device=depth.device)[:, None, None], rr, cc].reshape(n, 9)
    vals = torch.nan_to_num(vals, nan=100.0, posinf=100.0)
    return vals.median(dim=1).values


def camera_images(cam):
    rgb = cam.data.output["rgb"][..., :3]  # (N, H, W, 3) uint8
    depth = cam.data.output["distance_to_image_plane"][..., 0]  # (N, H, W) metres
    return rgb, depth


# ---------------- the camera-based observation ----------------
class CameraCubeEstimator:
    """Loads the trained detector once; turns camera images into a cube position."""

    def __init__(self, device):
        ckpt = torch.load(DETECTOR_PATH, map_location=device)
        self.model = CubeFinder().to(device).eval()
        self.model.load_state_dict(ckpt["state_dict"])
        self.depth_offset = float(ckpt["depth_offset"])  # calibrated in train_detector.py
        self.last_estimate_w = None  # kept so evaluate.py can measure the estimation error

    @torch.no_grad()
    def estimate_world(self, cam):
        rgb, depth = camera_images(cam)
        uv, _ = self.model(rgb.permute(0, 3, 1, 2).float().div(255))
        z = sample_depth(depth, uv) + self.depth_offset  # surface depth -> cube-centre depth
        self.last_estimate_w = pixel_to_world(cam, uv, z)
        return self.last_estimate_w


_ESTIMATOR = None
# evaluate.py flips this to compare the same policy with exact vs camera positions in one scene
MODE = {"use_camera": True}


def get_estimator(device):
    global _ESTIMATOR
    if _ESTIMATOR is None:
        _ESTIMATOR = CameraCubeEstimator(device)
    return _ESTIMATOR


def object_position_from_camera(env: ManagerBasedRLEnv, camera_cfg=SceneEntityCfg("front_camera"),
                                robot_cfg=SceneEntityCfg("robot")):
    """Drop-in replacement for mdp.object_position_in_robot_root_frame, but from the camera."""
    p_w = get_estimator(env.device).estimate_world(env.scene[camera_cfg.name])  # always run (for metrics)
    if not MODE["use_camera"]:
        p_w = object_world(env)  # exact position, for comparison
    robot = env.scene[robot_cfg.name]
    p_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, p_w)
    return p_b


def use_camera_for_object_position(env_cfg):
    """Swap the policy's privileged cube position for the camera estimate (same slot, same order)."""
    env_cfg.observations.policy.object_position = ObsTerm(func=object_position_from_camera)


# ---------------- helpers ----------------
def object_world(env):
    return env.scene["object"].data.root_pos_w[:, :3]


def goal_world(env):
    robot = env.scene["robot"]
    goal_b = env.command_manager.get_command("object_pose")[:, :3]
    goal_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, goal_b)
    return goal_w


def load_policy(env_wrapped, device):
    """Load the lesson-03 lift policy for inference."""
    from isaaclab_tasks.manager_based.manipulation.lift.config.franka.agents.rsl_rl_ppo_cfg import LiftCubePPORunnerCfg
    from rsl_rl.runners import OnPolicyRunner

    runner = OnPolicyRunner(env_wrapped, LiftCubePPORunnerCfg().to_dict(), log_dir=None, device=device)
    runner.load(POLICY_CHECKPOINT)
    return runner.get_inference_policy(device=device)
