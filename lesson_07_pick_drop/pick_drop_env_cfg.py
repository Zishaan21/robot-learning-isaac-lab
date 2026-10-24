"""Lesson 07: pick-and-drop task for the Franka - the lift task with a new goal and new rewards. See lesson.md.

Changes vs. lesson 03's Isaac-Lift-Cube-Franka-v0:
  * the goal is a spot ON THE TABLE (not a point in the air)
  * rewards: lift -> carry above the spot -> set it down -> open the gripper and move away
  * "placed" only counts if the cube was lifted earlier in the episode (no pushing it there)
  * 6 s episodes (pick + carry + place + release takes longer than lift-and-hold)

Imported by Isaac Lab via the gym registry (pick_drop_task.py) - only after the app is running.
"""
import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import combine_frame_transforms
from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.config.franka.agents.rsl_rl_ppo_cfg import LiftCubePPORunnerCfg
from isaaclab_tasks.manager_based.manipulation.lift.config.franka.joint_pos_env_cfg import FrankaCubeLiftEnvCfg

LIFT_HEIGHT = 0.04  # cube centre above this = lifted (same as lesson 03); it rests at ~0.024
# The cube SPAWNS at 0.055 m and drops onto the table, so it is "lifted" (> 0.04) for ~2 steps at the
# start of every episode. The "was it carried?" memory must ignore that, so it needs a real lift:
CARRIED_HEIGHT = 0.08
ON_TABLE_HEIGHT = 0.035  # cube centre below this = resting on the table
PLACE_RADIUS = 0.03  # within 3 cm (horizontally) of the goal spot = placed
RELEASE_HEIGHT = 0.05  # hand at least 5 cm above the cube = let go
GRIPPER_OPEN = 0.03  # Franka finger joint position (m); fully open = 0.04, closed = 0.0


# ---------------- helpers ----------------
def _goal_xy_dist(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Horizontal distance from the cube to the goal spot (the goal's height doesn't matter)."""
    robot, obj = env.scene["robot"], env.scene["object"]
    goal_b = env.command_manager.get_command(command_name)[:, :3]
    goal_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, goal_b)
    return torch.norm(goal_w[:, :2] - obj.data.root_pos_w[:, :2], dim=1)


def _was_lifted(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Memory: has the cube been really lifted (> 8 cm) at any point in this episode? (one flag per env)"""
    lifted = env.scene["object"].data.root_pos_w[:, 2] > CARRIED_HEIGHT
    flag = getattr(env, "_pick_drop_was_lifted", None)
    if flag is None or flag.shape != lifted.shape:
        flag = torch.zeros_like(lifted)
    flag = flag & (env.episode_length_buf > 1)  # forget it at the start of a new episode
    flag = flag | lifted
    env._pick_drop_was_lifted = flag
    return flag


def _placed(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    obj_z = env.scene["object"].data.root_pos_w[:, 2]
    return (obj_z < ON_TABLE_HEIGHT) & (_goal_xy_dist(env, command_name) < PLACE_RADIUS) & _was_lifted(env)


# ---------------- reward terms ----------------
# (function names end in _reward so they don't clash with the term names in PickDropRewardsCfg)
def carry_to_goal_reward(env: ManagerBasedRLEnv, std: float, command_name: str) -> torch.Tensor:
    """While lifted: reward being horizontally close to the goal spot (tanh kernel)."""
    lifted = env.scene["object"].data.root_pos_w[:, 2] > LIFT_HEIGHT
    return lifted * (1 - torch.tanh(_goal_xy_dist(env, command_name) / std))


def placed_reward(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Cube resting on the goal spot - and it got there by being carried, not pushed."""
    return _placed(env, command_name).float()


def released_reward(env: ManagerBasedRLEnv, command_name: str,
             gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["panda_finger.*"])) -> torch.Tensor:
    """Placed AND the gripper is open AND the hand has moved away: the cube was let go."""
    fingers = env.scene["robot"].data.joint_pos[:, gripper_cfg.joint_ids].mean(dim=1)
    hand_z = env.scene["ee_frame"].data.target_pos_w[:, 0, 2]
    obj_z = env.scene["object"].data.root_pos_w[:, 2]
    return (_placed(env, command_name) & (fingers > GRIPPER_OPEN) & (hand_z - obj_z > RELEASE_HEIGHT)).float()


@configclass
class PickDropRewardsCfg:
    """Every stage pays more than the one before (per second, at its best):
    hover above goal: lift 5 + carry 8 + fine 3 = 16  <  placed, still holding: ~21  <  placed + released: 40
    """

    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.1}, weight=1.0)
    lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": LIFT_HEIGHT}, weight=5.0)
    carry_to_goal = RewTerm(func=carry_to_goal_reward, params={"std": 0.3, "command_name": "object_pose"}, weight=8.0)
    carry_to_goal_fine = RewTerm(func=carry_to_goal_reward, params={"std": 0.05, "command_name": "object_pose"},
                                 weight=3.0)
    placed = RewTerm(func=placed_reward, params={"command_name": "object_pose"}, weight=20.0)
    released = RewTerm(func=released_reward, params={"command_name": "object_pose"}, weight=20.0)
    # same smoothness penalties (and curriculum) as the lift task
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-4, params={"asset_cfg": SceneEntityCfg("robot")})


@configclass
class FrankaPickDropEnvCfg(FrankaCubeLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards = PickDropRewardsCfg()
        self.episode_length_s = 6.0
        # goal = a spot on the table, in the same area the cube spawns in; keep it fixed for the episode
        goal = self.commands.object_pose
        goal.ranges.pos_x = (0.4, 0.6)
        goal.ranges.pos_y = (-0.25, 0.25)
        goal.ranges.pos_z = (0.025, 0.025)  # cube resting height
        goal.resampling_time_range = (100.0, 100.0)  # longer than an episode -> only changes at reset


@configclass
class FrankaPickDropEnvCfg_PLAY(FrankaPickDropEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class PickDropPPORunnerCfg(LiftCubePPORunnerCfg):
    """Same PPO settings as lesson 03; separate log folder."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "franka_pick_drop"
        self.max_iterations = 2000


# =====================================================================================================
# v2: drop the cube INTO A BOX.
# v1 (above) failed: the carry rewards only measured horizontal distance, so the arm held the cube
# perfectly above the spot - at 71 cm height - and never discovered putting it down (see lesson.md).
# v2 rewards the 3D distance to a drop point just above the box; from there, opening the gripper is
# a single action that drops the cube in.
# =====================================================================================================
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402

BOX_CENTRE = (0.5, -0.20)  # on the table, relative to the robot base (m)
BOX_INNER = 0.12  # inside width/length (m); the cube is ~4.8 cm
BOX_WALL = 0.01  # wall thickness (m)
BOX_HEIGHT = 0.06  # wall height (m); table top is z = 0
DROP_HEIGHT = 0.15  # drop point: cube centre this high above the table, over the box centre
IN_BOX_MARGIN = 0.045  # cube centre within +-4.5 cm of the box centre (inner half-width 6 cm)


def _box_wall(name, centre_xy, size_xy):
    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/BoxWall_" + name,  # flat name: Isaac Lab won't create a missing parent prim
        init_state=AssetBaseCfg.InitialStateCfg(pos=(centre_xy[0], centre_xy[1], BOX_HEIGHT / 2)),
        spawn=sim_utils.CuboidCfg(
            size=(size_xy[0], size_xy[1], BOX_HEIGHT),
            collision_props=sim_utils.CollisionPropertiesCfg(),  # collider, no rigid body -> fixed in place
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.35, 0.8)),
        ),
    )


def _in_box(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Cube inside the box (below the rim, within the walls) - and it was carried there."""
    robot, obj = env.scene["robot"], env.scene["object"]
    goal_b = env.command_manager.get_command(command_name)[:, :3]  # the drop point above the box centre
    goal_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, goal_b)
    d = (obj.data.root_pos_w[:, :2] - goal_w[:, :2]).abs()
    inside = (d[:, 0] < IN_BOX_MARGIN) & (d[:, 1] < IN_BOX_MARGIN) & (obj.data.root_pos_w[:, 2] < BOX_HEIGHT)
    return inside & _was_lifted(env)


def carry_to_drop_point_reward(env: ManagerBasedRLEnv, std: float, command_name: str) -> torch.Tensor:
    """While lifted: reward being close to the drop point in 3D (height included - the v1 fix)."""
    robot, obj = env.scene["robot"], env.scene["object"]
    goal_b = env.command_manager.get_command(command_name)[:, :3]
    goal_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, goal_b)
    lifted = obj.data.root_pos_w[:, 2] > LIFT_HEIGHT
    return lifted * (1 - torch.tanh(torch.norm(goal_w - obj.data.root_pos_w, dim=1) / std))


def in_box_reward(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    return _in_box(env, command_name).float()


def released_in_box_reward(env: ManagerBasedRLEnv, command_name: str,
                           gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["panda_finger.*"])
                           ) -> torch.Tensor:
    """In the box AND gripper open AND hand at least 5 cm above the cube."""
    fingers = env.scene["robot"].data.joint_pos[:, gripper_cfg.joint_ids].mean(dim=1)
    hand_z = env.scene["ee_frame"].data.target_pos_w[:, 0, 2]
    obj_z = env.scene["object"].data.root_pos_w[:, 2]
    return (_in_box(env, command_name) & (fingers > GRIPPER_OPEN) & (hand_z - obj_z > RELEASE_HEIGHT)).float()


@configclass
class DropInBoxRewardsCfg:
    """Best case per second: holding at the drop point: lift 15 + carry 8 + fine 3 (+ reach 1) = 27
    <  dropped in the box and released: in_box 25 + released 25 = 50.

    v2a used lift weight 5 (like v1): lifting wasn't discovered before the curriculum's 1000x
    smoothness penalty switched on (~iteration 417) and the policy collapsed to "don't move".
    v2b: lift weight back to lesson 03's proven 15, box rewards raised to keep the ordering,
    and the curriculum delayed (FrankaDropInBoxEnvCfg)."""

    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.1}, weight=1.0)
    lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": LIFT_HEIGHT}, weight=15.0)
    carry_to_drop_point = RewTerm(func=carry_to_drop_point_reward, params={"std": 0.3, "command_name": "object_pose"},
                                  weight=8.0)
    carry_to_drop_point_fine = RewTerm(func=carry_to_drop_point_reward,
                                       params={"std": 0.05, "command_name": "object_pose"}, weight=3.0)
    in_box = RewTerm(func=in_box_reward, params={"command_name": "object_pose"}, weight=25.0)
    released = RewTerm(func=released_in_box_reward, params={"command_name": "object_pose"}, weight=25.0)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-4, params={"asset_cfg": SceneEntityCfg("robot")})


@configclass
class FrankaDropInBoxEnvCfg(FrankaCubeLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards = DropInBoxRewardsCfg()
        self.episode_length_s = 6.0
        # the box: 4 walls on the table around BOX_CENTRE (fixed position - randomising it is an exercise)
        bx, by = BOX_CENTRE
        half = BOX_INNER / 2 + BOX_WALL / 2
        outer = BOX_INNER + 2 * BOX_WALL
        self.scene.box_wall_front = _box_wall("front", (bx + half, by), (BOX_WALL, outer))
        self.scene.box_wall_back = _box_wall("back", (bx - half, by), (BOX_WALL, outer))
        self.scene.box_wall_left = _box_wall("left", (bx, by + half), (outer, BOX_WALL))
        self.scene.box_wall_right = _box_wall("right", (bx, by - half), (outer, BOX_WALL))
        # the goal the policy observes = the drop point above the box (fixed)
        goal = self.commands.object_pose
        goal.ranges.pos_x = (bx, bx)
        goal.ranges.pos_y = (by, by)
        goal.ranges.pos_z = (DROP_HEIGHT, DROP_HEIGHT)
        goal.resampling_time_range = (100.0, 100.0)
        # spawn the cube only on the other side of the table (y from -0.02 to +0.25), never in/on the box
        self.events.reset_object_position.params["pose_range"]["y"] = (-0.02, 0.25)
        # 4 extra wall colliders per env overflow the lift task's GPU buffer (16*1024) at 4096 envs:
        # PhysX then reports "...totalAggregatePairsCapacity to 17814, otherwise, the simulation will
        # miss interactions" - i.e. silently skips collisions. Give it room.
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 64 * 1024
        # delay the 1000x smoothness penalty: 20000 per-env steps ~ iteration 830 (default 10000 ~ 417),
        # so lifting is learned first (v2a collapsed because it wasn't)
        self.curriculum.action_rate.params["num_steps"] = 20000
        self.curriculum.joint_vel.params["num_steps"] = 20000


@configclass
class FrankaDropInBoxEnvCfg_PLAY(FrankaDropInBoxEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class DropInBoxPPORunnerCfg(LiftCubePPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "franka_drop_in_box"
        self.max_iterations = 2000
