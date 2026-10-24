"""Registers the lesson-07 tasks with gym so Isaac Lab's train.py / play.py can use them by name.

Only strings here - Isaac Lab imports pick_drop_env_cfg.py later, after the simulator has started.
"""
import gymnasium as gym

TASKS = (
    # v1: place on a table spot (failed - see lesson.md)
    ("Lesson-PickDrop-Franka-v0", "FrankaPickDropEnvCfg", "PickDropPPORunnerCfg"),
    ("Lesson-PickDrop-Franka-Play-v0", "FrankaPickDropEnvCfg_PLAY", "PickDropPPORunnerCfg"),
    # v2: drop into a box
    ("Lesson-DropInBox-Franka-v0", "FrankaDropInBoxEnvCfg", "DropInBoxPPORunnerCfg"),
    ("Lesson-DropInBox-Franka-Play-v0", "FrankaDropInBoxEnvCfg_PLAY", "DropInBoxPPORunnerCfg"),
)

for task_id, env_cfg, runner_cfg in TASKS:
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": f"pick_drop_env_cfg:{env_cfg}",
            "rsl_rl_cfg_entry_point": f"pick_drop_env_cfg:{runner_cfg}",
        },
        disable_env_checker=True,
    )
