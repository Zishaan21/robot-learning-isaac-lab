"""Lesson 02: scripted pick-and-drop with a 6-joint UR10 arm (no training yet). See lesson.md.

Based on NVIDIA's example: isaacsim/standalone_examples/api/isaacsim.robot.manipulators/universal_robots/pick_place.py

Run from Learning\\Robotics:
    .\\run.bat lesson_02_ur10_pick_place\\ur10_pick_place.py              # with the viewport
    .\\run.bat lesson_02_ur10_pick_place\\ur10_pick_place.py --headless   # no window, just prints results

Change CUBE_START / DROP_TARGET below and rerun to see the arm adapt.
"""
import sys

from isaacsim import SimulationApp

HEADLESS = "--headless" in sys.argv
simulation_app = SimulationApp({"headless": HEADLESS})  # must come before other isaacsim imports

import numpy as np
from isaacsim.core.api import World
from isaacsim.robot.manipulators.examples.universal_robots.controllers.pick_place_controller import PickPlaceController
from isaacsim.robot.manipulators.examples.universal_robots.tasks import PickPlace

# --- things to experiment with (metres, robot base is at the origin) ---
CUBE_START = np.array([0.6, 0.3, 0.0257])  # where the cube spawns (z = half the 5 cm cube)
DROP_TARGET = np.array([0.3, 0.6, 0.0257])  # where it should be dropped
# DROP_TARGET = np.array([0.6, -0.4, 0.0257])  # where it should be dropped

PHASES = [
    "move above cube",
    "lower to cube",
    "settle",
    "close gripper",
    "lift",
    "carry to target",
    "lower to target",
    "release",
    "move up",
    "return",
]

world = World(stage_units_in_meters=1.0)
task = PickPlace(cube_initial_position=CUBE_START, target_position=DROP_TARGET)
world.add_task(task)  # adds ground plane, UR10 with suction gripper, cube and target marker
world.reset()

params = task.get_params()
cube_name = params["cube_name"]["value"]
robot_name = params["robot_name"]["value"]
ur10 = world.scene.get_object(robot_name)
controller = PickPlaceController(name="pick_place_controller", gripper=ur10.gripper, robot_articulation=ur10)
articulation_controller = ur10.get_articulation_controller()

print(f"UR10 joints ({ur10.num_dof}): {ur10.dof_names}")

last_phase = -1
done = False
step = 0
reset_needed = False
while simulation_app.is_running():
    world.step(render=not HEADLESS)
    # pressing Stop in the GUI destroys the physics view; reset before reading the robot again
    if world.is_stopped() and not reset_needed:
        reset_needed = True
    if not world.is_playing():
        continue
    if reset_needed:
        world.reset()
        controller.reset()
        reset_needed = False
        last_phase, done, step = -1, False, 0
        print("--- restarted after Stop/Play ---")
    step += 1

    obs = world.get_observations()
    actions = controller.forward(
        picking_position=obs[cube_name]["position"],
        placing_position=obs[cube_name]["target_position"],
        current_joint_positions=obs[robot_name]["joint_positions"],
        end_effector_offset=np.array([0, 0, 0.02]),
    )
    articulation_controller.apply_action(actions)

    phase = controller.get_current_event()
    if phase != last_phase and phase < len(PHASES):
        print(f"step {step:5d}  phase {phase}: {PHASES[phase]}")
        last_phase = phase

    if controller.is_done() and not done:
        done = True
        cube_pos = obs[cube_name]["position"]
        error_cm = np.linalg.norm(cube_pos[:2] - DROP_TARGET[:2]) * 100
        print(f"DONE after {step} steps - cube at {np.round(cube_pos, 3)}, {error_cm:.1f} cm from drop target")
        if HEADLESS:
            break
        print("Close the Isaac Sim window to exit.")

simulation_app.close()
