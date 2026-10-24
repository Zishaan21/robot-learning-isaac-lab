"""Lesson 01: drops a cube onto a ground plane. See lesson.md in this folder.

Run from Learning\\Robotics:
    .\\run.bat lesson_01_hello_isaac\\hello_isaac.py --headless   # no window
    .\\run.bat lesson_01_hello_isaac\\hello_isaac.py              # with viewport
"""
import sys

from isaacsim import SimulationApp

headless = "--headless" in sys.argv
app = SimulationApp({"headless": headless})  # must be created before other isaacsim imports

from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid

world = World()
world.scene.add_default_ground_plane()
cube = world.scene.add(
    DynamicCuboid(prim_path="/World/cube", name="cube", position=[0, 0, 1.0], size=0.2)
)
world.reset()

for i in range(300):
    world.step(render=not headless)
    if i % 50 == 0:
        print(f"step {i}: cube z = {cube.get_world_pose()[0][2]:.3f}")

print("DONE - cube landed at z =", round(float(cube.get_world_pose()[0][2]), 3))
app.close()
