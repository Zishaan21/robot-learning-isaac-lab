"""Lesson 04: RGB + depth cameras, and connecting pixels to 3D positions. See lesson.md.

Based on NVIDIA's example: isaacsim/standalone_examples/api/isaacsim.sensors.camera/camera.py

Run from Learning\\Robotics (close other Isaac Sim / training runs first):
    .\\run.bat lesson_04_cameras\\cameras.py --headless   # just saves images + prints numbers
    .\\run.bat lesson_04_cameras\\cameras.py              # also shows the scene

Output images go to lesson_04_cameras\\output\\
"""
import os
import sys

from isaacsim import SimulationApp

HEADLESS = "--headless" in sys.argv
simulation_app = SimulationApp({"headless": HEADLESS})  # must come before other isaacsim imports

import isaacsim.core.utils.numpy.rotations as rot_utils
import matplotlib

matplotlib.use("Agg")  # save images to files, no plot window
import matplotlib.pyplot as plt
import numpy as np
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.sensors.camera import Camera

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT_DIR, exist_ok=True)

CUBE_SIZE = 0.05  # same 5 cm cube as lesson 02
NUM_SHOTS = 3  # how many random cube positions to photograph
SETTLE_STEPS = 30  # physics steps for the cube to land after being moved
MAX_WAIT_STEPS = 1000  # first run can take a while: the renderer compiles shaders before the first frame

world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()
cube = world.scene.add(
    DynamicCuboid(
        prim_path="/World/cube",
        name="cube",
        position=np.array([0.5, 0.0, CUBE_SIZE / 2]),
        size=CUBE_SIZE,
        color=np.array([1.0, 0.0, 0.0]),
    )
)

# Camera orientation uses "world" axes: the camera looks along its +X axis.
# Euler (roll, pitch, yaw) in degrees: pitch 90 = looking straight down.
cameras = {
    # straight down from 1.2 m above the workspace centre
    "top": Camera(
        prim_path="/World/camera_top",
        position=np.array([0.5, 0.0, 1.2]),
        resolution=(320, 240),
        orientation=rot_utils.euler_angles_to_quats(np.array([0, 90, 0]), degrees=True),
    ),
    # from behind and above the robot base, tilted 45 degrees down (like a head camera)
    "angled": Camera(
        prim_path="/World/camera_angled",
        position=np.array([-0.3, 0.0, 0.9]),
        resolution=(320, 240),
        orientation=rot_utils.euler_angles_to_quats(np.array([0, 45, 0]), degrees=True),
    ),
}

world.reset()
for cam in cameras.values():
    cam.initialize()
    cam.add_distance_to_image_plane_to_frame()  # enables depth


def images_ready():
    for cam in cameras.values():
        rgb, depth = cam.get_rgb(), cam.get_depth()
        if rgb is None or depth is None or rgb.size == 0 or depth.size == 0:
            return False
    return True


print("waiting for the first camera frames...")
for step in range(MAX_WAIT_STEPS):
    world.step(render=True)
    if images_ready():
        print(f"cameras ready after {step + 1} steps")
        break
else:
    print(f"no camera frames after {MAX_WAIT_STEPS} steps - rendering may be broken")

rng = np.random.default_rng(0)
for shot in range(NUM_SHOTS):
    # put the cube somewhere random on the floor in front of where a robot would stand
    true_pos = np.array([rng.uniform(0.3, 0.7), rng.uniform(-0.25, 0.25), CUBE_SIZE / 2])
    cube.set_world_pose(position=true_pos)
    for _ in range(SETTLE_STEPS):
        world.step(render=True)
    true_pos = cube.get_world_pose()[0]  # where it actually settled

    print(f"\n=== shot {shot}: cube centre (world, m) = {np.round(true_pos, 3)}")
    for name, cam in cameras.items():
        rgb = cam.get_rgb()
        depth = cam.get_depth()
        if rgb is None or depth is None or rgb.size == 0:
            print(f"  [{name}] no image from this camera")
            continue

        # 3D -> pixel: where should the cube centre appear in this image?
        u, v = cam.get_image_coords_from_world_points(np.array([true_pos]))[0]
        # pixel + depth -> 3D: can we recover the cube position from the image alone?
        px, py = int(round(u)), int(round(v))
        inside = 0 <= px < rgb.shape[1] and 0 <= py < rgb.shape[0]
        if inside:
            d = float(depth[py, px])
            recovered = cam.get_world_points_from_image_coords(np.array([[u, v]]), np.array([d]))[0]
            # depth hits the cube's top/side surface, not its centre, so expect ~half a cube of error
            err_cm = np.linalg.norm(recovered - true_pos) * 100
            pixel_rgb = rgb[py, px]
            print(
                f"  [{name}] pixel (u,v)=({u:.1f},{v:.1f})  colour there={pixel_rgb}  depth={d:.3f} m  "
                f"recovered={np.round(recovered, 3)}  error={err_cm:.1f} cm"
            )
        else:
            print(f"  [{name}] cube is outside this camera's view (u,v)=({u:.1f},{v:.1f})")

        fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
        axes[0].imshow(rgb)
        axes[0].set_title(f"{name} RGB")
        if inside:
            axes[0].plot(u, v, "o", markerfacecolor="none", markeredgecolor="cyan", markersize=14)
        im = axes[1].imshow(depth, cmap="viridis")
        axes[1].set_title(f"{name} depth (m)")
        fig.colorbar(im, ax=axes[1], fraction=0.046)
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        path = os.path.join(OUT_DIR, f"shot{shot}_{name}.png")
        fig.savefig(path, dpi=100)
        plt.close(fig)
        print(f"  saved {path}")

print("\nDONE")
if not HEADLESS:
    print("Close the Isaac Sim window to exit.")
    while simulation_app.is_running():
        world.step(render=True)
simulation_app.close()
