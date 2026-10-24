"""Lesson 05, step 1: generate a labelled camera dataset in simulation. See lesson.md.

Each sample = top-camera RGB + depth image of a red cube at a random position/rotation (plus a grey
decoy cube), labelled with the red cube's true pixel (u, v) and 3D position. In simulation the
labels are free and exact - that is the whole point.

Run from Learning\\Robotics (nothing else using the GPU):
    .\\run.bat lesson_05_find_cube\\generate_data.py --headless
    .\\run.bat lesson_05_find_cube\\generate_data.py --headless --num 500   # quick test
Output: lesson_05_find_cube\\data\\cube_dataset.npz
"""
import argparse
import os
import sys
import time

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--num", type=int, default=3000, help="number of images to generate")
parser.add_argument("--headless", action="store_true")
args, _ = parser.parse_known_args()
simulation_app = SimulationApp({"headless": args.headless})  # must come before other isaacsim imports

import isaacsim.core.utils.numpy.rotations as rot_utils
import numpy as np
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.sensors.camera import Camera

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUT_DIR, exist_ok=True)

CUBE_SIZE = 0.05
RESOLUTION = (128, 96)  # small images train fast; (width, height)
# workspace = where the lesson-03 task puts cubes (x 0.4-0.6 +- 0.1, y +-0.25)
X_RANGE, Y_RANGE = (0.3, 0.7), (-0.25, 0.25)
MIN_STEPS = 3  # render at least this many frames after moving the cubes
MAX_STEPS = 30  # give up on a sample if the image still hasn't caught up
LABEL_TOLERANCE = 3.0  # px; red pixels in the image must be centred this close to the label
EDGE_MARGIN = 4  # px; skip samples where the cube centre is at the very edge of the image

world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()
cube = world.scene.add(
    DynamicCuboid(prim_path="/World/cube", name="cube", position=np.array([0.5, 0.0, CUBE_SIZE / 2]),
                  size=CUBE_SIZE, color=np.array([1.0, 0.0, 0.0]))
)
decoy = world.scene.add(
    DynamicCuboid(prim_path="/World/decoy", name="decoy", position=np.array([0.5, 0.2, CUBE_SIZE / 2]),
                  size=CUBE_SIZE, color=np.array([0.5, 0.5, 0.5]))
)
# straight down from 1.5 m above the workspace centre (higher than lesson 04 -> whole workspace in view)
camera = Camera(
    prim_path="/World/camera_top",
    position=np.array([0.5, 0.0, 1.5]),
    resolution=RESOLUTION,
    orientation=rot_utils.euler_angles_to_quats(np.array([0, 90, 0]), degrees=True),
)

world.reset()
camera.initialize()
camera.add_distance_to_image_plane_to_frame()
for _ in range(1000):  # wait for the first frames (first run compiles shaders)
    world.step(render=True)
    rgb, depth = camera.get_rgb(), camera.get_depth()
    if rgb is not None and depth is not None and rgb.size and depth.size:
        break

K = np.asarray(camera.get_intrinsics_matrix(), dtype=np.float32)  # lens: 3D camera point -> pixel
V = np.asarray(camera.get_view_matrix_ros(), dtype=np.float32)  # pose: world -> camera (ROS axes)

rng = np.random.default_rng(42)


def random_pose():
    pos = np.array([rng.uniform(*X_RANGE), rng.uniform(*Y_RANGE), CUBE_SIZE / 2])
    yaw = rng.uniform(0, 90)  # a cube looks the same every 90 degrees
    return pos, rot_utils.euler_angles_to_quats(np.array([0, 0, yaw]), degrees=True)


def red_centroid(rgb):
    """Centre of the red pixels - an independent check of where the cube is in the image."""
    rgb = rgb.astype(int)
    mask = (rgb[..., 0] > 150) & (rgb[..., 1] < 110) & (rgb[..., 2] < 110)
    ys, xs = np.nonzero(mask)
    return None if len(xs) == 0 else np.array([xs.mean(), ys.mean()])


rgbs, depths, pixels, positions = [], [], [], []
skipped = 0
start = time.time()
while len(rgbs) < args.num and simulation_app.is_running():
    cube_pos, cube_rot = random_pose()
    decoy_pos, decoy_rot = random_pose()
    while np.linalg.norm(decoy_pos[:2] - cube_pos[:2]) < 0.1:  # keep them from touching
        decoy_pos, decoy_rot = random_pose()
    for obj, pos, rot in ((cube, cube_pos, cube_rot), (decoy, decoy_pos, decoy_rot)):
        obj.set_world_pose(position=pos, orientation=rot)
        obj.set_linear_velocity(np.zeros(3))
        obj.set_angular_velocity(np.zeros(3))
    # The camera image lags the simulation by a few frames: right after moving the cube, get_rgb()
    # can still show the PREVIOUS position (we measured exactly that). So keep rendering until the
    # red pixels in the image sit where the label says the cube is.
    w, h = RESOLUTION
    ok = False
    for step in range(MAX_STEPS):
        world.step(render=True)
        if step + 1 < MIN_STEPS:
            continue
        true_pos = cube.get_world_pose()[0]
        u, v = camera.get_image_coords_from_world_points(np.array([true_pos]))[0]
        seen = red_centroid(np.asarray(camera.get_rgb()))
        if seen is not None and np.linalg.norm(seen - [u, v]) < LABEL_TOLERANCE:
            ok = True
            break
    in_view = EDGE_MARGIN <= u < w - EDGE_MARGIN and EDGE_MARGIN <= v < h - EDGE_MARGIN
    if not (ok and in_view):
        skipped += 1
        continue

    rgbs.append(np.asarray(camera.get_rgb(), dtype=np.uint8).copy())
    depths.append(np.asarray(camera.get_depth(), dtype=np.float16).copy())
    pixels.append([u, v])
    positions.append(true_pos)
    if len(rgbs) % 250 == 0:
        print(f"{len(rgbs)}/{args.num} images  ({time.time() - start:.0f} s)")

path = os.path.join(OUT_DIR, "cube_dataset.npz")
np.savez_compressed(
    path,
    rgb=np.stack(rgbs),  # (N, H, W, 3) uint8
    depth=np.stack(depths),  # (N, H, W) float16, metres
    pixel=np.array(pixels, dtype=np.float32),  # (N, 2) true (u, v) of the red cube centre
    position=np.array(positions, dtype=np.float32),  # (N, 3) true world position of the cube centre
    K=K,
    V=V,
    cube_size=np.float32(CUBE_SIZE),
)
print(f"DONE - saved {len(rgbs)} samples to {path} in {time.time() - start:.0f} s "
      f"(skipped {skipped}: image never matched the label, or cube at the image edge)")
simulation_app.close()
