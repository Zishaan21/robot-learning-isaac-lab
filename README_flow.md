# Robotics with Isaac Sim — course

Goal: train a 6-joint robot arm to **pick an object and drop it at a target**.

We get there in small steps: first make the simulator do something, then control an arm with
hand-written logic, then replace that logic with a policy learned by reinforcement learning (RL).

| # | Lesson | You learn | Status |
|---|--------|-----------|--------|
| 01 | [Hello Isaac](lesson_01_hello_isaac/lesson.md) | How a simulation script is built: app, world, objects, physics steps | ✅ done |
| 02 | [UR10 pick & place](lesson_02_ur10_pick_place/lesson.md) | Robot arms, joints, grippers, a scripted pick-and-place controller | ✅ done |
| 03 | [Training intro](lesson_03_training_intro/lesson.md) | What RL is, Isaac Lab, running a ready-made "lift cube" training | ✅ lifts + carries to target (4096 envs) |
| 04 | [Cameras](lesson_04_cameras/lesson.md) | Add RGB/depth cameras, read images, pixel ↔ 3D conversions | ✅ done |
| 05 | [Find the cube from the camera](lesson_05_find_cube/lesson.md) | Generate labelled sim images, train a heatmap CNN, pixel + depth → 3D | ✅ verified: 0.23 px / 3 mm |
| 06 | [Camera-based lift](lesson_06_camera_lift/lesson.md) | Front camera in the lift scene, retrain detector, drive the lesson-03 policy from the camera | ✅ camera: 99.7 % lifted, 2.1 cm from goal |
| 07 | [Pick and drop](lesson_07_pick_drop/lesson.md) | Same Franka, new reward. v1 (place on a spot) hovered at 71 cm; v2b drops into a box | ✅ 100 % in the box + released |
| 08 | [6-joint arm](lesson_08_six_joint_arm/lesson.md) | Move pick-and-drop to a 6-joint arm (Kinova Jaco2 first, then UR10e + Robotiq) | ⏸ paused after the arm inspection (see `inspect_kinova.log`) |

**Why cameras come after state-based training:** learning perception and control together from raw
pixels needs far more compute than this PC has. Standard practice is a *teacher* policy trained
with exact object positions (lesson 03), then a camera-based *student* (lessons 05–06).

## Setup (already done)

| Thing | Where |
|-------|-------|
| Isaac Sim 5.1 (simulator app) | `E:\Personal Projects\Robotics\isaacsim` |
| Python env `isaac_lab_5` (Python 3.11, CUDA torch) | `E:\Personal Projects\Robotics\isaac_lab_5` |
| Isaac Lab (training framework, from lesson 03) | `E:\Personal Projects\Robotics\IsaacLab` |
| NVIDIA driver | **580.88** — keep it; newer drivers crashed Isaac Sim's renderer |

## How to run any lesson script

From this folder, in cmd or PowerShell:

```bat
.\run.bat lesson_01_hello_isaac\hello_isaac.py --headless
.\run.bat lesson_02_ur10_pick_place\ur10_pick_place.py
```

`run.bat` activates the env, accepts the Isaac Sim EULA for that run, and starts Python unbuffered
(so `print` output shows up). `--headless` = no window: faster and uses less memory.

**Tips for this PC (16 GB RAM):** close the Isaac Sim GUI before running a script, and close
browsers/Discord/Teams while simulating.
