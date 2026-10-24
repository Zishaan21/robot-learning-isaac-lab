# Lesson 06 — Camera-based lifting (teacher → student)

**Goal:** run the lesson-03 lift policy with the cube position coming from a **camera** instead of
the simulator, and measure how much that costs.

```
lesson 03:  simulator ──exact cube position──────────────────────────► policy ──► arm
lesson 06:  front camera ──RGB──► CNN (lesson 05) ──pixel──┐
                         ──depth───────────────────────────┴─► 3D ──► policy ──► arm
```

The policy itself is **unchanged** — same network, same weights. Only one of its inputs is
swapped. This is the simplest form of the **teacher → student** idea: the policy learned with
privileged information (teacher); perception replaces the privileged part (student).

## Run it

Nothing else on the GPU. All from `<repo>`:

```bat
run.bat lesson_06_camera_lift\collect_data.py --headless        & rem 1. labelled images from the lift scene
run.bat lesson_06_camera_lift\train_detector.py                  & rem 2. retrain lesson 05's CNN on them
run.bat lesson_06_camera_lift\evaluate.py --headless             & rem 3. exact vs camera, same policy
run.bat lesson_06_camera_lift\evaluate.py --num_envs 8           & rem    ...or watch it (camera phase second)
```

Uses the trained lift policy from lesson 03 (`logs\rsl_rl\franka_lift\2026-10-24_17-36-25\model_1499.pt`).

**Videos:** `evaluate.py` records environment 0 in both phases to `lesson_06_camera_lift\output\videos\`
(skip with `--no_video`):

| File | Shows |
|------|-------|
| `overview_exact.mp4` | the arm from the side, policy using the **exact** cube position |
| `overview_camera.mp4` | same, policy using the **camera** estimate |
| `camera_exact.mp4` | what the front camera sees (exact phase); green ring = true cube centre, red cross = network estimate |
| `camera_camera.mp4` | same, camera phase — here the red cross is what the policy actually acts on |

Each is real time (25 fps), 3 episodes × 5 s = 15 s.

## 1. Why we couldn't reuse lesson 05's detector directly

Lesson 05 trained on: plain floor, top-down camera, **red** cube, grey decoy, no robot.
The lift scene has: a table, a **yellow DexCube**, the robot arm in front of / around the cube,
neighbouring robots in the background, a different camera angle and lighting.

A network only knows what it was trained on — feeding it this scene would be **distribution
shift** (lesson 05, exercise 4). So we keep the *method* and *network design*, and collect new
data from the scene where it will be used. (`cube_finder.py` from lesson 05 is imported unchanged.)

## 2. The camera ([camera_lift.py](camera_lift.py))

A `TiledCamera` is added to every environment (Isaac Lab renders all of them in one pass). It is
defined by `CAM_POS` and `CAM_ROT` in `camera_lift.py`, relative to each environment's robot base.

| | Value | Meaning |
|---|---|---|
| Position | **(1.5, 0, 0.45) m** from the robot base | 1.5 m in front of the robot, centred, 45 cm above the table |
| Facing | **yaw 180°** | turned around to look back at the robot (along −X) |
| Tilt | **pitch 11° down** | aimed at roughly (0.5, 0, 0.25) |
| Lens | focal length 18 mm | ~60° wide field of view |
| Image | 160 × 120 px | RGB + depth |

**Side view**

```
 z (up)
  │
 0.5 ┤                ▲ lifted goal (up to 0.5 m)
     │       arm
     │      ╭──╮                          📷 camera (1.5, 0, 0.45)
     │      │  ╰─╮                     ◄──── looking back, 11° down
     │      │    ✋ ■ cube         ◄────
 0.0 ┼──────┴─────────────────────────────── table top
     0     0.4   0.6        1.0        1.5   x (m, forward from robot)
   robot   workspace
   base
```

**Top view**

```
   y
   ▲     ┌─────────────── table ────────────────┐
  0.25 ┤ │        ░░░░░░░░                      │
       │ 🤖       ░ cube ░            ◄──── 📷   │  ← camera on the centre line (y = 0)
 -0.25 ┤ │        ░░░░░░░░                      │
       │ └──────────────────────────────────────┘
       0         0.4–0.6                     1.5  → x
```

**Why there:**

- **In front, not above** — from above, the gripper hides the cube exactly when it grasps it.
- **Low (0.45 m), small tilt** — the cube stays in view on the table *and* when lifted up to 0.5 m;
  a higher, steeper camera would lose the lifted cube off the top of the image.
- **1.5 m back + wide lens** — the whole workspace (y −0.25…+0.25 m, full lift height) fits in the
  frame.

So the robot is always *behind* the cube from the camera's point of view — that's why the arm
appears in the background of the images in `data\label_check.png`. Camera placement is a design
decision with real consequences.

**Rotation as a quaternion:** in Isaac Lab's "world" camera convention the camera looks along +X.
Yaw 180° then pitch 11° gives (w, x, y, z) = (0, −sin 5.5°, 0, cos 5.5°) = (0, −0.0958, 0, 0.9954).

**See it yourself:** run `evaluate.py --num_envs 8` (with a window), then in the viewport's camera
menu (the "Perspective" button at the top of the viewport) choose `FrontCamera` under
`/World/envs/env_0/` — the view switches to exactly what the network sees. Or watch
`output\videos\camera_camera.mp4` (below).

## 3. Data from the real task ([collect_data.py](collect_data.py))

The trained policy (with exact positions) drives 50 arms; every 6 steps each camera image is saved
with the exact label: the cube centre projected into the image (`world_to_pixel`), plus the cube
centre's depth.

**Label check (lesson 05's lesson):** the depth camera should see a surface about half a cube in
front of the cube centre at the labelled pixel. The script prints this and saves
`data\label_check.png` with labels drawn on images. If images lagged the labels, both would show it.

## 4. Depth calibration

The depth camera measures the cube's **front surface**; the policy needs the **centre**.
[train_detector.py](train_detector.py) measures the median gap on the training images
(`depth_offset`) and adds it to every estimate. That's **calibration**: measure a systematic error
once, correct it forever. (Lesson 05 used the "it's on the table" prior instead — that doesn't
work here because the cube gets lifted.)

## 5. Swapping the observation

The policy's input vector is built from observation terms in a fixed order. In
[camera_lift.py](camera_lift.py):

```python
env_cfg.observations.policy.object_position = ObsTerm(func=object_position_from_camera)
```

`object_position_from_camera` = camera images → CNN pixel → depth → `pixel_to_world` →
robot-base frame (`subtract_frame_transforms`, exactly like the original term). Same slot, same
units, so the policy can't tell the difference — except for the errors.

## 6. Results (verified on this PC, 2026-10-24)

**Data** — 4000 images, label check: centre depth − measured depth at the label pixel = median
2.6 cm, within 0–6 cm for 100 % of images (no lag; labels drawn in `data\label_check.png`).

**Detector** (800 validation images, 40 epochs, ~3 min):

```
depth offset (centre - surface), calibrated on training images: 2.56 cm
pixel error:       mean 0.22 px   95% < 0.68 px
3D position error: mean 0.34 cm   median 0.32 cm   95% < 0.67 cm
```

In `output\predictions.png` the worst validation images still have small *pixel* errors
(0.1–1.4 px) and mostly show the cube **held in the gripper** — so the larger 3D error most likely
comes from **depth**: next to the fingers/edges the 3×3 depth window can catch a finger or the
background instead of the cube face.

**Policy, exact vs camera** (`evaluate.py --headless`, 64 envs × 3 episodes, second half of each):

```
                                 exact    camera
cube lifted (% of time)         100.0%     99.7%
cube-goal distance (cm)            1.6       2.1
within 10 cm of goal (%)         99.2%     98.6%
camera estimate error (cm)        0.75      1.04
  95th percentile (cm)            0.98      1.04
```

**The camera version keeps almost all of the performance**: lifting 99.7 % vs 100 %, and the cube
ends up 2.1 cm from the goal instead of 1.6 cm. The policy was trained only with exact positions,
yet ~1 cm of camera error barely matters — the Franka's gripper and the policy's closed-loop
corrections absorb it.

Why is the estimate error during the run (0.75–1.04 cm) larger than offline (0.34 cm)? Offline
images were mostly of a still or slowly moving cube. During the run the cube is carried, and an
image is a few frames old (lesson 05: renderer lag), so the estimate trails the moving cube. With
the camera in the loop it's a bit larger still, because the arm then reacts to its own slightly
wrong estimates. (Inference from the numbers — not separately measured.)

## Try it

1. Watch the camera phase with `--num_envs 8` (not headless). Where does it fail?
2. Add noise to the camera estimate (e.g. `+ torch.randn_like(p_w) * 0.02` in
   `object_position_from_camera`). How much noise can the policy tolerate before lifting fails?
3. Move the camera (change `CAM_POS` / `CAM_ROT`), collect + train again. Does a worse viewpoint
   hurt?
4. Think: the policy was never trained with camera errors. How could you train it to be robust to
   them? (Hint: add noise to its observations during RL training — Isaac Lab calls this
   *observation corruption*; see `enable_corruption` in `lift_env_cfg.py`.)

## Key words

**teacher → student** · **privileged information** · **distribution shift** · **camera placement /
occlusion** · **calibration** · **observation swap** · **robustness**
