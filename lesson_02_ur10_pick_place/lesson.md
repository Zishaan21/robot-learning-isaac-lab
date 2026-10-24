# Lesson 02 — UR10 pick & place (scripted)

**Goal:** make a real 6-joint industrial arm pick up a cube and drop it at a target — using
hand-written logic. This is the baseline that lesson 03+ will replace with learning.

## Run it

Close the Isaac Sim GUI first (memory), then:

```bat
cd "<repo>"
.\run.bat lesson_02_ur10_pick_place\ur10_pick_place.py
```

You should see the arm move to the cube, grab it, carry it and release it. The terminal prints
each phase and, at the end, how far (cm) the cube landed from the target.

Verified output on this PC (2026-10-24):

```
UR10 joints (6): ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
step     1  phase 0: move above cube
step   100  phase 1: lower to cube
step   386  phase 2: settle
step   486  phase 3: close gripper
step   487  phase 4: lift
step   612  phase 5: carry to target
step   812  phase 6: lower to target
step  1012  phase 7: release
step  1013  phase 8: move up
step  1113  phase 9: return
DONE after 1126 steps - cube at [ 0.6   -0.4    0.026], 0.0 cm from drop target
```

1126 steps at 60 steps/s ≈ 19 s of simulated time. Warnings like `Cannot assign transform to
non-root articulation link` or `non-existent path .../visuals/wrist3` are harmless noise from the
robot model.

## The robot: UR10

A **6-DOF** (degrees of freedom) arm = 6 motorised joints:

| Joint | Moves |
|-------|-------|
| `shoulder_pan_joint` | rotates the whole arm around its base |
| `shoulder_lift_joint` | raises/lowers the upper arm |
| `elbow_joint` | bends the elbow |
| `wrist_1/2/3_joint` | orient the hand (tilt, turn, spin) |

6 joints is the minimum to put the hand at **any position (x, y, z) with any orientation
(roll, pitch, yaw)** in its reach — that is the "6D" in 6-DOF.

Its gripper here is a **suction gripper**: when "closed" it sticks to whatever it touches.

## How the script decides what to do

[ur10_pick_place.py](ur10_pick_place.py) uses a **state machine** — a fixed list of phases:

```
0 move above cube → 1 lower → 2 settle → 3 grip → 4 lift →
5 carry → 6 lower to target → 7 release → 8 move up → 9 return
```

Each phase gives a target position for the **hand** (end effector). A motion planner
(**RMPflow**) converts "hand should be *here*" into "each of the 6 joints should be at *these
angles*" — that conversion is **inverse kinematics**. It also avoids jerky or unsafe motions.

The loop is the same as lesson 01: *read state (observations) → decide (controller) →
act (apply joint targets) → step*.

## Try it

1. Change `DROP_TARGET` to `[0.3, 0.6, 0.0257]` and rerun — the arm finds a new path itself.
2. Move `CUBE_START` far away, e.g. `[1.5, 0.0, 0.0257]` — what happens when the cube is
   out of reach? (UR10 reach ≈ 1.3 m.)
3. Run with `--headless` — notice it only prints the phases and the final error.
4. After `DONE`, press **Stop ■** then **Play ▶** in the window — the scene resets and the arm
   repeats the task. (Stop destroys the physics view, so the script must call `world.reset()`
   before reading joint states again — without that you get
   `'NoneType' object has no attribute 'positions'`.)

## Why we will move to learning

This works only because a human wrote every phase and the planner knows the exact cube
position. It breaks if the object is a different shape, slips, or moves. In **reinforcement
learning** the robot discovers the motions itself by trial and error, rewarded for getting the
object to the target — that is lesson 03.

## Key words

**DOF** · **end effector** · **inverse kinematics** · **state machine** · **motion planner (RMPflow)**
· **suction gripper**
