# Lesson 08 — A 6-joint arm (plan)

**Status: planned.** Written while lesson 07 trains; nothing here has been run yet. Each step will
be verified on this PC before it's marked done.

**Goal:** move pick-and-drop from the 7-joint Franka to a **6-joint arm** — your original goal.

## 1. Why a new robot means new training

A policy is tied to the body it was trained on. The Franka policy reads 9 joint positions + 9
velocities and outputs 8 commands (7 arm joints + gripper). A 6-joint arm has different joint
counts, link lengths, joint limits, motor strengths and hand. The **method** (task, rewards, 4096
envs, PPO) carries over; the **trained weights** don't. So: same recipe, new robot config, train again.

## 2. Which 6-joint arm? (both are in Isaac Lab, neither has a lift/pick task yet)

| | **Kinova Jaco2 N6S300** (recommended first) | UR10e + Robotiq 2F-140 |
|---|---|---|
| Joints | 6 + 3-finger hand | 6 + 2-finger parallel gripper |
| Size / reach | table-sized, ~1 m | industrial, ~1.3 m |
| Fits the lift table (cube at 0.4–0.6 m)? | yes | cramped — Isaac Lab's UR10e reach task works at **0.64–1.14 m**, so the table would move |
| Gripper control | simple: 3 finger + 3 fingertip joints, open 0.2 / close 1.2 (from `kinova.py`) | one drive joint + linked "mimic" joints — fiddlier in simulation |
| Known quirks | soft joint gains (stiffness 40 / 15) + gravity on → may sag | base frame rotated 180° (`base` vs `base_link`, see `deploy/reach/config/ur_10e`) |
| Config | `KINOVA_JACO2_N6S300_CFG` in `isaaclab_assets/robots/kinova.py` | `UR10e_ROBOTIQ_GRIPPER_CFG` in `isaaclab_assets/robots/universal_robots.py` |

**Recommendation: Kinova first** — it fits the table we already have, and its gripper is simple.
The UR10e is a good follow-up once one 6-joint arm works.

## 3. Steps

1. **Inspect** ([inspect_arms.py](inspect_arms.py), written, not yet run) — spawn both arms at the lift
   table, hold the default pose 2 s, print joint names / body names / hand position, save pictures.
   Answers: what is the hand body called, where does the hand sit relative to the table, does it sag?
   (First run downloads both robot models — slow on a slow connection.)
2. **Kinova lift config** — copy the pattern of Isaac Lab's Franka config
   (`lift/config/franka/joint_pos_env_cfg.py`), changing only robot-specific lines:
   - `scene.robot = KINOVA_JACO2_N6S300_CFG`
   - `arm_action`: joint position on `.*_joint_[1-6]`
   - `gripper_action`: binary on `.*_finger_.*`, open 0.2 / close 1.2
   - `commands.object_pose.body_name` and `ee_frame` target = the hand body (from step 1), with an
     offset to the point between the fingers (the "tool centre point")
   - maybe the default pose, so the hand starts above the table
3. **Train lift first** (lesson 03's task, 4096 envs) — proves the robot config works before adding the
   harder reward. One change at a time.
4. **Pick-and-drop** — reuse lesson 07's rewards; only the "gripper open" check changes (Kinova
   fingers: open = *small* angle, the opposite direction to the Franka).
5. **Evaluate + video** with lesson 07's `evaluate.py` approach.

## 4. What could go wrong (and how we'd see it)

| Risk | Symptom |
|---|---|
| hand body / tool-centre offset wrong | `reaching_object` plateaus low; video shows the wrist, not the fingertips, going to the cube |
| arm too weak / sags | inspect step: large joint deviation; training: jittery, cube dropped |
| 3-finger grasp on a cube is unstable | `lifting_object` rises then collapses, `object_dropping` high |
| default pose collides with the table | robot explodes / jumps at reset |
