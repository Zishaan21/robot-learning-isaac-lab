# Lesson 03 — Deep dive: how the arm learns

All values below are from this PC's Isaac Lab copy:
`IsaacLab\source\isaaclab_tasks\isaaclab_tasks\manager_based\manipulation\lift\`
(`lift_env_cfg.py`, `mdp\rewards.py`, `config\franka\agents\rsl_rl_ppo_cfg.py`).

## 1. How does it know where the cube is?

It doesn't *see* — there is no camera. Each step the simulator hands the policy a list of numbers
(`ObservationsCfg.PolicyCfg`):

| Input | Meaning |
|-------|---------|
| `joint_pos` | each joint's angle relative to the default pose |
| `joint_vel` | each joint's speed |
| `object_position` | **cube x, y, z in the robot base frame** |
| `target_object_position` | the commanded goal point in the air |
| `actions` | the previous action |

The cube position is exact, straight from the physics engine — **privileged information**.
A real robot needs a camera + pose estimation to get it; that's a later step
(train with exact state first, then add perception).

## 2. How it learns — PPO in 5 steps

The **policy (actor)** is an MLP `256 → 128 → 64` (ELU). Output: joint position targets + gripper
open/close.

1. **Try** — every environment steps 24 times; actions include random noise (exploration,
   `entropy_coef = 0.006` keeps it curious).
2. **Score** — collect rewards.
3. **Compare** — the **critic** (same size network) predicts expected future reward; an action is
   good if the outcome beat that prediction (the *advantage*).
4. **Adjust** — small update: `learning_rate = 1e-4`, `num_learning_epochs = 5`.
   PPO limits how far one update can move the policy, which keeps training stable.
5. **Repeat** — 1500 iterations by default.

`gamma = 0.98` = how much future reward counts: actions get credit for rewards several dozen
steps later, so reaching gets credit for the later lift.

## 3. What helps it learn: reward shaping

| Term | Formula (`rewards.py`) | Weight | Teaches |
|------|------------------------|-------:|---------|
| `reaching_object` | `1 − tanh(d / 0.1)` → 1.0 at contact, ≈0.24 at 10 cm, ≈0 at 30 cm | 1 | get the hand to the cube |
| `lifting_object` | 1 if cube z > 0.04 m else 0 | 15 | pick it up |
| `object_goal_tracking` | lifted × `(1 − tanh(d_goal / 0.3))` | 16 | carry toward the goal |
| `object_goal_tracking_fine_grained` | lifted × `(1 − tanh(d_goal / 0.05))` | 5 | be precise |
| `action_rate`, `joint_vel` | squared change / speed | −0.0001 | move smoothly |

A single "goal reached" reward would almost never happen by chance → nothing to learn from.
Shaped rewards give a gradient to climb: **reach → lift (sudden jump) → track the goal**.

## 4. Other things that help

- **Parallel environments** — many arms gather experience simultaneously.
- **Randomization** — each reset moves the cube (x ±0.1 m, y ±0.25 m) and the goal is resampled
  (x 0.4–0.6, y −0.25–0.25, z 0.25–0.5 m), so the policy must *use* the observations rather than
  memorise one motion.
- **Episodes** — 5 s attempts, then reset.
- **Joint-position actions** — the network outputs target angles; built-in joint controllers handle
  the motor forces.

## Scripted (lesson 02) vs learned (lesson 03)

| | Scripted | Learned |
|---|---|---|
| Motion comes from | 10 hand-written phases | a neural network |
| Cube position | exact | exact (privileged) |
| New situations | only what the script covers | generalises within the randomised range |
| Main work | writing logic | designing rewards + training time |

## Check yourself

1. Why would training fail if the only reward were "cube at goal"?
2. Why does `object_goal_tracking` multiply by "is lifted"? (What could the arm do otherwise?)
3. What breaks if the cube always spawned at the same spot during training but not at test time?
