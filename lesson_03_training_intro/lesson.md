# Lesson 03 — Training intro: teach an arm to lift a cube

**Goal:** run your first reinforcement-learning (RL) training, watch the trained arm, and
understand what was learned. We use Isaac Lab's ready-made task **`Isaac-Lift-Cube-Franka-v0`**:
a Franka arm must grasp a cube and lift it to a target point in the air.

> The Franka has 7 joints (not 6). We start here because the task is ready-made and known to
> train well; lesson 04 moves the same idea to a 6-joint arm.

## 1. The idea: learning instead of scripting

In lesson 02 *you* (well, NVIDIA) wrote the recipe: approach → lower → grip → lift …
In RL nobody writes the recipe. The loop is:

```
observe  →  policy (neural network) picks joint actions  →  simulate one step  →  reward
   ↑                                                                               │
   └──────────────── adjust the network so high-reward actions become likelier ────┘
```

At first the arm flails randomly. Actions that happened before higher reward get reinforced.
After millions of steps a useful behaviour emerges.

**Why simulation:** the arm needs millions of attempts. Isaac Lab runs many copies of the
scene in parallel on the GPU (the default is **4096 arms at once**), so hours of practice
take minutes.

> Deeper explanation with formulas and the PPO steps: [how_it_learns.md](how_it_learns.md)

## 2. What the arm sees, does, and is rewarded for

**Observations (input to the network):** joint positions and velocities, the cube's position,
the target position, and the previous action.

**Actions (output):** a target angle for each arm joint, plus open/close for the gripper.

**Rewards** — from
`IsaacLab\source\isaaclab_tasks\isaaclab_tasks\manager_based\manipulation\lift\lift_env_cfg.py`:

| Term | Weight | Rewards… |
|------|-------:|----------|
| `reaching_object` | 1 | hand getting close to the cube |
| `lifting_object` | 15 | cube higher than 4 cm |
| `object_goal_tracking` | 16 | lifted cube moving toward the target point (coarse) |
| `object_goal_tracking_fine_grained` | 5 | … and getting precisely there |
| `action_rate` | −0.0001 | penalty for jerky, changing actions |
| `joint_vel` | −0.0001 | penalty for fast joint motion |

This is **reward shaping**: big task (lift to target) broken into smaller rewards that guide
learning step by step. Designing these is most of the work in robot RL — lesson 05 will add
a "drop at target" reward.

**Episode:** each attempt lasts 5 s, then the scene resets with a new cube/target position.

**Algorithm:** **PPO** (Proximal Policy Optimization), via the **RSL-RL** library. One
*iteration* = every arm takes 24 steps, then the network is updated once.

## 3. Run it

All commands from `<repo>` in a fresh cmd terminal.
Close the Isaac Sim GUI first.

### a) Smoke test (≈ 2–4 min) — does training start at all?

```bat
run.bat "%ISAACLAB%\scripts\reinforcement_learning\rsl_rl\train.py" --task Isaac-Lift-Cube-Franka-v0 --num_envs 16 --max_iterations 5 --headless
```

Verified on this PC (2026-10-24): starts in ~1 min, then ~0.65 s per iteration with 16 envs, e.g.

```
Learning iteration 4/5
Mean reward: 0.72
Metrics/object_pose/position_error: 0.1859
Iteration time: 0.67s
```

Checkpoints appear in `logs\rsl_rl\franka_lift\<date-time>\model_0.pt … model_4.pt`.

### b) Real training (long — leave it running)

```bat
run.bat "%ISAACLAB%\scripts\reinforcement_learning\rsl_rl\train.py" --task Isaac-Lift-Cube-Franka-v0 --num_envs 256 --max_iterations 1500 --headless
```

- `--num_envs 256`: number of arms practising in parallel. The default 4096 needs far more than
  this PC's 16 GB RAM / 10 GB VRAM. If it crashes or swaps heavily, use 128 or 64.
- `--max_iterations 1500`: the task's default. With fewer envs each iteration has less data, so
  the policy improves more slowly — expect useful lifting only well into the run.
- `--headless`: no window — much faster.
- Checkpoints are saved every 50 iterations to `logs\rsl_rl\franka_lift\<date-time>\model_<N>.pt`
  (inside this Learning\Robotics folder). You can stop with Ctrl+C at any time and keep the
  latest checkpoint.

### c) Watch the trained arm

Use the `isaaclab_run.py` wrapper (see note below):

```bat
run.bat isaaclab_run.py "%ISAACLAB%\scripts\reinforcement_learning\rsl_rl\play.py" --task Isaac-Lift-Cube-Franka-Play-v0 --num_envs 8
```

It loads the newest checkpoint automatically and opens a window with 8 arms. To record a video
instead (closes by itself), add `--video --video_length 200 --enable_cameras`; the mp4 lands in
`logs\rsl_rl\franka_lift\<run>\videos\play\`.

> **Why the wrapper:** Isaac Sim's RTX-sensor extension ships its own `hdf5.dll`. In GUI runs it
> loads first and Isaac Lab's `import h5py` then fails with
> `DLL load failed while importing _errors` (Windows error `0xc0000139`). `isaaclab_run.py`
> imports h5py *before* Isaac Sim starts. Side effect: the RTX lidar/radar extension
> (`isaacsim.sensors.rtx`) logs a DLL error and is unavailable in that run — not used in these
> lessons. Headless training doesn't hit the problem.

## 4. Reading the training output

Every line explained, with healthy/unhealthy patterns:
[reading_training_output.md](reading_training_output.md).

Every iteration prints a block. The lines that matter:

| Line | Meaning | Good sign |
|------|---------|-----------|
| `Mean reward` | average total reward per episode | rises over time |
| `Episode_Reward/lifting_object` | how often the cube gets lifted | goes from ~0 to clearly positive |
| `Episode_Reward/object_goal_tracking` | lifted cube near the target | rises after lifting is learned |
| `Mean episode length` | steps per attempt | stays at the 5 s max (no early failures) |
| `Learning iteration X/1500` + `ETA` | progress | — |

Typical pattern: reaching is learned first, then lifting (a sudden jump), then precise tracking.

## 5. What went wrong in our first run (and the fix)

First real run: 64 envs, defaults. Read from the TensorBoard log:

| Iteration | 0 | 181 | 362 | 543 | 724 |
|---|---|---|---|---|---|
| Mean reward | 0.72 | 0.71 | 1.23 | **−7.12** | **−2.57** |
| reaching_object | 0 | 0.01 | 0.20 | **0** | **0** |
| position error (m) | 0.11 | 0.25 | 0.39 | 0.45 | 0.55 |
| action noise std | 1.0 | 0.96 | 0.95 | 0.64 | 0.46 |

**Cause — a curriculum tuned for 4096 envs.** `lift_env_cfg.py` → `CurriculumCfg` raises the
`action_rate` and `joint_vel` penalties from −0.0001 to **−0.1** (1000×) once
`env.common_step_counter > 10000`. That counter counts *per-env* steps (24 per iteration), so the
jump happens at ~iteration 417 regardless of how many envs run.

- 4096 envs by iteration 417 ≈ 41 M samples → the arm already lifts; the penalty just smooths it.
- 64 envs by iteration 417 ≈ 0.64 M samples → the arm can't lift yet; a 1000× movement penalty
  makes **"don't move"** the best strategy. Reward goes negative, reaching collapses, and the
  shrinking noise std shows it *converged* on that bad behaviour.

**Lesson:** a **curriculum** (changing the task/rewards during training) must match how much
experience the agent has actually had. Settings copied from a big-GPU setup can silently break on
a small one.

**Fix — delay the penalty jump with command-line overrides** (Isaac Lab's `train.py` accepts
`env.*` / `agent.*` config overrides; no source edits needed):

```bat
run.bat "%ISAACLAB%\scripts\reinforcement_learning\rsl_rl\train.py" --task Isaac-Lift-Cube-Franka-v0 --num_envs 256 --max_iterations 1500 --headless env.curriculum.action_rate.params.num_steps=1000000 env.curriculum.joint_vel.params.num_steps=1000000
```

`num_steps=1000000` ≈ iteration 41 000, i.e. the strong penalties never switch on in this run.
256 envs also gives 4× more experience per iteration than 64.

**Result of the fixed run** (`2026-10-24_16-07-34`, 256 envs, 1500 iterations, 20.5 min,
9.2 M timesteps):

| | it 85 | it 885 | it 1499 |
|---|---|---|---|
| Mean reward | 0.70 | 4.6 | 4.89 |
| reaching_object | 0.007 | 0.76 | **0.84** |
| lifting_object | 0.12 | 0.12 | 0.12 (baseline only) |
| noise std | 0.95 | 0.82 | 0.72 |

Recorded play video: all arms move to their cube and hover at table height — **reaching
learned, grasp-and-lift not yet discovered**. 9.2 M timesteps is ~4× less experience than the
default 4096-env setup has by iteration 417, so this is expected. Next step: continue the same
run (`--resume`) for more iterations.

**Resumed to iteration 3387 — plateau.** reaching ≈ 0.80, lifting still 0.12, noise std
0.72 → 0.65, entropy 7.6 → 6.1. The policy settled in a **local optimum** ("reach and hover"
earns steady reward) while exploration shrank, so lifting — which needs *close gripper + move
up* together by chance — got even less likely. Stopped with Ctrl+C.

## 6. How many envs can this PC actually run?

Measured with 10-iteration test runs (RTX 3080 10 GB, 16 GB RAM, nothing else heavy open):

| `--num_envs` | Iteration time | Steps/s | Peak GPU memory | Min free RAM |
|---:|---:|---:|---:|---:|
| 256 | ~0.80 s | ~7 700 | — | — |
| 1024 | 0.87 s | 28 209 | 2.6 GB | 3.1 GB |
| 2048 | 0.84 s | 58 466 | 2.9 GB | 2.7 GB |
| **4096** | **0.88 s** | ~115 000 | **3.5 GB** | **1.7 GB** |

Iteration time barely changes: the GPU simulates all envs in parallel, and the per-step overhead
dominates. So 4096 envs gives **16× more experience for the same wall-clock time** as 256. The
earlier guess that this PC needed few envs was wrong for this task — RAM, not the GPU, is the
limit, and 4096 still fits.

**Lesson:** measure, don't guess. And with the full 4096 envs the task's own curriculum is
correct again, so no overrides are needed:

```bat
run.bat "%ISAACLAB%\scripts\reinforcement_learning\rsl_rl\train.py" --task Isaac-Lift-Cube-Franka-v0 --num_envs 4096 --headless
```

1500 iterations × ~0.9 s ≈ 23 min. Close browsers / Discord while it runs (RAM).

**Result** (run `2026-10-24_17-36-25`, 4096 envs, defaults, 25 min, 147 M timesteps):

| | it 203 | it 1491 | max* |
|---|---|---|---|
| Mean reward | 27.5 | **141** | — |
| lifting_object | 4.99 | **13.4** | 15 (cube in the air ~89 % of each episode) |
| object_goal_tracking | 0.32 | **12.9** | 16 |
| object_goal_tracking_fine_grained | 0.0004 | **2.9** | 5 |
| position_error | 0.48 m | **0.10 m** | — |
| object_dropping | 10 % | **0.02 %** | — |
| noise std | 1.24 | 0.37 | — |

\*max = the term's weight (per-second average over the 5 s episode).

What happened, in order: lifting discovered by **iteration ~200** (value loss spiked to 2.1 as the
critic met rewards it had never seen; noise std *rose* to 1.24 to explore around the discovery),
then goal tracking, then fine tracking. The curriculum raised the smoothness penalties to −0.1 at
~iteration 417 — *after* lifting was learned, as intended — and the policy kept its skills.

Same PC, same task: 256 envs never lifted in 4500 iterations; 4096 envs lifted in 200.

## Try it

1. Run the smoke test and note the time per iteration.
2. Start the real training. After ~100 iterations, run `play.py` in a second terminal — the arm
   should at least reach toward the cube. Compare again later.
3. Open `lift_env_cfg.py` and find `RewardsCfg`. Predict: what would the arm learn if
   `lifting_object` had weight 0?

## Key words

**reinforcement learning** · **policy** · **observation / action / reward** · **reward shaping**
· **episode** · **PPO** · **iteration** · **checkpoint** · **parallel environments**
