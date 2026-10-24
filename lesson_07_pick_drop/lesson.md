# Lesson 07 — Pick and drop: designing a reward

**Goal:** teach the Franka to pick up the cube, carry it, and **drop it into a box**. Same robot, same
training setup as lesson 03 — the only new things are the **reward** and the box.
(Changing one thing at a time keeps problems debuggable. A 6-joint arm comes in lesson 08.)

This lesson has two versions, because the first reward design **failed** — and *why* it failed is the
most useful part.

## Run it

From `<repo>`, close other GPU programs:

```bat
rem train v2: drop into a box (4096 arms, 2000 iterations, ~35 min)
run.bat lesson_07_pick_drop\run_task.py "%ISAACLAB%\scripts\reinforcement_learning\rsl_rl\train.py" --task Lesson-DropInBox-Franka-v0 --num_envs 4096 --headless

rem score it + video (output\videos\box_<run>.mp4)
run.bat lesson_07_pick_drop\evaluate.py --headless --video

rem watch it live
run.bat lesson_07_pick_drop\run_task.py "%ISAACLAB%\scripts\reinforcement_learning\rsl_rl\play.py" --task Lesson-DropInBox-Franka-Play-v0 --num_envs 8
```

v1 (table spot) is still available: task `Lesson-PickDrop-Franka-v0`, `evaluate.py --variant table`.
`run_task.py` registers the lesson's tasks, then runs Isaac Lab's normal `train.py` / `play.py`.

## 1. A policy only does what it was rewarded for

Baseline — the lesson-03 **lift** policy, unchanged, in the new tasks (64 envs):

| | v1: place on a table spot | v2: drop into the box |
|---|---|---|
| cube carried | 100 % | 100 % |
| goal reached (placed / in box) | 0 % | 0 % |
| released | 0 % | 0 % |
| where the cube ends up | 9.3 cm from the spot, hovering | **15.0 cm high, 2.3 cm from the drop point** — right above the box |

It carries the cube to the target and holds it there forever: it learned "hold the cube near a
point", and "let go" was never rewarded. New behaviour needs new rewards.

## 2. Version 1: place on a spot on the table — FAILED

[pick_drop_env_cfg.py](pick_drop_env_cfg.py), `FrankaPickDropEnvCfg`: goal = a spot on the table;
rewards = reach (1), lift (5), carry *horizontally* near the spot (8 + 3), placed (20), released (20).

### Rule 1: every stage must pay more than the one before

| Arm's situation | Earns | Total / s |
|---|---|---:|
| holding the cube above the spot | lift 5 + carry 8 + fine 3 | 16 |
| cube on the spot, still gripped | placed 20 (+ reach ~1) | ~21 |
| cube on the spot, released | placed 20 + released 20 | 40 |

### Rule 2: close the loopholes

The arm could **push** the cube across the table onto the spot. So "placed" only counts if the cube
**was carried** earlier in the episode — a per-environment memory (`_was_lifted`), set when the cube
goes above 8 cm, cleared at each new episode.

*Bug caught in the smoke test:* the memory first used 4 cm. But the cube **spawns** at 5.5 cm and
drops onto the table — "lifted" for ~2 steps every episode (lesson 03's free baseline). So the
memory always said "carried" and the loophole was open. Fix: 8 cm (`CARRIED_HEIGHT`).
**Test every reward term against cases where it must *not* fire.**

### What happened (run `2026-10-24_19-59-40`, 4096 envs, 2000 it, 34 min)

```
Episode_Reward/lifting_object: 4.5575      (max 5)
Episode_Reward/carry_to_goal: 7.1151       (max 8)
Episode_Reward/carry_to_goal_fine: 2.4318  (max 3)
Episode_Reward/placed: 0.0000              <- never, in 2000 iterations
Episode_Reward/released: 0.0000
Mean action noise std: 0.14                <- stopped exploring
```

`evaluate.py --variant table`: cube carried 100 %, placed 0 %, horizontal distance to the spot
**0.2 cm** — but cube height **median 70.9 cm** (10–90 %: 58–77 cm). It held the cube perfectly
above the spot, arm stretched up high.

### Why it failed

The carry rewards measured only **horizontal** distance. At 70 cm or 3 cm height they paid the same,
so **nothing pulled the cube down**. To ever find "placed", the arm would have had to lower the cube
~68 cm with zero reward on the way — and lose the lift reward just before getting there. Rule 1 held
at the *end points*, but there was no slope between them. With exploration noise down to 0.14, it
was never going to happen.

**Lesson:** a reward needs a *gradient* toward the goal, not only a big payout at the goal.

## 3. Version 2: drop into a box

`FrankaDropInBoxEnvCfg`:

| Change | Why |
|---|---|
| a **box** on the table: 12 × 12 cm inside, 6 cm walls, at (0.5, −0.20) m — 4 static wall colliders (`_box_wall`) | something to drop into |
| goal = **drop point 15 cm above the box centre** (fixed) | the policy observes it as its target |
| carry rewards use the **3D** distance to the drop point | pulls the cube to the right height (the v1 fix) |
| `in_box`: cube inside the walls, below the rim, after being carried | the goal |
| `released`: in the box + gripper open + hand > 5 cm above the cube | let go |
| cube spawns only on the far side (y −0.02…+0.25) | never starts in or on the box |

| Arm's situation | Earns | Total / s |
|---|---|---:|
| holding the cube at the drop point | lift 15 + carry 8 + fine 3 (+ reach 1) | ~27 |
| opened the gripper → cube falls in | in_box 25 + released 25 | 50 |

From the drop point, success is **one action: open the gripper**. That's far easier to discover by
exploration than a long slow descent. And the 3D carry reward gives a slope all the way to the drop
point. (The baseline above shows the old lift policy already gets to that point — it just never opens.)

### v2a collapsed — two problems at once

The first v2 run (`..._v2a_collapsed`) used lift weight **5** and the default curriculum:

| | it 398 | it 788 |
|---|---|---|
| Mean reward | 4.8 | **−4.6** |
| lifting_object | 0.033 (free baseline) | 0.033 |
| noise std | 0.85 | 0.46 |

Lifting was never discovered, and at ~iteration 417 the curriculum's 1000× smoothness penalty
switched on → "don't move" (the same collapse as lesson 03's 64-env run). Stopped at 788.

When restarting, the log showed a second problem:

```
PhysX error: The application needs to increase PxGpuDynamicsMemoryConfig::totalAggregatePairsCapacity
to 17814 , otherwise, the simulation will miss interactions
```

The 4 box walls × 4096 envs overflow a GPU physics buffer that the lift task sets to 16 × 1024. When
it overflows, PhysX **silently skips collisions** — possibly fingers against the cube. v2a used the
same config, so it very likely had this too (its output went to a terminal, so not confirmed) — a
plausible extra reason lifting was never found. **Always read warnings/errors at the start of a run.**

### v2b — the fixes

1. lift weight **15** (lesson 03's proven value); box rewards raised to **25 + 25** to keep the order
2. curriculum delayed to 20 000 steps (~iteration 830) so lifting comes first
3. `sim.physx.gpu_total_aggregate_pairs_capacity = 64 * 1024` — the restart showed **0 PhysX errors**

## 4. What to watch while it trains

Read with [lesson 03's guide](../lesson_03_training_intro/reading_training_output.md). Expected order:

1. `reaching_object` ↑ → `lifting_object` jumps (~iteration 200 with 4096 envs)
2. `carry_to_drop_point` ↑
3. `in_box` > 0 — it discovers that opening above the box drops the cube in
4. `released` follows `in_box` closely (opening *is* how it gets in)

If `carry_to_drop_point_fine` gets near its max (3) while `in_box` stays at 0 for hundreds of
iterations: hovering again — it hasn't discovered opening. Fixes to consider: more exploration
(`agent.algorithm.entropy_coef`), or a bigger drop reward.

## 5. Results — v2b works (verified on this PC, 2026-10-24)

Run `2026-10-24_21-09-44`, 4096 envs, 2000 iterations, ~37 min, 0 PhysX errors.

**Training progress** (from the TensorBoard log, checked every ~7 min):

| iteration | 283 | 440 | 827 | 1197 | 1999 | max |
|---|---|---|---|---|---|---|
| Mean reward | 98 | 138 | 250 | 239 | 253 | — |
| lifting_object | 11.2 | 11.9 | 0.86 | 0.64 | 0.57 | 15 |
| carry_to_drop_point | 3.4 | 5.8 | 0.26 | 0.18 | 0.16 | 8 |
| **in_box** | 0.48 | 1.30 | **20.2** | 21.4 | **21.6** | 25 |
| **released** | 0.25 | 1.19 | **20.1** | 21.3 | **21.6** | 25 |
| noise std | 1.21 | 1.30 | 1.80 | 1.05 | 0.82 | — |

- Lifting was learned before iteration 283 — *before* the curriculum — this time.
- First drops into the box already by iteration 283; by ~800 it's routine.
- **lift and carry fall** once it's good: it gets the cube into the box early in each 6 s episode,
  so it spends little time carrying. The rewards measure *time spent in a state* — fast success
  means less carry reward, and that's fine.
- The 1000× smoothness penalty switched on at ~830 (reward 250 → 239 briefly); the skill survived.

**Evaluation** (`evaluate.py --headless --video`, 64 envs × 3 episodes, state at episode end):

```
cube was carried (> 8 cm)       : 100.0%
cube in the box                 : 100.0%   (after being carried)
... and released (SUCCESS)       : 100.0%   (gripper open, hand > 5 cm above)
cube-to-target distance          : median 4.2 cm (horizontal)
cube height (centre)             : median 2.1 cm, 10-90%: 2.1-2.1 cm
```

The video (`output\videos\box_2026-10-24_21-09-44.mp4`) shows the real thing: pick up the cube from
the table, carry it over the box, open, cube falls in, arm moves aside with the gripper open. The
4.2 cm is from the box centre — the cube settles near a side/corner of the 12 cm box after falling.

### The whole story in one table

| Version | Change | Result |
|---|---|---|
| lift policy (lesson 03) | — | carries, never lets go (0 %) |
| v1 | place on a table spot, horizontal carry reward | hovered at 71 cm, placed 0 % |
| v2a | drop into a box, 3D carry, lift weight 5 | collapsed ("don't move") + PhysX overflow |
| **v2b** | lift 15, box 25+25, curriculum delayed, PhysX buffer 64K | **100 % in the box and released** |

Four runs to get one success — that's normal in RL. Each failure was diagnosed from the numbers
and the video, and each fix changed one understood thing.

## Try it

1. Before training v2: predict the iteration where `in_box` first rises above 0.
2. Randomise the box position each episode (harder: the walls are static colliders; you'd make the box
   one kinematic rigid object and move it in a reset event). How would the reward need to change?
3. Remove the `_was_lifted(env)` condition from `_in_box`. Could the arm "cheat"? (Hint: the cube can't
   be pushed *over* a 6 cm wall — so maybe not. Loopholes depend on the scene.)
4. Start v2 from the lift policy instead of from scratch: add `--resume --load_run 2026-10-24_17-36-25`
   with the lift run copied into `logs\rsl_rl\franka_drop_in_box\`. It already reaches the drop point.
   What might help, what might hurt? (Hint: its exploration noise is already low.)

## Key words

**reward design** · **reward shaping** · **reward gradient** · **stage ordering** · **local optimum** ·
**reward hacking / loopholes** · **episodic memory in rewards** · **baseline**
