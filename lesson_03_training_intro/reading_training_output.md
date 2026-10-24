# Lesson 03 — Reference: every line of the training output

Example block (run `2026-10-24_16-07-34`, 256 envs, iteration 85):

```
                       Learning iteration 85/1500
                       Computation: 9053 steps/s (collection: 0.549s, learning 0.130s)
             Mean action noise std: 0.94
          Mean value_function loss: 0.0010
               Mean surrogate loss: -0.0048
                 Mean entropy loss: 10.8867
                       Mean reward: 0.70
               Mean episode length: 250.00
    Episode_Reward/reaching_object: 0.0073
     Episode_Reward/lifting_object: 0.1200
Episode_Reward/object_goal_tracking: 0.0182
Episode_Reward/object_goal_tracking_fine_grained: 0.0000
        Episode_Reward/action_rate: -0.0017
          Episode_Reward/joint_vel: -0.0023
            Curriculum/action_rate: -0.0001
              Curriculum/joint_vel: -0.0001
Metrics/object_pose/position_error: 0.2336
Metrics/object_pose/orientation_error: 3.0564
      Episode_Termination/time_out: 1.0000
Episode_Termination/object_dropping: 0.0000
                   Total timesteps: 528384
                    Iteration time: 0.68s
                      Time elapsed: 00:01:06
                               ETA: 00:18:16
```

Timing facts for this task (from the run's saved `params\env.yaml`): physics `dt = 0.01 s`,
`decimation = 2` → the policy acts every **0.02 s (50 Hz)**; `episode_length_s = 5` →
**250 policy steps per episode**.

## Progress & speed

| Line | Meaning | In the example |
|------|---------|----------------|
| `Learning iteration 85/1500` | one iteration = every env takes 24 steps (`num_steps_per_env`), then one network update | 85 of 1500 done |
| `Computation: N steps/s` | env-steps simulated per second, all envs together | 9053 |
| `collection` / `learning` | time spent simulating vs. updating the network | 0.55 s sim, 0.13 s learning → simulation is the bottleneck |
| `Total timesteps` | env-steps so far = iteration × envs × 24 | 528 384 ≈ 86 × 256 × 24 |
| `Iteration time` | seconds per iteration | 0.68 s |
| `Time elapsed` / `ETA` | wall-clock so far / estimated remaining | ~18 min left |

## PPO internals (how the learning itself is going)

| Line | Meaning | Healthy pattern |
|------|---------|-----------------|
| `Mean action noise std` | how much randomness is added to actions — **exploration** | starts at 1.0, falls slowly as the policy gets confident. Falling *fast while reward is bad* = converging on a bad habit (our first run) |
| `Mean entropy loss` | the same idea as noise, measured as entropy (higher = more random) | falls together with noise std |
| `Mean value_function loss` | how wrong the **critic's** reward predictions are | small and stable; spikes when the task suddenly gets easier/harder (e.g. lifting discovered) |
| `Mean surrogate loss` | PPO's policy objective for the update (negative = the update improved expected advantage) | small values near 0, either sign; not a progress meter — don't read too much into it |

## Rewards — the main progress signals

`Episode_Reward/<term>` = that term's reward summed over an episode, averaged over episodes that
ended in this iteration, and (Isaac Lab convention) divided by the episode length in seconds (5).
Each term's **weight** is in `lift_env_cfg.py → RewardsCfg`.

| Line | Weight | Meaning | Watch for |
|------|-------:|---------|-----------|
| `Mean reward` | — | total reward per episode (all terms, not divided by 5) | rises over training; **negative = penalties dominate** (first run) |
| `Mean episode length` | — | policy steps per episode | 250 = always the full 5 s (no early terminations) |
| `reaching_object` | 1 | hand close to the cube | first thing to rise |
| `lifting_object` | 15 | cube above 0.04 m | **0.12 is a free baseline**: the cube spawns at 0.055 m and is "lifted" for ~2 steps while it settles (15 × 0.02 s × 2 ÷ 5 s = 0.12). Real lifting = clearly **above 0.12** |
| `object_goal_tracking` | 16 | lifted cube near the goal (coarse, 0.3 m scale) | rises after lifting is learned |
| `object_goal_tracking_fine_grained` | 5 | lifted cube *very* near the goal (0.05 m scale) | last to rise |
| `action_rate` | −0.0001 | penalty for changing actions abruptly | small negative |
| `joint_vel` | −0.0001 | penalty for fast joints | small negative |

## Curriculum

| Line | Meaning |
|------|---------|
| `Curriculum/action_rate`, `Curriculum/joint_vel` | the **current weight** of those penalties. Default schedule jumps them to −0.1 after 10 000 per-env steps (~iteration 417). We override that with `env.curriculum.*.params.num_steps=1000000`, so they stay at −0.0001 |

## Task metrics (not rewards — just measurements)

| Line | Meaning |
|------|---------|
| `Metrics/object_pose/position_error` | distance (m) between the cube and the goal point. Should **fall** as tracking is learned |
| `Metrics/object_pose/orientation_error` | angle (rad) between cube and goal orientation. **Ignore in this task** — orientation isn't rewarded, so it stays around 3 |

## Terminations (why episodes ended)

| Line | Meaning |
|------|---------|
| `Episode_Termination/time_out` | fraction of episodes that ran the full 5 s. 1.0 = all |
| `Episode_Termination/object_dropping` | fraction that ended because the cube fell off the table |

## Quick health check

| You see | It means |
|---------|----------|
| reaching ↑, then lifting jumps above 0.12, then goal tracking ↑, position error ↓ | learning normally |
| Mean reward negative, reaching → 0, noise std falling fast | penalties too strong too early → policy learned "don't move" |
| Everything flat for many hundreds of iterations, noise std ~1.0 | not enough experience yet — more envs or more iterations |
| `object_dropping` rising | the arm is knocking the cube off the table |
