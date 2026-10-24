# Lesson 05 — Find the cube from the camera

**Goal:** replace the "free" cube position (privileged information, lesson 03) with one measured
from a camera. We train a small neural network to find the cube's pixel in a top-camera image,
then use lesson 04's pixel + depth → 3D conversion to get its world position.

```
camera RGB ──► CNN ──► pixel (u, v) ──┐
camera depth ─────────────────────────┴──► back-projection ──► cube (x, y, z) ──► lift policy (lesson 06)
```

This is ordinary **supervised learning** (learn from examples with correct answers), not RL —
much faster and more reliable, because the simulator gives us the exact answer for every image.

## Run it

Nothing else should be using the GPU (finish/stop training first).

**Step 1 — generate data** (Isaac Sim, ~11 minutes for 3000):

```bat
cd /d "<repo>"
run.bat lesson_05_find_cube\generate_data.py --headless
```

Saves 3000 labelled images to `lesson_05_find_cube\data\cube_dataset.npz`.
Quick test first with `--num 500` if you like.

**Step 2 — train the detector** (PyTorch only, ~1–3 min on the GPU):

```bat
run.bat lesson_05_find_cube\train_detector.py
```

Prints the validation error every 5 epochs and a final table; saves
`models\cube_detector.pt`, `output\predictions.png` (images with true vs predicted centre +
heatmaps) and `output\training_curve.png`.

## 1. The dataset — why simulation is perfect for this

[generate_data.py](generate_data.py) repeats, 3000 times:

1. Put the **red cube** at a random position in the lesson-03 workspace (x 0.3–0.7 m,
   y −0.25–0.25 m) with a random rotation.
2. Put a **grey decoy cube** somewhere else (≥ 10 cm away) — so the network must learn
   "the *red* one", not just "any cube".
3. Render the top camera (128 × 96 px, 1.5 m above the table, looking down).
4. Save RGB, depth, and the **labels**: the true pixel (projection, lesson 04) and true 3D position.

In the real world each label would need a human clicking on the cube, or a motion-capture
system. Here it's exact and free. Also saved: the camera's **K** (lens) and **V** (pose), so the
training script can do the pixel → 3D maths itself.

**Randomisation is the key idea:** anything the network should *ignore* (position, rotation,
decoys, lighting) must vary in the training data, otherwise it learns shortcuts. Lighting is
randomised in step 2 with brightness/contrast jitter (`augment()`). This is a small taste of
**domain randomisation**, the main trick for making simulation-trained vision work on real cameras.

## 2. The network — a heatmap, not two numbers

[cube_finder.py](cube_finder.py) → `CubeFinder` (used by `train_detector.py`, reused in lesson 06):

```
image (3×96×128) → 6 conv layers (two of them halve the size) → 64×24×32 features
                 → 1×1 conv → heatmap 24×32 → softmax → probabilities
                 → predicted (u, v) = probability-weighted average of cell positions   ("soft-argmax")
```

Why a heatmap instead of a network that outputs (u, v) directly?

- Convolutions slide the same filter over every location — they are naturally good at saying
  "there's something red *here*", but bad at turning that into a coordinate number.
- A heatmap keeps the "where" in the image layout; soft-argmax converts it to coordinates in a
  way that's differentiable (so it can be trained) and sub-pixel accurate.

**Loss:** smooth-L1 distance in pixels between predicted and true (u, v).
**Split:** 80 % training, 20 % validation (never trained on — that's how we know it generalises).

## 3. From pixel to 3D (lesson 04, by hand)

`pixel_to_world()`:

```
d       = depth at (u, v)            (3×3 median, robust to one bad pixel)
p_cam   = d · K⁻¹ · [u, v, 1]        point in camera frame
p_world = V⁻¹ · [p_cam, 1]           camera frame → world frame
```

The final table compares three versions:

| Row | Meaning |
|-----|---------|
| true pixel + depth | the best any detector could do — limited by depth hitting the cube's *top face* (~2.5 cm, as in lesson 04) |
| predicted pixel + depth | our detector |
| predicted + "on the floor" | same, but set z = half the cube size because we *know* it rests on the table — using prior knowledge removes most of the top-face error |

## 4. Reading the results

- **Pixel error** of 1 px at this resolution/height ≈ 1 cm on the table (rough guide).
- If the training error keeps falling but validation stops improving → **overfitting**: more or
  more varied data helps.
- Look at `output\predictions.png`: the heatmap should be a tight bright blob on the red cube,
  **not** on the grey decoy.

How accurate is "good enough"? The lesson-03 policy was trained with *exact* positions. A few
cm of error may still allow grasping (the Franka gripper opens ~8 cm); lesson 06 tests exactly that.

## 5. The bug we hit: labels one frame behind

First full run: 3000 images, 40 epochs — the loss **never moved** (23.1 → 22.9) and the
validation error stayed at ~36 px, the same as always guessing the image centre.
`output\predictions.png` showed why: the green "true" circles were **not on the red cube**.

Measurement (centre of the red pixels in image *i* vs. the labels):

| compare image *i* with… | median distance |
|---|---|
| label *i* (its own) | 48.3 px |
| **label *i − 1*** | **1.19 px** |
| label *i − 2* … *i − 5* | ~48 px |

Every image showed the cube where the *previous* sample had put it. The camera's image lags the
simulation by a few rendered frames, and the generator only waited 3 frames after moving the
cube (lesson 04 waited 30, which hid the problem). The network was asked to map images to
answers that didn't match them — so the best it could do was guess the middle.

**Fix:** after moving the cube, keep rendering until an independent check agrees — the centre of
the red pixels must be within 3 px of the label — and skip the sample otherwise
(`red_centroid()` in `generate_data.py`).

**Result after the fix** (verified on this PC, 2026-10-24):

```
DONE - saved 3000 samples ... in 663 s (skipped 434: image never matched the label, or cube at the image edge)
image i vs label i-0: median distance   1.23 px        <- labels now match their own image
epoch   1  train loss  1.142  val pixel error: mean 13.90 px
epoch   5  train loss  0.064  val pixel error: mean 0.41 px
epoch  40  train loss  0.016  val pixel error: mean 0.23 px, median 0.19 px

pixel error (detector):                 mean 0.23 px   95% < 0.47 px
3D error, true pixel + depth:           mean 2.50 cm  (limit: depth sees the top face)
3D error, predicted pixel + depth:      mean 2.50 cm
3D error, predicted + 'on the floor':   mean 0.30 cm   95% < 0.52 cm
```

- The detector is as good as a perfect one once depth is involved (2.50 cm both) — the remaining
  error is the top-face offset, not the network.
- Adding the prior "the cube rests on the table" brings it to **3 mm**.

**Ghosts — why the image lags.** Look closely at `output\predictions.png`: there are faint,
semi-transparent copies of cubes where earlier samples put them. Isaac Sim's real-time renderer
blends several recent frames to reduce noise (temporal accumulation), so a moved object fades
out over a few frames. That's the lag that broke the labels, and the reason 434 samples were
skipped (ghost red pixels pulled the red-pixel centre off the label). The trained network learns to
pick the solid cube and ignore ghosts — but a real static camera wouldn't show them, so for a
serious sim-to-real dataset you'd render more frames per sample (raise `MIN_STEPS`) or change
renderer settings.

**Lessons that transfer to every ML project:**

- **Look at your data before training.** One picture with labels drawn on it would have caught
  this immediately.
- **"Loss doesn't move at all"** usually means the inputs and labels don't match — not that the
  model is too small.
- **Sensors have latency.** In real robots too: a camera frame is always from slightly in the
  past. Timestamps matter.

## Try it

1. Train with `--epochs 5`, then 40. How does the error change? Look at `training_curve.png`.
2. Generate only `--num 300` images and retrain. What happens to the validation error?
3. Remove the decoy (comment out the `decoy` lines) and regenerate. Is the task easier? Would a
   detector trained *without* decoys still work *with* them? (Try: train without, test with.)
4. Change the cube colour in `generate_data.py` to blue but keep the trained model. What happens?
   This is the **distribution shift** problem real robots face.
5. Set `MIN_STEPS = 10` and regenerate. Do the ghosts fade? How many samples are skipped now,
   and how much longer does generation take?

## Key words

**supervised learning** · **labels** · **dataset / train-validation split** · **CNN** ·
**heatmap / soft-argmax** · **augmentation** · **domain randomisation** · **overfitting** ·
**distribution shift**
