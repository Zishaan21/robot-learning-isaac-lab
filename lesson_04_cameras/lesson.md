# Lesson 04 — Cameras: what the robot sees

**Goal:** put RGB and depth cameras into a scene, save what they see, and learn the two
conversions every vision-based robot relies on:

```
3D point in the world  ──(projection)──►  pixel (u, v) in the image
pixel (u, v) + depth   ──(back-projection)──►  3D point in the world
```

In lesson 03 the policy got the cube position for free ("privileged information"). Here we
start replacing that with what a camera can actually measure. Lesson 05 then trains a network
to *find* the cube in the image; this lesson shows why, once you have its pixel and depth,
you also have its 3D position.

## Run it

Close any Isaac Sim window or training run first (they share the GPU), then:

```bat
cd /d "<repo>"
run.bat lesson_04_cameras\cameras.py --headless
```

It photographs a red 5 cm cube at 3 random positions with 2 cameras and saves
`lesson_04_cameras\output\shot<N>_<camera>.png` (RGB with the projected cube centre circled in
cyan, next to the depth image). The terminal prints, per camera:

```
[top] pixel (u,v)=(...)  colour there=[...]  depth=... m  recovered=[...]  error=... cm
```

Verified output on this PC (2026-10-24):

```
cameras ready after 3 steps
=== shot 0: cube centre (world, m) = [ 0.555 -0.115  0.025]
  [top] pixel (u,v)=(234.8,84.4)  colour there=[228  82  82]  depth=1.150 m  recovered=[ 0.554 -0.113  0.05 ]  error=2.5 cm
  [angled] pixel (u,v)=(231.8,128.9)  colour there=[162  10  14]  depth=1.188 m  recovered=[ 0.53  -0.112  0.05 ]  error=3.5 cm
=== shot 1: cube centre (world, m) = [ 0.316 -0.242  0.025]
  [top] pixel (u,v)=(317.1,239.3)  colour there=[220  82  83]  depth=1.150 m  recovered=[ 0.32  -0.237  0.05 ]  error=2.6 cm
  [angled] cube is outside this camera's view (u,v)=(335.0,252.4)
=== shot 2: cube centre (world, m) = [0.625 0.206 0.025]
  [top] pixel (u,v)=(25.9,38.6)  colour there=[226  57  57]  depth=1.150 m  recovered=[0.623 0.202 0.05 ]  error=2.6 cm
  [angled] pixel (u,v)=(36.2,98.7)  colour there=[136   9  14]  depth=1.240 m  recovered=[0.601 0.201 0.048]  error=3.4 cm
```

What this shows:

- **Projection works:** the colour at every projected pixel is red (`[228 82 82]`, `[162 10 14]`
  — darker for the angled camera because it sees the cube's shaded side).
- **Back-projection works:** recovered x, y are within ~0.5 cm for the top camera. The error is
  almost all in **z: 0.05 vs 0.025** — depth hits the cube's **top face**, half a cube (2.5 cm)
  above its centre. Exactly as predicted in section 4.
- **The angled camera is a bit worse (3.5 cm):** at a slant, depth hits the cube's top/front edge,
  which is offset in x as well.
- **Field of view matters:** in shot 1 the cube is at pixel (317, 239) in a 320 × 240 top image —
  the very corner — and *outside* the angled camera entirely. A robot camera must be placed to
  cover the whole workspace.

> First run on a PC can take longer before frames appear (the renderer compiles shaders);
> the script waits up to 1000 steps for the first images.

## 1. The two cameras

| Camera | Position (m) | Looking | Like… |
|--------|--------------|---------|-------|
| `top` | (0.5, 0, 1.2) | straight down (pitch 90°) | a ceiling camera over the workspace |
| `angled` | (−0.3, 0, 0.9) | forward, tilted 45° down | a robot "head" camera behind the arm |

Orientation is given as roll/pitch/yaw in degrees, in Isaac Sim's **"world" camera axes**: the
camera looks along its own +X. Pitch 90° tips that view straight down.

## 2. What the camera outputs

| Output | Shape | Meaning |
|--------|-------|---------|
| **RGB** (`get_rgb()`) | H × W × 3, values 0–255 | colour image, what a normal camera sees |
| **Depth** (`get_depth()`) | H × W, metres | for every pixel, distance from the camera's image plane to the first surface hit |

Depth must be switched on with `add_distance_to_image_plane_to_frame()`. Real equivalents:
Intel RealSense, ZED, Kinect-style RGB-D cameras.

## 3. Projection: 3D → pixel

A pinhole camera maps a world point to a pixel with:

```
pixel = K · [R | t] · point
```

- **Extrinsics `[R | t]`** — where the camera is and which way it faces (its pose).
- **Intrinsics `K`** — focal length and image centre, i.e. how the lens spreads the scene over
  the pixels (`get_intrinsics_matrix()`).

`get_image_coords_from_world_points()` does exactly this. Check it: the printed `colour there`
at the projected pixel should be **red** (high first value), and the cyan circle in the saved
image should sit on the cube.

## 4. Back-projection: pixel + depth → 3D

One pixel alone only gives a *direction* (a ray from the camera). Depth says *how far* along
that ray. Together → a 3D point (`get_world_points_from_image_coords()`).

The printed `error` compares that recovered point with the true cube centre. Expect a few cm,
not 0: depth measures the cube's **visible surface**, which is up to half a cube (2.5 cm) closer
to the camera than its centre. A real system corrects that (e.g. adds half the object size).

**This is the key idea for lesson 05:** if a network can find the cube's pixel, depth +
back-projection give its 3D position — which is exactly what the lesson 03 policy needs as input.

## 5. RGB vs depth for robots

| | RGB | Depth |
|---|---|---|
| Good for | recognising *what* something is (colour, texture, labels) | knowing *where* it is, shape, distance |
| Weak at | distance (a small near cube looks like a big far one) | colour; shiny/transparent surfaces in real sensors |
| Sim-to-real gap | big (lighting, textures differ) | smaller (geometry is geometry) |

Many robot learning systems use depth (or RGB-D) for exactly that last reason.

## Try it

1. Open the saved images. Does the cyan circle sit on the cube in both cameras?
2. Change `top` camera height from 1.2 to 2.5 m. Predict: does the cube get bigger or smaller in
   pixels? Does the recovered-position error change?
3. Set `resolution=(640, 480)`. What happens to the pixel coordinates? (Hint: they roughly double
   — the 3D answer shouldn't change.)
4. Move the cube outside `rng.uniform(0.3, 0.7)` range (e.g. x = 1.5). Which camera loses it first?

## Optional: training from images (Isaac Lab example)

Isaac Lab includes a camera-based RL task on a simple problem — balancing a pole from pixels
(`Isaac-Cartpole-RGB-Camera-Direct-v0`). Cameras need the `--enable_cameras` flag:

```bat
run.bat "%ISAACLAB%\scripts\reinforcement_learning\rsl_rl\train.py" --task Isaac-Cartpole-RGB-Camera-Direct-v0 --num_envs 32 --headless --enable_cameras
```

Not yet verified on this PC. Compare its speed to the lift training: rendering an image per env
per step is why vision RL is so much more expensive.

## Key words

**RGB** · **depth** · **RGB-D** · **pinhole camera** · **intrinsics (K)** · **extrinsics (pose)**
· **projection** · **back-projection** · **sim-to-real gap**
