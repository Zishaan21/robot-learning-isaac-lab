# Lesson 01 — Hello Isaac

**Goal:** understand the skeleton every Isaac Sim Python script shares, by dropping a cube on a floor.

## Run it

```bat
cd "<repo>"
.\run.bat lesson_01_hello_isaac\hello_isaac.py --headless
```

Expected output (verified on this PC):

```
step 0: cube z = 0.992
step 50: cube z = 0.125
step 100: cube z = 0.100
...
DONE - cube landed at z = 0.1
```

Drop `--headless` to watch it in a window.

## The 4 building blocks

Open [hello_isaac.py](hello_isaac.py) alongside this.

1. **`SimulationApp`** — starts the simulator. It **must be created before any other `isaacsim`
   import**, because those modules only exist once the app is running.
2. **`World`** — the simulation container. It owns the physics scene (gravity, time step) and the
   `scene` you add things to.
3. **Objects** — `add_default_ground_plane()` gives a floor *with collision*.
   `DynamicCuboid` is a cube with a **rigid body** (gravity moves it) and a **collider**
   (it can hit things). Without either, it would float or fall through the floor.
4. **The loop** — `world.step()` advances physics by one small time step (1/60 s by default).
   Everything in robotics simulation is: *read state → decide → act → step*.

## Why z ends at 0.1

The cube is 0.2 m tall and its position is its **centre**, so resting on the floor (z = 0) its
centre sits at 0.1 m. Reading positions like this is how a robot "sees" the world in simulation.

## Try it

1. Change `position=[0, 0, 1.0]` to `[0, 0, 3.0]` — how many steps until it lands now?
2. Change `size=0.2` to `0.5` — what is the final z? (Predict first.)
3. Add a second cube at `[0.05, 0, 2.0]` so it lands on the first one.

## Key words

**rigid body** · **collider** · **physics step** · **headless** · **prim path** (`/World/cube` —
the object's address in the scene tree, like a file path)
