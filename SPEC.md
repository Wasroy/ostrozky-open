# rescuesim: behaviour specification

A small open-source simulator for a last-mile drone search task: a quadcopter starts at the
edge of an area, must find an orange box hidden among props, and touch it without touching
anything else. Written from scratch, MIT licence.

This spec describes **behaviour and numbers only**. It is the only input for the
implementation. Choose every internal design (class layout, controller code, terrain
layouts, object shapes) yourself.

## 1. Public interface (must match exactly; an existing agent depends on it)

```python
from rescuesim import RescueEnv

env = RescueEnv(terrain="forest@v3")          # "<type>@v<N>", N >= 1
obs, info = env.reset(seed=1)
obs, reward, terminated, truncated, info = env.step([roll, pitch, yaw, throttle])
env.close()
```

- Gymnasium-style 5-tuple. `reward` is 1.0 on success, else 0.0. Gymnasium itself is optional.
- `info` (debugging only, never given to the agent): `is_success` (bool), `collision` (bool),
  `outcome` (None while flying, else one of the outcome strings in §6), `distance_m` (true 3D
  distance from the drone centre to the box surface), `pos`, `vel`, `rpy` (true state, world
  frame), `t` (s).
- An agent is a class `Agent` with `reset()` (called before each flight) and `act(obs)`
  (returns 4 floats). The same instance flies several flights, so it relies on `reset()`.

### Observation dict

| Key | Type, shape | Meaning |
|---|---|---|
| `detected` | int 0 or 1 | Target detected this frame |
| `box` | float32 (4,) | `(x1, y1, x2, y2)` normalised to [0, 1] in the 160 × 90 detection image (x right, y down); zeros when not detected |
| `depth_m` | float32 (126, 224) | Forward depth image, rows top to bottom, columns left to right |
| `gyro_rad_s` | float32 (3,) | Body angular rates |
| `accel_m_s2` | float32 (3,) | Body specific force |
| `vio_xyz_m` | float32 (3,) | Noisy (x, y, altitude) relative to the arena centre |

There is no attitude, velocity, time, or target position in the observation.

### Action

`(roll, pitch, yaw, throttle)`, four floats in [-1, 1]. Anything else (wrong shape, NaN, inf,
out of range) ends the flight with outcome `error`; so does an exception in `act`.

## 2. Frames and signs

- World: x, y horizontal, z up, ground at z = 0, arena centre at (0, 0).
- Body: FLU (x forward, y left, z up).
- Sticks:
  - roll and pitch are **attitude setpoints**: ±1 is ±60°, 0 is level;
  - **pitch + = nose down**, so the drone accelerates **forward**;
  - **roll + = right wing down**, so the drone accelerates **to the right** (body −y);
  - yaw is a **rate setpoint**: yaw stick +1 = 200°/s **clockwise seen from above** (turning right),
    i.e. body z-rate setpoint = −200°/s × yaw;
  - throttle sets the collective thrust: `T = T_hover × (1 + throttle)`, clipped to what the
    motors can give. So −1 is zero thrust, 0 is hover thrust (whatever the tilt), +1 is twice
    hover thrust. Tilting at throttle 0 therefore loses height (vertical part shrinks by cos tilt).

## 3. Dynamics

- Physics at 240 Hz; one `step` = 8 physics steps (30 Hz control).
- Gravity 9.81 m/s². Rigid body, mass 1.68 kg. Thrust-to-weight 5.64 (so hover is 0.177 of
  max thrust). Choose a plausible small racing-quad geometry and inertia (about 5-inch class).
- Quadratic air drag on the body: `F = −m k |v| v` with `k = 0.0154 1/m`. Check: level flight
  at 60° pitch tops out near 33 m/s.
- No wind (keep an optional constant wind vector, default zero, and optional thrust-scale
  parameter, default 1, for robustness tests; never randomised by default).
- Controller (fixed part of the simulator, not the agent): an angle-mode cascade.
  - Attitude loop: P, gain 6 1/s, gives body-rate setpoints for roll and pitch.
  - Rate loop: P, gains 30, 30, 10 1/s on roll, pitch, yaw rate errors, gives angular
    acceleration commands.
  - Allocation to four motors with saturation and a 5% minimum (idle) motor command. Yaw gets
    only the authority left after thrust and roll/pitch.
  - Expected feel: about 1 s to tilt 30° and reach 4 m/s from hover.
  - Plain PyBullet + NumPy is enough. You may apply per-motor forces or an equivalent body
    force and torque, as long as saturation and the priorities above hold.

## 4. Sensors (sampled once per `step`, after the physics)

Use a dedicated `numpy.random.Generator` per flight, seeded from the flight seed.

- **Camera geometry (both images).** Pinhole, 102° horizontal FOV, 16:9 (vertical FOV about
  69.6°). Mounted 0.30 m ahead of the body origin, optical axis **tilted 25° up** from body x.
  Rigidly attached: rolls and pitches with the airframe. Pixel centres at
  `((i + 0.5)/W, (j + 0.5)/H)`.
- **Detector** (160 × 90 segmentation render, ground-truth mask of the target only).
  - Box = bounding box of the target's visible pixels (occlusion and image-edge clipping show
    in it), normalised to [0, 1].
  - Not reported beyond 250 m of 3D distance from the camera to the box surface.
  - A visible, in-range target is missed independently 5% of the time.
  - Each of the 4 edges gets Gaussian jitter, σ = max(0.5 px, 1% of the box size along that
    axis); clip to [0, 1] and keep x1 ≤ x2, y1 ≤ y2.
  - The detector never fires on anything else.
- **Depth** (224 × 126 render). Distance **along the camera axis** (z-depth), not along each
  ray. Clip to [0, 40] m **first**, **then** add per-pixel Gaussian noise
  σ = max(0.03, 0.03 d) m (so empty sky reads about 40 ± 1.2 m). Do not re-clip after the noise.
- **Gyro.** True body rates (FLU) + N(0, 0.002 rad/s) per axis. No bias.
- **Accelerometer.** Specific force in the body frame, `R^T (a − g_vec)`, with `a` from the
  finite difference of the true velocity between control steps; reads about [0, 0, 9.81] in a
  hover. The first reading after reset assumes zero acceleration. + N(0, 0.03 m/s²) per axis.
- **VIO.** Anchored at the true start position on reset. Each step, the horizontal estimate
  adds the true horizontal displacement plus N(0, σ) per axis with σ = 0.003 + 0.01 × ‖Δxy‖ m
  (accumulating drift). Altitude = true z + N(0, 0.05 m), no drift, measured above z = 0.

## 5. World

- Ground: 500 × 500 m plane at z = 0, centred on (0, 0). Arena limit |x|, |y| < 350 m.
- **Target:** one static box, 4 × 1.65 × 1.55 m (length × width × height), resting on the
  ground, random yaw, orange (about RGB 1.0, 0.45, 0.0). Exactly one per flight.
- **Props** stand within 150 m of the centre in x and y (rendering budget), plus anything
  needed near the target. Static. Every prop is a collision body.
- **Terrain types** (your own procedural designs, built from primitives: boxes, cylinders,
  spheres, capsules; no external mesh files needed):
  - `field`: open farmland, hedgerows, a few trees and sheds, hay bales, rocks;
  - `forest`: dense trees (trunk + canopy that the drone can hit), undergrowth bushes, rocks,
    clearings, trails;
  - `village`: houses, sheds, fences, hedges, cars, gardens, a few streets;
  - `city`: street grid, tall blocks, alleys, courtyards, parked cars, a park;
  - `rubble` (new): a damaged area with collapsed walls, debris piles, wrecked cars and craters,
    to test generalisation.
- **Layouts and poses.** Each terrain type has one layout per `layout_seed` (default fixed per
  type, so `forest@v3` is always the same world) and a list of at least 40 target poses
  (`v1`… `v40`). Choose poses that are **hidden but reachable**: beside walls, under canopies,
  in alleys, among cars, in or by bushes, against rocks. No prop may overlap the box (bushes may
  touch it); any two poses at least 12 m apart; at least one free approach corridor about 3 m
  wide from above or from a side. Also expose `generate_terrain(type, layout_seed, pose_index)`
  so new unseen layouts can be generated for testing generalisation.
- Keep the scenes light enough for PyBullet's CPU renderer (TinyRenderer, DIRECT mode): target
  at least 40 control steps per second per core with both cameras on the densest terrain.

## 6. Flights and outcomes

- **Start:** horizontal distance 225–275 m from the centre, altitude 5–25 m, heading within
  ±20° of pointing at the centre, speed 0–10 m/s along the heading, level attitude, zero body
  rates. Redraw until no body is within 2 m of the drone.
- Seeding: a flight is fully reproducible from `(seed, terrain string, run index)`; use
  `numpy.random.SeedSequence([seed, stable_hash(terrain), run])` (stable across processes, not
  Python's `hash`).
- Outcomes, checked after each step, first one wins:

| Outcome | Condition |
|---|---|
| `success` | drone in contact with the target box and with nothing else |
| `collision` | contact with any other body, or true altitude < 0.2 m |
| `out_of_arena` | \|x\| or \|y\| ≥ 350 m |
| `timeout` | 120 s of simulated time (configurable) |
| `error` | `act` raised, or returned an invalid action |

Contact speed does not matter.

## 7. Evaluation script

`python -m rescuesim.evaluate --agent <dir> [--terrains field forest ...] [--poses N]
[--runs R] [--seed S] [--workers W] [--timeout 120] [--out DIR]`

- `<dir>` is a folder that is importable as a package and contains `agent.py` with class
  `Agent` (for example `--agent path/to/handin` makes `handin` importable from its parent and
  imports `handin.agent`). Imports inside the agent like `from handin.core import ...` must work.
- Flies every pose `v1..vN` of each chosen terrain, R runs each, in parallel processes (one env
  and one `Agent` per process job), each flight seeded as in §6.
- Measure the wall time spent inside `act` per step (latency); warn if the mean exceeds 33 ms.
- Writes `DIR/flights.csv` (one row per flight: terrain, pose, run, seed, start state, outcome,
  time, path length, first detection time, final distance, mean latency) and `DIR/metrics.json`
  per terrain type and overall, with these exact keys:
  `success_rate`, `collision_rate`, `timeout_rate`, `out_of_arena_rate`, `error_rate`,
  `reach_time_s`, `reach_mean_speed_m_s`, `path_efficiency` (start distance to the box ÷ path
  flown, successful flights), `detection_rate`, `first_detection_s`, `failed_final_distance_m`,
  `latency_mean_ms`, `agent_fps`, plus `flights`.
- Prints a summary table per terrain (flights, success, crash, timeout, out, error, mean reach
  time, detection rate, latency).
- Optional `--log` writes one compressed `.npz` per flight (true pos/vel/rpy, actions,
  detected, box, vio, latency per step; depth every 30th step).

Also provide `python -m rescuesim.check_agent --agent <dir>`: 10 steps on `village@v1`, checks
the action shape, range and finiteness, prints "Agent API OK".

## 8. Packaging

- Python 3.12, dependencies `numpy` and `pybullet` only (plus `pytest` for tests).
- `pyproject.toml`, `LICENSE` (MIT), `README.md` with the task, interface, sensors, outcomes,
  how to run, and the known limits (synthetic detector from ground-truth segmentation, Gaussian
  depth noise, synthetic VIO, approximate controller, no wind by default).
- `tests/`: sign conventions (pitch+ → forward, roll+ → right, yaw+ → clockwise, throttle 0
  holds height level), hover reads accel ≈ [0, 0, 9.81], depth shape and clip-then-noise,
  detector range and miss rate, VIO anchoring, outcome detection (collision with a prop,
  success on the box, out of arena, timeout), reproducibility from the seed, terrain pose
  constraints (no overlap, 12 m spacing), and that a hovering stub agent passes check_agent.
