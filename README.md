# rescuesim

A small open-source simulator for a last-mile drone search task. A quadcopter starts
225-275 m from the centre of an area, has to find an orange box (4 x 1.65 x 1.55 m)
hidden among props, and touch it without touching anything else.

Python 3.12, NumPy and PyBullet (CPU TinyRenderer, DIRECT mode). MIT licence.

```
pip install -e .            # or just put this folder on PYTHONPATH
pip install pytest          # for the tests (matplotlib optional, for previews)
```

## Interface

```python
from rescuesim import RescueEnv

env = RescueEnv(terrain="forest@v3")      # "<type>@v<N>", N >= 1
obs, info = env.reset(seed=1)
obs, reward, terminated, truncated, info = env.step([roll, pitch, yaw, throttle])
env.close()
```

Gymnasium-style 5-tuple; `reward` is 1.0 on success, else 0.0. `info` (debugging only,
never give it to the agent): `is_success`, `collision`, `outcome` (None while flying),
`distance_m` (drone centre to box surface), true `pos`, `vel`, `rpy` (world frame), `t`.

Optional constructor arguments: `layout_seed` (None = the fixed default layout of that
type, an int = a new unseen layout), `run` (run index for seeding), `max_time_s` (120),
`wind` (constant world-frame wind, default zero), `thrust_scale` (default 1; the
controller does not know about it), `render_depth` / `render_detector` (switch the cameras
off for fast dynamics-only work). `reset(seed, options={"run": r})` selects the run;
`options` may also contain `start_pos`, `start_yaw`, `start_speed` (tests, debugging).

An agent is a package folder with `agent.py` defining `class Agent` with `reset()` (called
before every flight; one instance flies several flights) and `act(obs)` returning 4 floats.

### Observation

| key | type, shape | meaning |
|---|---|---|
| `detected` | int 0/1 | target detected this frame |
| `box` | float32 (4,) | `(x1, y1, x2, y2)` in [0, 1] of the 160 x 90 detector image (x right, y down); zeros if not detected |
| `depth_m` | float32 (126, 224) | forward depth image (z-depth), rows top to bottom |
| `gyro_rad_s` | float32 (3,) | body rates (FLU) |
| `accel_m_s2` | float32 (3,) | body specific force (about [0, 0, 9.81] in a hover) |
| `vio_xyz_m` | float32 (3,) | noisy x, y (drifting) and altitude relative to the arena centre |

### Action

`(roll, pitch, yaw, throttle)` in [-1, 1]; anything else (shape, NaN, inf, range) ends the
flight with outcome `error`.

* roll, pitch: attitude setpoints, +-1 = +-60 deg. **pitch + = nose down = accelerate
  forward**, **roll + = right wing down = accelerate right**.
* yaw: rate setpoint, +1 = 200 deg/s **clockwise seen from above**.
* throttle: collective `T = T_hover (1 + throttle)`, limited by the motors: -1 = idle,
  0 = hover thrust whatever the tilt (tilting at 0 loses height), +1 = twice hover thrust.

## Simulation

* Body FLU, world z up, ground at z = 0. 240 Hz physics, one `step` = 8 physics steps (30 Hz).
* Mass 1.68 kg, thrust-to-weight 5.64, X-quad with motors 78 mm off-axis in x and y
  (5-inch class), inertia (3.0, 3.0, 5.5) g m^2, quadratic drag `F = -m k |v| v`,
  k = 0.0154 1/m.
* Fixed angle-mode controller: attitude P (6 1/s) -> body-rate setpoints, rate P
  (30, 30, 10 1/s) -> angular accelerations -> torques -> four motors with 5 % idle and
  saturation. Roll/pitch keep priority (collective shifted if needed); yaw only gets what is
  left.
* Measured feel: 30 deg tilt in 0.5 s and 4 m/s in 1.0 s from hover at pitch 0.5; level
  flight at 60 deg (pitch 1, throttle 1) tops out at 33.2 m/s.

### Sensors (once per step, after the physics; one NumPy Generator per flight)

* Both cameras: pinhole, 102 deg horizontal FOV, 16:9 (69.6 deg vertical), 0.30 m ahead of
  the body origin, optical axis tilted 25 deg up, rigidly attached. Pixel (i, j) samples the
  direction of its centre ((i + 0.5)/W, (j + 0.5)/H).
* Detector: ground-truth segmentation render (160 x 90) of the target only; box = bounding
  box of the visible target pixels (occlusion and image-edge clipping included); nothing
  beyond 250 m (camera to box surface); 5 % independent misses; Gaussian jitter on each
  edge, sigma = max(0.5 px, 1 % of the box size on that axis), clipped to [0, 1], ordered.
* Depth: z-depth render (224 x 126), clipped to [0, 40] m, then noise
  sigma = max(0.03, 0.03 d) m, not re-clipped (empty sky reads about 40 +- 1.2 m).
* Gyro: true rates + N(0, 0.002 rad/s). Accelerometer: `R^T (a - g)` with `a` from the
  velocity difference between steps (zero on the first reading) + N(0, 0.03 m/s^2).
* VIO: anchored at the true start position; each step adds the true horizontal
  displacement plus N(0, 0.003 + 0.01 |dxy|) per axis; altitude = true z + N(0, 0.05 m).

## World and terrains

Ground 500 x 500 m; arena limit |x|, |y| < 350 m; props within 150 m of the centre. Five
procedural terrain types built from boxes, spheres, cylinders and capsules:

| type | content |
|---|---|
| `field` | farmland plots, hedgerows with gaps, trees, farmsteads (barns, sheds, bale stacks, tractors), round bales, rocks, bushes |
| `forest` | ~1400 deciduous and conifer trees (trunks and canopies are solid), undergrowth, rocks, fallen logs, clearings, trails |
| `village` | streets, plots with gabled houses, fences, hedges, sheds, garden trees and bushes, parked and driveway cars |
| `city` | 5 x 5 street grid, perimeter blocks with alleys and courtyards, towers with plazas, parked cars, a park |
| `rubble` | ruined shells with gaps, collapsed slabs, debris piles, craters, wrecked (tilted, flipped) cars, burnt trees |

Every layout is fixed per `layout_seed` (each type has a default, so `forest@v3` is always
the same world) and has 48 target poses `v1..v48`. Poses are chosen automatically from
hiding-spot candidates (beside walls, hedges, fences and rocks, under canopies, in alleys
and courtyards, among parked cars, in bushes, in craters) and kept only if:

* no prop except bushes overlaps or touches the box (8 cm clearance), poses >= 12 m apart;
* a free ~3 m corridor exists (a 3 x 3 m shaft from above, or a 3 m wide, 0.3-3 m high,
  25 m long side corridor in one of 8 directions);
* a detector render sees the box from at least one low viewpoint (2 or 4 m altitude,
  25 m away) and does not see it from at least one other viewpoint.

`generate_terrain(type, layout_seed=None, pose_index=1)` returns the layout (primitives,
poses, target) for any seed, so unseen layouts can be generated for generalisation tests;
`python -m rescuesim.preview` writes top-down PNGs (see `previews/`).

## Flights and outcomes

Start: 225-275 m from the centre, altitude 5-25 m, heading within +-20 deg of the centre,
speed 0-10 m/s along the heading, level, zero rates; redrawn until nothing is within 2 m.
Every flight is reproducible from `numpy.random.SeedSequence([seed, crc32(terrain), run])`.

Outcomes, checked after every step (contacts at every physics sub-step), first match wins:
`success` (touching the box and nothing else), `collision` (touching anything else, or
altitude < 0.2 m), `out_of_arena` (|x| or |y| >= 350 m), `timeout` (120 s, truncated),
`error` (`act` raised or returned an invalid action). Contact speed does not matter.

## Running

```
python -m rescuesim.check_agent --agent examples/hover_agent      # 10 steps on village@v1
python -m rescuesim.evaluate --agent examples/simple_agent --terrains field forest \
       --poses 5 --runs 1 --workers 8 --out eval_out [--seed 0] [--timeout 120] [--log]
python -m rescuesim.diagnostics                                   # dynamics, speed, visibility numbers
python -m rescuesim.preview --out previews                        # top-down layout images
python -m pytest
```

`evaluate` imports `<dir>/agent.py` as `<dirname>.agent` with the parent folder on
`sys.path` (so `from handin.core import ...` works inside `--agent path/to/handin`), runs
one env and one `Agent` per (terrain, pose) job in parallel processes, times `act`
(warning if the mean exceeds 33 ms), and writes `flights.csv` and `metrics.json`
(`success_rate`, `collision_rate`, `timeout_rate`, `out_of_arena_rate`, `error_rate`,
`reach_time_s`, `reach_mean_speed_m_s`, `path_efficiency`, `detection_rate`,
`first_detection_s`, `failed_final_distance_m`, `latency_mean_ms`, `agent_fps`, `flights`;
per terrain type under `terrains` and under `overall`). `--log` writes one compressed
`.npz` per flight.

## Performance notes

TinyRenderer's cost grows with every triangle in the scene, on screen or not. Props are
therefore merged into one low-poly mesh per (20 m tile, colour), tiles outside the camera
frustum are switched off for each render, and the detector image is rendered only over the
pixel window the target can cover, with only the tiles between camera and target. Both
shortcuts give pixel-identical images (tested). Round shapes are drawn as low-poly hulls
that circumscribe the collision primitive (icosahedra for spheres, octagonal prisms for
cylinders), so depth never shows free space where there is a collision surface.

## GPU simulator (training)

`gpusim/` runs thousands of rescuesim flights at once on an NVIDIA GPU (NVIDIA Warp), for
training policies. The GPU engine was originally written by @sckoott (OSTROZKY team) for the
EDTH Paris 2026 hackathon and has been ported to rescuesim, which is its reference: same
terrains (built by rescuesim itself), dynamics and controller, cameras, sensor noise models,
starts (identical for the same `(seed, terrain, run)`), and outcome rules.

```
pip install -e .[gpu]                      # warp-lang >= 1.19, CUDA GPU
```

```python
from gpusim import FastBatchEnv

env = FastBatchEnv(n=1024, terrains=("field", "forest", "village", "city", "rubble"),
                   seed=0, obs="pooled", noise=True, timeout_s=120)
obs = env.reset()                          # or reset(idx, terrains=["city@v3", ...], runs=[...])
obs, done, outcome, info = env.step(actions)   # actions (N, 4); outcome codes in gpusim.OUTCOMES
env.reset(idx=done.nonzero()[0])           # finished flights stay frozen until reset
one = env.single(obs, 0)                   # rescuesim observation dict (obs="full" for depth_m)
```

One fused kernel per step does the 8 physics sub-steps and the contact test of every drone;
depth (with its noise, min-pooled to 9 x 16 for `obs="pooled"`) and the detector box are
ray-cast on the GPU against the same triangles rescuesim renders; IMU, VIO, detector misses
and jitter are applied on the host. `gpusim.dynamics` is the same dynamics in batched NumPy
(no Warp needed).

Measured with `python -m gpusim.validate` (RTX 3050 Laptop, 4 GB):

| check | result |
|---|---|
| open-loop dynamics, 8 manoeuvres x 4 s | max 0.03 cm position, 0.01 deg attitude error (float32); NumPy version exact |
| contacts, 1000 straight paths into props / boxes (821 touch) | first contact on the same 1/240 s sub-step with the same body in 98.8 %, 1 sub-step early (sub-mm grazing) in 1.0 %, other 0.2 % |
| rendering, 150 poses on the 5 terrains, noise off | depth median error 0.000 cm, p99 0.14 cm, sky/hit mismatch 0.001 %; detector box pixel-identical in 75/75 frames |
| end to end, team agent (`handin`), 50 flights (5 terrains x v1..v10) | identical starts; success 90 % rescuesim, 84 % gpusim; outcome agreement 82 % (gpusim noise), 88 % (`noise="rescuesim"`: rescuesim's own noise samples); rescuesim against itself with other noise samples: 84 % success, 86 % agreement |
| speed, `obs="pooled"` | 24 k (n = 256), 30 k (n = 1024 and 4096) drone-steps/s |
| speed, `obs="full"` | 17 k (n = 256 and 1024) drone-steps/s |

```
python -m gpusim.validate                  # everything (~15 min; e2e uses all CPU cores)
python -m gpusim.validate --only dynamics render speed --quick
python -m pytest tests/test_gpusim.py      # skipped without warp-lang / CUDA
```

The end-to-end disagreements are not a bias but the chaos of closed-loop flight: fed the same
noise samples, a gpusim flight stays within 1 mm of the rescuesim flight for 5-30 s, until a few
silhouette pixels of the depth image (ray-cast vs rasterised edges; e.g. 4 of 28224) flip a
discrete decision of the agent; with other noise, rescuesim disagrees with itself as often. On
96 city flights (the most sensitive terrain), success is 40 % in rescuesim and 34 % in gpusim.

Differences from rescuesim: the noise of a flight is drawn by a counter-based generator keyed by
`(seed, terrain, run, step)` (reproducible per flight whatever the batch, same distributions,
not the same samples as rescuesim's sequential Generator), and contacts are computed
analytically against the exact primitives, where PyBullet reports a grazing contact up to
~0.7 mm later. gpusim reads back which props collide in rescuesim's PyBullet world, so the two
always agree; every collidable prop does collide (rescuesim splits its compound bodies into
chunks of 16, PyBullet's per-compound limit, and `tests/test_terrain.py` checks it).

## Known limits

* The detector is synthetic: ground-truth segmentation plus misses and jitter; no false
  positives, no appearance or lighting effects.
* Depth noise is independent Gaussian per pixel; no holes, edge artefacts or range bias.
* VIO is synthetic (random-walk drift on xy, white noise on altitude), not a vision pipeline.
* The flight controller and motor model are approximate (P loops, instantaneous motors, no
  battery sag, no rotor drag or ground effect); collisions end the flight, there is no
  bounce or contact physics.
* No wind by default (constant wind and thrust scale are available for robustness tests).
* Visual shapes of round props are slightly larger than their collision shapes (low-poly
  circumscribed hulls).
