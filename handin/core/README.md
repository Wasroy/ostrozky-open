# `handin/core`: shared building blocks

Strategy-free code every attempt can reuse: facts about the simulator's sensors, sticks and drag, not
choices about how to search or fly. Optional: use it, extend it, or ignore it.

```python
from handin.core import Estimator, HeadingInit, VelocityController, localise, depth_points
```

| Module | What it gives you | Measured |
|---|---|---|
| `camera.py` | Camera model shared by the detector and depth cameras (102° HFOV, 25° up-tilt, 0.30 m mount); `world_to_pixel`, `pixel_ray_cam`, `depth_points` (depth → world points, clipped pixels dropped), `localise` (detection box → world point on the box's visible face, with sigma and kind `depth` / `ground` / `bearing`) | Real detection at 15–20 m: point within 0.7 m of the box outline (`tools/tests`) |
| `estimator.py` | `Estimator`: gyro attitude (no attitude is given), VIO + accelerometer Kalman filter for position and velocity, heading correction from VIO vs IMU. `HeadingInit`: starting heading from the first VIO motion (start velocity lies along the heading), or a short forward push if slow | Over 30 s of manoeuvres on 8 official starts: heading ≤ 7.5°, tilt ≤ 5°, position ≤ 0.4 m, velocity RMS 0.3–0.5 m/s |
| `sticks.py` | `VelocityController`: desired world velocity + heading → (roll, pitch, yaw, throttle), with the simulator's drag fed forward (top speed 119.5 km/h at 60°) | Signs checked for every heading; holds 10 m/s ± 1 at constant altitude in closed loop |

State dict convention (returned by `Estimator.update`): `pos`, `vel` (world, VIO frame, z up),
`R` (world-from-body, body FLU), `rpy`, `acc`.

Typical use in `Agent.act`:

```python
s = self.est.update(obs)
if not self.heading_known:
    sticks = self.init.step(obs, s["rpy"][2])
    if sticks is not None:
        return sticks                      # still finding the heading (about 0.2-1.2 s)
    self.est.set_yaw(self.init.yaw); s = self.est.state(); self.heading_known = True
return self.ctrl.sticks(s, v_des, yaw_des)
```

Tests: `uv run --with pytest pytest -q tools/tests` (12 tests, about 1 min).
Dev tools that use it: `tools/fly.py`, `tools/check_estimator.py`.
