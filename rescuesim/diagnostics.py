"""Measurements used to validate the simulator against the spec.

    python -m rescuesim.diagnostics [--dynamics] [--speed] [--visibility] [--agent DIR]

--dynamics   sign conventions, hover accelerometer, tilt/speed timing, top speed, sky depth
--speed      control steps per second (both cameras) per terrain, one core
--visibility render-based hidden/visible check of every target pose
--agent DIR  also time full flights of that agent on each terrain (--speed)
With no flag, everything runs.
"""
from __future__ import annotations

import argparse
import math
import time

import numpy as np

from .env import RescueEnv
from .sensors import aimed_camera
from .terrain import TERRAIN_TYPES, get_layout
from .world import CollisionWorld, PoseValidator, RenderWorld

_WORLDS: dict = {}


def _worlds(layout):
    key = (layout.kind, layout.layout_seed)
    if key not in _WORLDS:
        _WORLDS[key] = (RenderWorld(list(layout.prims), layout.ground_rgba), PoseValidator(CollisionWorld(list(layout.prims))))
    return _WORLDS[key]


def pose_visibility(layout, pose_index: int) -> dict:
    """Detector visibility of pose ``v<pose_index>`` from a ring of viewpoints aimed at the box.

    low:    8 azimuths x altitudes 2 m and 4 m, 25 m horizontal distance
    others: 8 azimuths at 60 m / 15 m alt, 8 at 40 m / 25 m alt, one 45 m overhead
    Viewpoints inside a prop are skipped."""
    rw, val = _worlds(layout)
    pose = layout.poses[pose_index - 1]
    rw.set_target(pose)
    centre = np.array(pose.center)

    def ring(dist, alt, n=8):
        a = pose.yaw + np.arange(n) * 2 * math.pi / n
        return [np.array([pose.x + dist * math.cos(b), pose.y + dist * math.sin(b), alt]) for b in a]

    def visible(eye):
        mask, _, _ = rw.render_target_mask(aimed_camera(eye, centre))
        return mask is not None and bool(mask.any())

    low = ring(25, 2.0) + ring(25, 4.0)
    others = ring(60, 15.0) + ring(40, 25.0) + [np.array([pose.x + 0.01, pose.y, 45.0])]
    res = {"low_free": 0, "low_visible": 0, "free": 0, "visible": 0}
    for i, eye in enumerate(low + others):
        if not val.free_point(eye):          # same free-space test as the pose selection
            continue
        v = visible(eye)
        res["free"] += 1
        res["visible"] += v
        if i < len(low):
            res["low_free"] += 1
            res["low_visible"] += v
    res["hidden_any"] = res["free"] - res["visible"]
    return res


# ----------------------------------------------------------------------------
def dynamics_report():
    env = RescueEnv("empty@v1", render_depth=False, render_detector=False)
    out = {}

    def fly(action, secs, pos=(0, -200, 30), yaw=0.0):
        obs, info = env.reset(seed=0, options={"start_pos": pos, "start_yaw": yaw, "start_speed": 0.0})
        hist = [(obs, info)]
        for _ in range(int(round(secs * 30))):
            obs, _, term, trunc, info = env.step(action)
            hist.append((obs, info))
            if term or trunc:
                break
        return hist

    h = fly([0, 0, 0, 0], 3)
    out["hover_accel_mean"] = np.mean([o["accel_m_s2"] for o, _ in h], 0).round(3).tolist()
    out["hover_dz_3s_m"] = round(float(h[-1][1]["pos"][2] - 30), 4)
    h = fly([0, 0.5, 0, 0], 2)
    out["pitch+0.5 after 1 s: vx, vy (m/s)"] = h[30][1]["vel"][:2].round(2).tolist()
    out["time to 29.5 deg tilt (s)"] = round(next(i["t"] for _, i in h if i["rpy"][1] >= math.radians(29.5)), 3)
    out["time to 4 m/s (s)"] = round(next(i["t"] for _, i in h if i["vel"][0] >= 4.0), 3)
    h = fly([0.5, 0, 0, 0], 1)
    out["roll+0.5 after 1 s: vx, vy (m/s)"] = h[-1][1]["vel"][:2].round(2).tolist()
    h = fly([0, 0, 0.5, 0], 1)
    out["yaw+0.5 after 1 s: heading change (deg)"] = round(math.degrees(h[-1][1]["rpy"][2]), 1)
    h = fly([0, 1, 0, 1], 25, pos=(-300, 0, 60))
    v = h[-1][1]["vel"]
    out["pitch+1 throttle+1 after 25 s: speed, vz (m/s)"] = [round(math.hypot(v[0], v[1]), 2), round(float(v[2]), 3)]
    env.close()
    env = RescueEnv("empty@v1")
    obs, _ = env.reset(seed=0, options={"start_pos": (0, -200, 20), "start_yaw": 0.0})
    d = np.concatenate([env.step([0, 0, 0, 0])[0]["depth_m"].ravel() for _ in range(10)])
    out["sky depth mean, std (m)"] = [round(float(d.mean()), 3), round(float(d.std()), 3)]
    env.close()
    return out


def speed_report(kinds=TERRAIN_TYPES, flights=40, steps=30, agent_dir=None):
    rows = {}
    for kind in kinds:
        env = RescueEnv(f"{kind}@v1")
        rng = np.random.default_rng(1)
        n, tt = 0, 0.0
        for k in range(flights):
            tgt = env.terrain.poses[k % len(env.terrain.poses)]
            a, d, z = rng.uniform(0, 2 * math.pi), rng.uniform(15, 60), [2.5, 6.0, 15.0][k % 3]
            pos = (tgt.x + d * math.cos(a), tgt.y + d * math.sin(a), z)
            env.reset(seed=k, options={"start_pos": pos, "start_yaw": a + math.pi + rng.normal(0, 0.3)})
            t0 = time.perf_counter()
            for _ in range(steps):
                _, _, term, trunc, _ = env.step([0, 0.15, 0.1, 0.03])
                n += 1
                if term or trunc:
                    break
            tt += time.perf_counter() - t0
        row = {"triangles": env.rworld.n_triangles, "visual_bodies": len(env.rworld.bodies),
               "collision_bodies": len(env.cworld.prop_bodies), "prims": len(env.terrain.prims),
               "probe_steps_per_s": round(n / tt, 1)}
        if agent_dir:
            from .agent_loader import load_agent_class
            agent = load_agent_class(agent_dir)()
            n, tt = 0, 0.0
            for pose in (1, 2, 3):
                e2 = RescueEnv(f"{kind}@v{pose}")
                obs, _ = e2.reset(seed=0)
                agent.reset()
                t0 = time.perf_counter()
                while True:
                    obs, _, term, trunc, _ = e2.step(agent.act(obs))
                    n += 1
                    if term or trunc:
                        break
                tt += time.perf_counter() - t0
                e2.close()
            row["agent_flight_steps_per_s"] = round(n / tt, 1)
            row["agent_flight_steps"] = n
        rows[kind] = row
        env.close()
    return rows


def visibility_report(kinds=TERRAIN_TYPES):
    rows = {}
    for kind in kinds:
        lay = get_layout(kind)
        low_ok = hid_ok = 0
        for k in range(len(lay.poses)):
            r = pose_visibility(lay, k + 1)
            low_ok += r["low_visible"] >= 1
            hid_ok += r["hidden_any"] >= 1
        rows[kind] = {"poses": len(lay.poses), "visible_from_a_low_25m_viewpoint": low_ok,
                      "hidden_from_some_viewpoint": hid_ok, "approach": _count(p.approach for p in lay.poses),
                      "kinds": _count(p.kind for p in lay.poses)}
    return rows


def _count(it):
    out = {}
    for x in it:
        out[x] = out.get(x, 0) + 1
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dynamics", action="store_true")
    ap.add_argument("--speed", action="store_true")
    ap.add_argument("--visibility", action="store_true")
    ap.add_argument("--agent", default=None)
    ap.add_argument("--terrains", nargs="+", default=list(TERRAIN_TYPES))
    args = ap.parse_args(argv)
    every = not (args.dynamics or args.speed or args.visibility)
    if args.dynamics or every:
        print("== dynamics and sensors (empty world)")
        for k, v in dynamics_report().items():
            print(f"  {k}: {v}")
    if args.speed or every:
        print("== speed (one core, both cameras)")
        for k, v in speed_report(args.terrains, agent_dir=args.agent).items():
            print(f"  {k:8s} {v}")
    if args.visibility or every:
        print("== target pose visibility (detector renders)")
        for k, v in visibility_report(args.terrains).items():
            print(f"  {k:8s} {v}")


if __name__ == "__main__":
    main()
