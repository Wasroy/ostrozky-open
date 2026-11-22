"""RescueEnv: the Gymnasium-style search-and-touch environment."""
from __future__ import annotations

import math

import numpy as np

from . import sensors as S
from .dynamics import QuadParams, Quadrotor
from .geometry import euler_from_matrix, point_box_distance, quat_to_matrix
from .scene import ARENA_HALF_M, TARGET_HALF
from .terrain import DEFAULT_LAYOUT_SEEDS, generate_terrain, parse_terrain, stable_hash
from .world import DRONE_REACH_M, CollisionWorld, RenderWorld

PHYSICS_HZ = 240
CONTROL_HZ = 30
SUBSTEPS = PHYSICS_HZ // CONTROL_HZ
DT_PHYS = 1.0 / PHYSICS_HZ
DT_CTRL = 1.0 / CONTROL_HZ
MIN_ALTITUDE_M = 0.2
OUTCOMES = ("success", "collision", "out_of_arena", "timeout", "error")


class RescueEnv:
    """Find the orange box and touch it, touching nothing else.

    Parameters
    ----------
    terrain : "<type>@v<N>" (types: field, forest, village, city, rubble).
    layout_seed : None for the fixed default layout of the type, or an int for
        a different (unseen) layout of the same type.
    run : run index used in the flight seed (``reset(options={"run": r})`` overrides).
    max_time_s : flight time limit (timeout).
    wind : constant wind vector (m/s, world frame); default zero.
    thrust_scale : multiplies the real motor thrust (controller is unaware); default 1.
    render_depth / render_detector : turn the cameras off for fast dynamics-only runs.
    """

    metadata = {"render_modes": []}

    def __init__(self, terrain: str = "village@v1", layout_seed: int | None = None, run: int = 0,
                 max_time_s: float = 120.0, wind=(0.0, 0.0, 0.0), thrust_scale: float = 1.0,
                 render_depth: bool = True, render_detector: bool = True, params: QuadParams | None = None):
        kind, idx = parse_terrain(terrain)
        self.terrain_kind, self.pose_index = kind, idx
        self.layout_seed = layout_seed
        self.terrain = generate_terrain(kind, layout_seed, idx)
        self.terrain_str = self.terrain.name
        if layout_seed is not None and layout_seed != DEFAULT_LAYOUT_SEEDS[kind]:
            self.terrain_str = f"{kind}#L{layout_seed}@v{idx}"
        self.default_run = int(run)
        self.run = self.default_run
        self.max_time_s = float(max_time_s)
        self.render_depth_on = render_depth
        self.render_detector_on = render_detector
        self.quad = Quadrotor(params, wind, thrust_scale)
        self.cworld = CollisionWorld(list(self.terrain.prims), self.quad.p.collision_half_extents)
        self.rworld = RenderWorld(list(self.terrain.prims), self.terrain.layout.ground_rgba)
        self.target = self.terrain.target
        self.cworld.set_target(self.target)
        self.rworld.set_target(self.target)
        self.rng = np.random.default_rng(0)
        self.seed = None
        self.t = 0.0
        self.steps = 0
        self.outcome = None
        self.collision = False
        self._last = None
        self.start = None

    # ------------------------------------------------------------------ API
    def reset(self, seed: int | None = None, options: dict | None = None):
        options = dict(options or {})
        if seed is None:
            seed = int(np.random.SeedSequence().entropy % (2 ** 31)) if self.seed is None else self.seed + 1
        self.seed = int(seed)
        run = int(options.get("run", self.default_run))
        self.run = run
        self.rng = np.random.default_rng(np.random.SeedSequence([self.seed, stable_hash(self.terrain_str), run]))
        pos, yaw, speed = self._draw_start()
        if "start_pos" in options:
            pos = np.asarray(options["start_pos"], float)
        if "start_yaw" in options:
            yaw = float(options["start_yaw"])
        if "start_speed" in options:
            speed = float(options["start_speed"])
        self.quad.reset(pos, yaw, speed)
        self.start = {"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2]), "yaw": float(yaw), "speed": float(speed)}
        self.t, self.steps, self.outcome, self.collision = 0.0, 0, None, False
        self.imu = S.IMU(self.rng, DT_CTRL)
        self.imu.reset(self.quad.state.vel)
        self.vio = S.VIO(self.rng)
        self.vio.reset(self.quad.state.pos)
        obs = self._observe(first=True)
        info = self._info()
        self._last = (obs, info)
        return obs, info

    def step(self, action):
        if self.outcome is not None:
            obs, info = self._last
            return obs, 0.0, self.outcome != "timeout", self.outcome == "timeout", info
        act = self._validate(action)
        if act is None:
            return self.fail("error")
        st = self.quad.state
        prev_pos = st.pos.copy()
        positions = []
        quats = []
        for _ in range(SUBSTEPS):
            self.quad.substep(act, DT_PHYS)
            positions.append(st.pos.copy())
            quats.append(st.quat)
        self.t += DT_CTRL
        self.steps += 1
        contacts = self._sweep_contacts(prev_pos, positions, quats)
        obs = self._observe()
        self.outcome = self._outcome(contacts)
        info = self._info()
        self._last = (obs, info)
        reward = 1.0 if self.outcome == "success" else 0.0
        terminated = self.outcome is not None and self.outcome != "timeout"
        truncated = self.outcome == "timeout"
        return obs, reward, terminated, truncated, info

    def fail(self, outcome: str = "error"):
        """End the flight immediately (used by the evaluator when ``act`` raises)."""
        if self._last is None:
            raise RuntimeError("call reset() first")
        self.outcome = outcome
        obs, _ = self._last
        info = self._info()
        self._last = (obs, info)
        return obs, 0.0, True, False, info

    def close(self):
        self.cworld.close()
        self.rworld.close()

    # ------------------------------------------------------------- internals
    @staticmethod
    def _validate(action):
        try:
            a = np.asarray(action, dtype=float)
        except (TypeError, ValueError):
            return None
        if a.shape != (4,) or not np.all(np.isfinite(a)) or np.any(np.abs(a) > 1.0):
            return None
        return tuple(float(x) for x in a)

    def _draw_start(self):
        rng = self.rng
        for _ in range(1000):
            r = rng.uniform(225.0, 275.0)
            bearing = rng.uniform(-math.pi, math.pi)
            z = rng.uniform(5.0, 25.0)
            heading = bearing + math.pi + rng.uniform(-math.radians(20), math.radians(20))
            speed = rng.uniform(0.0, 10.0)
            pos = np.array([r * math.cos(bearing), r * math.sin(bearing), z])
            if self.cworld.clearance_ok(pos, 2.0):
                return pos, math.atan2(math.sin(heading), math.cos(heading)), speed
        raise RuntimeError("could not find a free start position")

    def _sweep_contacts(self, prev_pos, positions, quats) -> set:
        lo = np.minimum(prev_pos, np.min(positions, axis=0)) - DRONE_REACH_M
        hi = np.maximum(prev_pos, np.max(positions, axis=0)) + DRONE_REACH_M
        cands = self.cworld.bodies_near(lo, hi)
        if not cands:
            return set()
        for k, (pos, q) in enumerate(zip(positions, quats)):
            hits = self.cworld.drone_contacts(pos, q, cands)
            if hits:
                # freeze the state at the first touching sub-step
                st = self.quad.state
                st.pos = np.array(pos)
                st.quat = q
                return hits
        return set()

    def _outcome(self, contacts: set):
        if contacts and contacts == {self.cworld.target}:
            return "success"
        z = self.quad.state.pos[2]
        if contacts or z < MIN_ALTITUDE_M:
            self.collision = True
            return "collision"
        x, y = self.quad.state.pos[:2]
        if abs(x) >= ARENA_HALF_M or abs(y) >= ARENA_HALF_M:
            return "out_of_arena"
        if self.t >= self.max_time_s - 1e-9:
            return "timeout"
        return None

    def camera(self):
        st = self.quad.state
        return S.camera_frame(st.pos, quat_to_matrix(st.quat))

    def _observe(self, first: bool = False) -> dict:
        st = self.quad.state
        R = quat_to_matrix(st.quat)
        cam = S.camera_frame(st.pos, R)
        if self.render_depth_on:
            depth = S.noisy_depth(self.rworld.render_depth(cam), self.rng)
        else:
            depth = np.full((S.DEPTH_H, S.DEPTH_W), S.DEPTH_MAX_M, np.float32)
        detected, box = 0, np.zeros(4, np.float32)
        if self.render_detector_on:
            mask, off, dist = self.rworld.render_target_mask(cam)
            detected, box = S.detect_from_mask(mask, off, dist, self.rng)
        gyro, accel = self.imu.read(st.vel, R, st.omega, first=first)
        vio = self.vio.read(st.pos, update=not first)
        return {
            "detected": int(detected),
            "box": box,
            "depth_m": depth,
            "gyro_rad_s": gyro,
            "accel_m_s2": accel,
            "vio_xyz_m": vio,
        }

    def distance_to_target(self) -> float:
        t = self.target
        return point_box_distance(self.quad.state.pos, t.center, TARGET_HALF, t.yaw)

    def _info(self) -> dict:
        st = self.quad.state
        R = quat_to_matrix(st.quat)
        return {
            "is_success": self.outcome == "success",
            "collision": bool(self.collision),
            "outcome": self.outcome,
            "distance_m": self.distance_to_target(),
            "pos": st.pos.copy(),
            "vel": st.vel.copy(),
            "rpy": np.array(euler_from_matrix(R)),
            "t": self.t,
        }
