"""Predictive obstacle avoidance with an obstacle memory (Wasroy attempt).

Why: a filter that only checks the straight line along the commanded direction misses momentum.
After a turn the drone keeps sliding along its old velocity, the camera has yawed to the new
direction, and the wall it slides into is outside the 102 deg view. Measured on the GPU avoidance
benchmark (tools/avoid_bench.py), that is how most such crashes happen.

What this does, each tick:
1. Depth pixels -> world points, merged into a voxel map of the last `memory_s` seconds, so a wall
   stays known after it leaves the image.
2. Candidate commands = directions (yaw offsets x climbs) x speed fractions (including stop).
3. Each candidate is scored on the path the drone will actually fly: the velocity relaxes from the
   current one toward the command with time constant `tau` (the controller + attitude lag), so the
   sideways momentum is part of the prediction.
4. A candidate is safe if that predicted path stays clear of every remembered point over `horizon` s,
   by a radius that grows with its speed (0.9 m creeping, 1.3 m from 2.3 m/s). Among safe ones, the
   one closest to the desired velocity wins. If none is safe: the one that avoids contact (0.5 m)
   longest, then keeps the most distance, then is slowest.
5. Near the box we mean to touch, only the box's own remembered points are dropped; props next to it
   stay obstacles.

Interface: as handin.avoidance.Avoidance, filter(v_des, depth, state, exclude_box=None) -> v, plus an
optional `target` (the box to touch) whose remembered points are not obstacles.
"""
import numpy as np
from scipy.spatial import cKDTree

from .core.camera import depth_points, camera_pose, DEPTH_VALID, TAN_H, TAN_V


class PredictiveAvoidance:
    def __init__(self, radius=1.3, radius_min=0.9, radius_per_mps=0.3, hard_radius=0.5, ground_z=0.35, min_z=0.5, tau=0.8, horizon=2.0, dt_pred=0.1,
                 memory_s=4.0, voxel=0.5, step=3, dt=1 / 30, mem_every=3,
                 yaw_offsets_deg=(0, 15, -15, 30, -30, 45, -45, 60, -60, 90, -90, 135, -135, 180),
                 climbs_deg=(0, 30, 60), speed_fracs=(1.0, 0.6, 0.3, 0.0), level=True):
        #### level (from @sckoott's rl-patches 88f918f): also offer level flight when the command descends. Under a
        #### canopy every descent is a predicted ground hit and every climb hits the canopy, so the drone froze.
        self.level = level
        self.radius, self.ground_z, self.min_z = radius, ground_z, min_z
        self.radius_min, self.radius_per_mps, self.hard_radius = radius_min, radius_per_mps, hard_radius
        self.tau, self.horizon, self.dt_pred = tau, horizon, dt_pred
        self.mem_every = mem_every
        self.memory_ticks = int(round(memory_s / dt))
        self.voxel, self.step = voxel, step
        self.yaw_offsets = np.radians(yaw_offsets_deg)
        self.climbs = np.radians(climbs_deg)
        self.speed_fracs = np.asarray(speed_fracs, dtype=float)
        t = np.arange(dt_pred, horizon + 1e-9, dt_pred)
        self.t = t
        self.k_cmd = t - tau * (1 - np.exp(-t / tau))          # weight of the command in p(t)
        self.k_now = tau * (1 - np.exp(-t / tau))              # weight of the current velocity
        self.reset()

    def reset(self):
        self.mem_keys = np.zeros(0, np.int64)
        self.mem_pts = np.zeros((0, 3))
        self.mem_tick = np.zeros(0, np.int64)
        self.tick = 0
        self.box_held, self.box_tick = None, -100
        self.last = {}

    def _thin(self, pts):
        """Keep one point per voxel. Returns (packed int64 keys, points)."""
        if not len(pts):
            return np.zeros(0, np.int64), pts
        v = np.floor(pts / self.voxel).astype(np.int64) + (1 << 20)          # non-negative, 21 bits each
        keys = (v[:, 0] << 42) | (v[:, 1] << 21) | v[:, 2]
        keys, keep = np.unique(keys, return_index=True)
        return keys, pts[keep]

    def _memory(self, depth, state, exclude_box, exclude_band=None):
        """World points seen now plus a voxel map of the last `memory_s` seconds.

        The map is updated every `mem_every` ticks: the new frame is merged in, each voxel keeps its
        newest point and time, voxels older than the memory are dropped. Long enough to remember low
        props seen from afar before the drone descends onto them (the camera only sees ~10 deg below
        the horizon when level)."""
        now = self._thin(depth_points(depth, state, step=self.step, exclude_box=exclude_box,
                                      min_z=self.ground_z, max_depth=DEPTH_VALID, exclude_band=exclude_band))
        self.tick += 1
        if self.tick % self.mem_every == 0:
            keys = np.concatenate([now[0], self.mem_keys])               # newest first: unique keeps it
            pts = np.concatenate([now[1], self.mem_pts])
            tk = np.concatenate([np.full(len(now[0]), self.tick), self.mem_tick])
            keys, keep = np.unique(keys, return_index=True)
            pts, tk = pts[keep], tk[keep]
            rel = pts - np.asarray(state["pos"], float)
            fresh = (tk > self.tick - self.memory_ticks) & (np.einsum("ij,ij->i", rel, rel) < 45.0 ** 2)
            self.mem_keys, self.mem_pts, self.mem_tick = keys[fresh], pts[fresh], tk[fresh]
        return np.concatenate([now[1], self.mem_pts]) if len(self.mem_pts) else now[1]

    def _drop_target(self, pts, state, exclude_box, target):
        """Remove remembered points that are the box we mean to touch, and nothing else.

        The current frame's box pixels are already dropped (exclude_box), but earlier frames may hold
        the box (detector misses, frames before detection). With a detection now: drop memory points
        that project inside the detection box (padded) and lie at or beyond the box's depth (minus
        1 m), so obstacles in front of the box stay. Without one: drop points in a small cylinder at
        the target (r 1.5 m, below 1.7 m; the box is 1.55 m tall). Props next to the box stay obstacles:
        a 3.5 m all-height exclusion caused most terminal crashes in the kit (JOURNAL 2026-11-21).
        """
        target = np.asarray(target, float)
        if exclude_box is not None:
            eye, R_cam = camera_pose(state)
            d = (pts - eye) @ R_cam                                  # camera frame: forward, left, up
            fwd = np.maximum(d[:, 0], 1e-3)
            u = 0.5 - d[:, 1] / fwd / (2 * TAN_H)
            v = 0.5 - d[:, 2] / fwd / (2 * TAN_V)
            x1, y1, x2, y2 = exclude_box
            pad = 0.03
            inside = (d[:, 0] > 0) & (u > x1 - pad) & (u < x2 + pad) & (v > y1 - pad) & (v < y2 + pad)
            box_fwd = (target - eye) @ R_cam[:, 0]
            return pts[~(inside & (d[:, 0] > box_fwd - 1.0))]
        near = (np.linalg.norm(pts[:, :2] - target[:2], axis=1) < 1.5) & (pts[:, 2] < 1.7)
        return pts[~near]

    #### Goal handling (from @sckoott's rl-patches 88f918f): `goal` = the box aim point (world, box-centre height).
    GOAL_POINT_R = 1.5      # m: remembered points this close (horizontally) to the aim point, below GOAL_TOP_Z, are the box
    GOAL_TOP_Z = 1.9        # m: box top 1.55 + margin
    GOAL_ARRIVE_R = 1.5     # m: a predicted path this close to the aim point has reached the box (contact = success)
    GOAL_LOW_R = 3.0        # m: within this of the aim point the ground check uses GOAL_MIN_Z instead of min_z
    GOAL_MIN_Z = 0.3

    def filter(self, v_des, depth, state, exclude_box=None, target=None, exclude_band=None, goal=None):
        """target: world point of the box we mean to touch (its remembered points are not obstacles).
        exclude_band: (d_lo, d_hi) z-depths inside exclude_box that are the box; nearer pixels stay obstacles.
        goal: aim point on the box; a predicted path that reaches it first is safe whatever comes after."""
        v_des = np.asarray(v_des, dtype=float)
        #### Detector misses (5% of frames): hold the last box for a few frames so the box's pixels never
        #### enter the memory as an obstacle
        if exclude_box is not None:
            self.box_held, self.box_tick = exclude_box, self.tick
        elif target is not None and self.tick - self.box_tick <= 6:
            exclude_box = self.box_held
        pts = self._memory(depth, state, exclude_box, exclude_band)
        if target is not None and len(pts):
            pts = self._drop_target(pts, state, exclude_box, target)
        speed = np.linalg.norm(v_des)
        p0, v0 = np.asarray(state["pos"], float), np.asarray(state["vel"], float)
        if speed < 0.2:
            self.last = {"blocked": False, "n_points": len(pts)}
            return v_des
        d0 = v_des / speed
        heading, elev0 = np.arctan2(d0[1], d0[0]), np.arcsin(np.clip(d0[2], -1, 1))
        #### Candidate commands (C, 3)
        dirs = []
        elevs = [elev0 + climb if climb > 0 else elev0 for climb in self.climbs]
        if self.level and elev0 < -0.05:
            elevs.insert(1, 0.0)                                  # level flight, between "as asked" and climbs
        for el in elevs:
            for off in self.yaw_offsets:
                dirs.append([np.cos(el) * np.cos(heading + off), np.cos(el) * np.sin(heading + off), np.sin(el)])
        dirs = np.array(dirs)
        cmds = (dirs[None] * (speed * self.speed_fracs)[:, None, None]).reshape(-1, 3)
        #### Predicted paths (C, T, 3); only points the paths can reach matter
        paths = p0 + cmds[:, None] * self.k_cmd[None, :, None] + v0 * self.k_now[None, :, None]
        g = None if goal is None else np.asarray(goal, float)
        if g is not None and len(pts):
            on_box = (np.linalg.norm(pts[:, :2] - g[:2], axis=1) < self.GOAL_POINT_R) & (pts[:, 2] < self.GOAL_TOP_Z)
            pts = pts[~on_box]
        if len(pts):
            reach = np.linalg.norm(paths - p0, axis=2).max() + 4 * self.radius
            pts = pts[np.einsum("ij,ij->i", pts - p0, pts - p0) < reach * reach]
        if len(pts):
            tree = cKDTree(pts)
            dist, _ = tree.query(paths.reshape(-1, 3), k=1, distance_upper_bound=4 * self.radius)
            dist = np.minimum(dist.reshape(len(cmds), -1), 4 * self.radius)
        else:
            dist = np.full(paths.shape[:2], 4 * self.radius)
        low = paths[:, :, 2] < self.min_z                         # treat flying into the ground as a hit
        if g is not None:
            near_goal = np.linalg.norm(paths[:, :, :2] - g[:2], axis=2) < self.GOAL_LOW_R
            low = np.where(near_goal, paths[:, :, 2] < self.GOAL_MIN_Z, low)
        dist = np.where(low, 0.0, dist)
        #### Safety radius grows with the commanded speed: 1.3 m from 2.3 m/s up, 0.9 m when creeping, so a
        #### box next to a wall can still be reached slowly (a fixed 1.3 m left no safe command there)
        r_c = np.clip(0.6 + self.radius_per_mps * np.linalg.norm(cmds, axis=1), self.radius_min, self.radius)
        hit = dist < r_c[:, None]
        if g is not None:
            arrived = np.linalg.norm(paths - g, axis=2) < self.GOAL_ARRIVE_R
            first_arrive = np.where(arrived.any(1), arrived.argmax(1), len(self.t))
            hit &= np.arange(len(self.t))[None, :] < first_arrive[:, None]
        safe = ~hit.any(1)
        #### Cost: closeness to the desired velocity (direction first, then speed)
        dev = np.linalg.norm(cmds - v_des, axis=1)
        if safe.any():
            i = int(np.argmin(np.where(safe, dev, np.inf)))
        else:
            #### Nothing safe: avoid physical contact for longest, then keep the most distance on average,
            #### then go slow. (Scoring by "delay the first margin violation" picked fast sideways dashes
            #### along walls in the kit, city@v32.)
            hard = dist < self.hard_radius
            first_hard = np.where(hard.any(1), hard.argmax(1), len(self.t))
            score = first_hard * 10.0 + dist.mean(1) - 0.3 * np.linalg.norm(cmds, axis=1)
            i = int(np.argmax(score))
        v = cmds[i]
        self.last = {"blocked": bool(dev[i] > 1e-6), "safe": bool(safe[i]), "n_points": len(pts),
                     "n_safe": int(safe.sum())}
        return v
