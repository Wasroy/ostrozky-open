"""Reactive obstacle avoidance from the depth map (sckoott attempt).

Each tick, every depth pixel becomes a 3D point. A point is a hazard for a candidate direction if it
lies ahead along it (within the look-ahead distance) and closer than `radius` to the line of travel.
The desired velocity is kept if its direction is clear; otherwise we try directions rotated left and
right, then climbing, and take the clear one closest to the desired direction. Speed is limited so
we can stop before the nearest hazard. Ground points (world z below `ground_z`) are ignored here
(the mission keeps a minimum altitude), and so are the target's own pixels when we mean to hit it.
"""
import numpy as np

from .core.camera import depth_points


class Avoidance:
    def __init__(self, radius=1.2, ground_z=0.35, brake_acc=4.0, margin=2.0, step=2,
                 yaw_offsets_deg=(0, 15, -15, 30, -30, 45, -45, 60, -60, 80, -80), climbs_deg=(0, 25, 50)):
        self.radius, self.ground_z, self.brake_acc, self.margin, self.step = radius, ground_z, brake_acc, margin, step
        self.yaw_offsets = np.radians(yaw_offsets_deg)
        self.climbs = np.radians(climbs_deg)
        self.last = {}

    def points(self, depth, state, exclude_box=None):
        """World points (N, 3) of the obstacles seen this frame (ground dropped), and the camera eye."""
        from .core.camera import camera_pose
        pts = depth_points(depth, state, step=self.step, exclude_box=exclude_box, min_z=self.ground_z)
        return pts, camera_pose(state)[0]

    def clearance(self, pts, eye, direction, look):
        """Distance along `direction` (unit, world) to the first hazard within `look`, else `look`."""
        rel = pts - eye
        along = rel @ direction
        perp = np.linalg.norm(rel - np.outer(along, direction), axis=1)
        hit = (along > 0) & (along < look) & (perp < self.radius)
        return float(along[hit].min()) if hit.any() else look

    def filter(self, v_des, depth, state, exclude_box=None):
        speed = np.linalg.norm(v_des)
        if speed < 0.2:
            return np.asarray(v_des, dtype=float)
        pts, eye = self.points(depth, state, exclude_box)
        look = max(8.0, speed * speed / (2 * self.brake_acc) + self.margin + 4.0)
        d0 = v_des / speed
        heading = np.arctan2(d0[1], d0[0])
        elev0 = np.arcsin(np.clip(d0[2], -1, 1))
        best = None
        for climb in self.climbs:
            for off in self.yaw_offsets:
                el = elev0 + climb if climb > 0 else elev0
                d = np.array([np.cos(el) * np.cos(heading + off), np.cos(el) * np.sin(heading + off), np.sin(el)])
                c = self.clearance(pts, eye, d, look)
                if c >= look - 1e-6:
                    best = (d, c, off, climb)
                    break
                if best is None or c > best[1]:
                    best = (d, c, off, climb)
            if best is not None and best[1] >= look - 1e-6:
                break
        d, c, off, climb = best
        #### Speed that still lets us stop before the hazard on the chosen direction
        v_max = np.sqrt(max(0.0, 2 * self.brake_acc * (c - self.margin)))
        v = d * min(speed, v_max)
        self.last = {"clear": c, "look": look, "offset_deg": np.degrees(off), "climb_deg": np.degrees(climb),
                     "blocked": c < look - 1e-6, "n_points": len(pts)}
        return v
