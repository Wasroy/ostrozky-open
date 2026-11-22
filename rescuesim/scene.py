"""Scene description: primitives, target poses and a small procedural builder.

A layout is a flat list of :class:`Prim` (box, sphere, cylinder, capsule, or a
flat visual-only ``quad`` decal) plus candidate hiding spots. Nothing here
touches PyBullet.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .geometry import IDENTITY_QUAT, quat_from_axis_angle, quat_mul, quat_to_matrix, world_aabb, yaw_quat

TARGET_HALF = (2.0, 0.825, 0.775)           # 4 x 1.65 x 1.55 m
TARGET_RGBA = (1.0, 0.45, 0.0, 1.0)
PROP_EXTENT_M = 150.0                       # props stay within |x|, |y| <= 150 m
ARENA_HALF_M = 350.0
GROUND_HALF_M = 250.0


@dataclass(frozen=True)
class Prim:
    shape: str                  # "box" | "sphere" | "cylinder" | "capsule" | "quad"
    size: tuple                 # box: half extents; sphere: (r,); cylinder/capsule: (r, length); quad: (hx, hy)
    pos: tuple
    quat: tuple = IDENTITY_QUAT
    color: tuple = (0.5, 0.5, 0.5, 1.0)
    tag: str = "prop"
    caps: bool = True           # visual caps on cylinders

    @property
    def collides(self) -> bool:
        return self.shape != "quad"

    @property
    def is_bush(self) -> bool:
        return self.tag == "bush"

    def aabb(self):
        return world_aabb(self.shape, self.size, self.pos, self.quat)


@dataclass(frozen=True)
class TargetPose:
    x: float
    y: float
    yaw: float
    kind: str = ""              # what hides it (wall, canopy, cars, ...)
    approach: str = ""          # a free ~3 m corridor found by the validator

    @property
    def center(self) -> tuple:
        return (self.x, self.y, TARGET_HALF[2])


@dataclass
class Candidate:
    x: float
    y: float
    yaw: float
    kind: str


def rgba(r, g, b, a=1.0):
    return (float(r), float(g), float(b), float(a))


class Scene:
    """Accumulates primitives and hiding-spot candidates, with a coarse 2D occupancy grid."""

    RES = 0.5
    EXT = 175.0

    def __init__(self, rng: np.random.Generator, ground_rgba=(0.4, 0.5, 0.3, 1.0)):
        self.rng = rng
        self.prims: list[Prim] = []
        self.candidates: list[Candidate] = []
        self.ground_rgba = ground_rgba
        n = int(2 * self.EXT / self.RES)
        self.occ = np.zeros((n, n), bool)

    # ------------------------------------------------------------ occupancy
    def _window(self, x0, x1, y0, y1):
        r = self.RES
        i0 = max(0, int((x0 + self.EXT) / r))
        i1 = min(self.occ.shape[0], int((x1 + self.EXT) / r) + 1)
        j0 = max(0, int((y0 + self.EXT) / r))
        j1 = min(self.occ.shape[1], int((y1 + self.EXT) / r) + 1)
        return i0, i1, j0, j1

    def _rect_cells(self, cx, cy, hx, hy, yaw, margin):
        ext = math.hypot(hx, hy) + margin
        i0, i1, j0, j1 = self._window(cx - ext, cx + ext, cy - ext, cy + ext)
        if i0 >= i1 or j0 >= j1:
            return None
        xs = (np.arange(i0, i1) + 0.5) * self.RES - self.EXT
        ys = (np.arange(j0, j1) + 0.5) * self.RES - self.EXT
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        c, s = math.cos(yaw), math.sin(yaw)
        lx = c * (X - cx) + s * (Y - cy)
        ly = -s * (X - cx) + c * (Y - cy)
        inside = (np.abs(lx) <= hx + margin) & (np.abs(ly) <= hy + margin)
        return (slice(i0, i1), slice(j0, j1)), inside

    def mark_rect(self, cx, cy, hx, hy, yaw=0.0, margin=0.0):
        res = self._rect_cells(cx, cy, hx, hy, yaw, margin)
        if res is not None:
            sl, inside = res
            self.occ[sl] |= inside

    def free_rect(self, cx, cy, hx, hy, yaw=0.0, margin=0.0) -> bool:
        if max(abs(cx), abs(cy)) + max(hx, hy) > PROP_EXTENT_M:
            return False
        res = self._rect_cells(cx, cy, hx, hy, yaw, margin)
        if res is None:
            return False
        sl, inside = res
        return not np.any(self.occ[sl] & inside)

    def mark_circle(self, x, y, r, margin=0.0):
        self.mark_rect(x, y, r, r, 0.0, margin)

    def free_circle(self, x, y, r, margin=0.0) -> bool:
        return self.free_rect(x, y, r, r, 0.0, margin)

    # ------------------------------------------------------------ primitives
    def add(self, prim: Prim) -> Prim:
        self.prims.append(prim)
        return prim

    def box(self, c, half, yaw=0.0, color=(0.5, 0.5, 0.5, 1), tag="prop", quat=None, mark=True):
        q = quat if quat is not None else yaw_quat(yaw)
        self.add(Prim("box", tuple(map(float, half)), tuple(map(float, c)), tuple(q), color, tag))
        if mark and quat is None:
            self.mark_rect(c[0], c[1], half[0], half[1], yaw)
        elif mark:
            lo, hi = world_aabb("box", half, c, q)
            self.mark_rect((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (hi[0] - lo[0]) / 2, (hi[1] - lo[1]) / 2)

    def ground_box(self, x, y, hx, hy, height, yaw=0.0, color=(0.5, 0.5, 0.5, 1), tag="prop", z0=0.0, mark=True):
        self.box((x, y, z0 + height / 2), (hx, hy, height / 2), yaw, color, tag, mark=mark)

    def cylinder(self, x, y, z0, r, h, color, tag="prop", caps=True, mark=True):
        self.add(Prim("cylinder", (float(r), float(h)), (float(x), float(y), float(z0 + h / 2)), IDENTITY_QUAT, color, tag, caps))
        if mark:
            self.mark_circle(x, y, r)

    def sphere(self, x, y, z, r, color, tag="prop", mark=True):
        self.add(Prim("sphere", (float(r),), (float(x), float(y), float(z)), IDENTITY_QUAT, color, tag))
        if mark:
            self.mark_circle(x, y, r * 0.8)

    def lying(self, shape, x, y, r, length, yaw, color, tag="prop", z=None, mark=True):
        """Cylinder or capsule with a horizontal axis along ``yaw``, resting on the ground."""
        q = quat_mul(yaw_quat(yaw), quat_from_axis_angle((0, 1, 0), math.pi / 2))
        zc = r if z is None else z
        self.add(Prim(shape, (float(r), float(length)), (float(x), float(y), float(zc)), q, color, tag))
        if mark:
            full = length / 2 + (r if shape == "capsule" else 0.0)
            self.mark_rect(x, y, full, r, yaw)

    def decal(self, x, y, hx, hy, yaw, color, z=0.02):
        self.add(Prim("quad", (float(hx), float(hy)), (float(x), float(y), float(z)), yaw_quat(yaw), color, "decal"))

    def group(self, parts, x, y, yaw=0.0, tilt=None, tag="prop", sink=0.0, mark=True):
        """Place a rigid group of local parts [(shape, size, local_pos, color)] at (x, y, yaw),
        optionally tilted by an extra quaternion, then drop it so it rests on the ground."""
        q0 = yaw_quat(yaw)
        if tilt is not None:
            q0 = quat_mul(q0, tilt)
        R = quat_to_matrix(q0)
        placed = []
        zmin = math.inf
        for shape, size, lp, color in parts:
            c = R @ np.asarray(lp, float)
            lo, _ = world_aabb(shape, size, c, q0)
            zmin = min(zmin, lo[2])
            placed.append((shape, size, c, color))
        dz = -zmin - sink
        for shape, size, c, color in placed:
            pos = (x + c[0], y + c[1], c[2] + dz)
            self.add(Prim(shape, tuple(map(float, size)), tuple(map(float, pos)), q0, color, tag))
            if mark:
                lo, hi = world_aabb(shape, size, pos, q0)
                self.mark_rect((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (hi[0] - lo[0]) / 2, (hi[1] - lo[1]) / 2)

    # ------------------------------------------------------------ composite props
    def tree(self, x, y, hc, rc, trunk_r=0.25, canopy=(0.20, 0.45, 0.16, 1), trunk=(0.36, 0.26, 0.17, 1)):
        """Deciduous tree: trunk up to the canopy centre ``hc``, spherical canopy radius ``rc``."""
        self.cylinder(x, y, 0.0, trunk_r, hc, trunk, "trunk", caps=False)
        self.sphere(x, y, hc, rc, canopy, "canopy", mark=False)

    def conifer(self, x, y, h, rb, canopy=(0.12, 0.32, 0.15, 1), trunk=(0.33, 0.24, 0.16, 1)):
        base = 1.7
        self.cylinder(x, y, 0.0, 0.22, base + 0.3, trunk, "trunk", caps=False)
        h1 = (h - base) * 0.55
        self.cylinder(x, y, base, rb, h1, canopy, "canopy", mark=False)
        self.cylinder(x, y, base + h1, rb * 0.55, h - base - h1, canopy, "canopy", mark=False)
        self.mark_circle(x, y, 0.3)

    def bush(self, x, y, r, color=(0.22, 0.42, 0.18, 1)):
        self.sphere(x, y, 0.55 * r, r, color, "bush")

    def rock(self, x, y, r, color=(0.50, 0.49, 0.46, 1)):
        if self.rng.random() < 0.5:
            self.sphere(x, y, 0.25 * r, r, color, "rock")
        else:
            tilt = quat_from_axis_angle(self.rng.normal(size=3) + [0, 0, 3], self.rng.uniform(0.2, 0.6))
            hx, hy, hz = r * self.rng.uniform(0.7, 1.1), r * self.rng.uniform(0.5, 0.9), r * self.rng.uniform(0.4, 0.7)
            self.group([("box", (hx, hy, hz), (0, 0, 0), color)], x, y, self.rng.uniform(0, math.pi), tilt, "rock", sink=0.25 * hz)

    def house(self, x, y, yaw, w, d, h, wall, roof):
        """Box house with a 45-degree gable roof (ridge along local x)."""
        self.ground_box(x, y, w / 2, d / 2, h, yaw, wall, "house")
        s = d / math.sqrt(2) + 0.25
        q = quat_mul(yaw_quat(yaw), quat_from_axis_angle((1, 0, 0), math.pi / 4))
        self.add(Prim("box", (w / 2 + 0.3, s / 2, s / 2), (x, y, h), q, roof, "roof"))

    def shed(self, x, y, yaw, w, d, h, wall=(0.45, 0.33, 0.22, 1), roof=(0.30, 0.30, 0.32, 1)):
        self.ground_box(x, y, w / 2, d / 2, h, yaw, wall, "shed")
        self.box((x, y, h + 0.1), (w / 2 + 0.25, d / 2 + 0.25, 0.1), yaw, roof, "shed", mark=False)

    def car(self, x, y, yaw, color, tilt=None, tag="car"):
        parts = [
            ("box", (2.15, 0.88, 0.36), (0.0, 0.0, 0.66), color),
            ("box", (1.1, 0.8, 0.3), (-0.25, 0.0, 1.32), (0.18, 0.2, 0.24, 1)),
        ]
        if tilt is None:
            for shape, size, lp, col in parts:
                c, s = math.cos(yaw), math.sin(yaw)
                self.box((x + c * lp[0] - s * lp[1], y + s * lp[0] + c * lp[1], lp[2]), size, yaw, col, tag)
        else:
            self.group(parts, x, y, yaw, tilt, tag, sink=0.05)

    def segment(self, a, b, height, thick, color, tag, z0=0.0, mark=True):
        ax, ay = a
        bx, by = b
        L = math.hypot(bx - ax, by - ay)
        if L < 0.05:
            return
        yaw = math.atan2(by - ay, bx - ax)
        self.ground_box((ax + bx) / 2, (ay + by) / 2, L / 2, thick / 2, height, yaw, color, tag, z0=z0, mark=mark)

    def fence(self, a, b, h=1.3, color=(0.50, 0.38, 0.24, 1)):
        self.segment(a, b, h, 0.10, color, "fence")

    def hedge(self, a, b, h=2.2, w=1.3, color=(0.17, 0.36, 0.14, 1)):
        self.segment(a, b, h, w, color, "hedge")

    # ------------------------------------------------------------ candidates
    def spot(self, x, y, yaw, kind):
        self.candidates.append(Candidate(float(x), float(y), float(yaw), kind))

    def spots_along(self, a, b, offset, kind, n=2, both=True):
        """Candidates parallel to segment a-b, ``offset`` from its centre line (box centre)."""
        ax, ay = a
        bx, by = b
        L = math.hypot(bx - ax, by - ay)
        if L < 4.5:
            return
        yaw = math.atan2(by - ay, bx - ax)
        nx, ny = -math.sin(yaw), math.cos(yaw)
        for _ in range(n):
            t = self.rng.uniform(2.2 / L, 1 - 2.2 / L) if L > 4.4 else 0.5
            px, py = ax + (bx - ax) * t, ay + (by - ay) * t
            sides = (1, -1) if both else (1,)
            for sgn in sides:
                flip = math.pi if self.rng.random() < 0.5 else 0.0
                self.spot(px + sgn * nx * offset, py + sgn * ny * offset, yaw + flip + self.rng.normal(0, 0.03), kind)

    def spots_around(self, x, y, radius, kind, n=3):
        for _ in range(n):
            a = self.rng.uniform(0, 2 * math.pi)
            tangent = a + math.pi / 2
            self.spot(x + radius * math.cos(a), y + radius * math.sin(a), tangent + self.rng.normal(0, 0.15), kind)

    def spots_box_sides(self, x, y, yaw, hx, hy, gap, kind):
        """Candidates against the four walls of a yaw-rotated rectangle."""
        c, s = math.cos(yaw), math.sin(yaw)
        tw = TARGET_HALF[1]
        for (lx, ly, ang, half_len) in (
            (0.0, hy + gap + tw, 0.0, hx), (0.0, -(hy + gap + tw), 0.0, hx),
            (hx + gap + tw, 0.0, math.pi / 2, hy), (-(hx + gap + tw), 0.0, math.pi / 2, hy),
        ):
            if half_len < 2.3:
                continue
            slide = self.rng.uniform(-(half_len - 2.2), half_len - 2.2)
            if ang == 0.0:
                lx = slide
            else:
                ly = slide
            flip = math.pi if self.rng.random() < 0.5 else 0.0
            self.spot(x + c * lx - s * ly, y + s * lx + c * ly, yaw + ang + flip, kind)
