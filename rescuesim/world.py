"""PyBullet worlds: one DIRECT client for collision queries, one for rendering.

Rendering uses TinyRenderer. Its cost is roughly proportional to the number of
triangles of every *visible-flagged* body, whether or not it is on screen, so
static props are merged into one low-poly mesh per (tile, colour) and tiles are
switched on and off per render (alpha 0 bodies are skipped entirely):

* depth render: tiles intersecting the depth frustum (40 m);
* detector render: rendered only over the pixel window the target can cover,
  with only the tiles that intersect that narrow frustum up to the target
  (nothing else can occlude it). Both shortcuts give the same pixels as a full
  render; ``tests/test_sensors.py`` checks that.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pybullet as p

from . import sensors as S
from .geometry import (box_corners, box_mesh, point_box_distance, prism_mesh, quad_mesh,
                       quat_to_matrix, sphere_mesh, world_aabb, yaw_quat)
from .scene import GROUND_HALF_M, TARGET_HALF, TARGET_RGBA, Prim, TargetPose

TILE_M = 20.0
CONTACT_TOL_M = 1e-3
DRONE_REACH_M = 0.2          # half-diagonal of the drone collision box, rounded up


def _tile_key(pos) -> tuple:
    return (int(math.floor(pos[0] / TILE_M)), int(math.floor(pos[1] / TILE_M)))


def _dummy_visual(cid: int) -> int:
    return p.createVisualShape(p.GEOM_MESH, vertices=[[0, 0, -100], [0.001, 0, -100], [0, 0.001, -100]],
                               indices=[0, 1, 2], rgbaColor=[0, 0, 0, 0], physicsClientId=cid)


def prim_mesh(prim: Prim) -> tuple:
    if prim.shape == "box":
        v, f = box_mesh(prim.size)
    elif prim.shape == "sphere":
        v, f = sphere_mesh(prim.size[0])
    elif prim.shape == "cylinder":
        v, f = prism_mesh(prim.size[0], prim.size[1], 8, prim.caps)
    elif prim.shape == "capsule":
        v, f = prism_mesh(prim.size[0], prim.size[1] + 2 * prim.size[0], 8, True)
    elif prim.shape == "quad":
        v, f = quad_mesh(*prim.size)
    else:
        raise ValueError(prim.shape)
    R = quat_to_matrix(prim.quat)
    return v @ R.T + np.asarray(prim.pos), f


# =============================================================================
COMPOUND_MAX = 16


class CollisionWorld:
    """Static props as compound collision bodies (one per tile; bushes separate)."""

    def __init__(self, prims: list[Prim], drone_half=(0.14, 0.14, 0.04)):
        self.cid = p.connect(p.DIRECT)
        cid = self.cid
        self._dummy = _dummy_visual(cid)
        groups = defaultdict(list)
        for pr in prims:
            if pr.collides:
                groups[(_tile_key(pr.pos), pr.is_bush)].append(pr)
        self.prop_bodies: list[int] = []
        self.bush_bodies: set[int] = set()
        for (key, bush), lst in sorted(groups.items()):
            # PyBullet keeps at most 16 child shapes per compound: larger chunks silently drop collisions
            for k in range(0, len(lst), COMPOUND_MAX):
                uid = self._compound(lst[k:k + COMPOUND_MAX])
                self.prop_bodies.append(uid)
                if bush:
                    self.bush_bodies.add(uid)
        self.prop_set = set(self.prop_bodies)
        tcs = p.createCollisionShape(p.GEOM_BOX, halfExtents=TARGET_HALF, physicsClientId=cid)
        self.target = p.createMultiBody(0, tcs, self._dummy, basePosition=[0, 0, -50], physicsClientId=cid)
        dcs = p.createCollisionShape(p.GEOM_BOX, halfExtents=drone_half, physicsClientId=cid)
        self.drone = p.createMultiBody(0, dcs, self._dummy, basePosition=[0, 0, -60], physicsClientId=cid)
        self.target_pose: TargetPose | None = None

    def _compound(self, lst: list[Prim]) -> int:
        types, radii, halves, lengths, pos, orn = [], [], [], [], [], []
        for pr in lst:
            if pr.shape == "box":
                types.append(p.GEOM_BOX); halves.append(list(pr.size)); radii.append(0.0); lengths.append(0.0)
            elif pr.shape == "sphere":
                types.append(p.GEOM_SPHERE); halves.append([0, 0, 0]); radii.append(pr.size[0]); lengths.append(0.0)
            elif pr.shape == "cylinder":
                types.append(p.GEOM_CYLINDER); halves.append([0, 0, 0]); radii.append(pr.size[0]); lengths.append(pr.size[1])
            elif pr.shape == "capsule":
                types.append(p.GEOM_CAPSULE); halves.append([0, 0, 0]); radii.append(pr.size[0]); lengths.append(pr.size[1])
            else:
                raise ValueError(pr.shape)
            pos.append(list(pr.pos))
            orn.append(list(pr.quat))
        cs = p.createCollisionShapeArray(shapeTypes=types, radii=radii, halfExtents=halves, lengths=lengths,
                                         collisionFramePositions=pos, collisionFrameOrientations=orn,
                                         physicsClientId=self.cid)
        return p.createMultiBody(0, cs, self._dummy, physicsClientId=self.cid)

    def set_target(self, pose: TargetPose | None):
        self.target_pose = pose
        if pose is None:
            p.resetBasePositionAndOrientation(self.target, [0, 0, -50], [0, 0, 0, 1], physicsClientId=self.cid)
        else:
            p.resetBasePositionAndOrientation(self.target, pose.center, yaw_quat(pose.yaw), physicsClientId=self.cid)

    def bodies_near(self, lo, hi) -> list[int]:
        ov = p.getOverlappingObjects(list(lo), list(hi), physicsClientId=self.cid) or []
        out = []
        for uid, _link in ov:
            if (uid in self.prop_set or uid == self.target) and uid not in out:
                out.append(uid)
        return out

    def place_drone(self, pos, quat):
        p.resetBasePositionAndOrientation(self.drone, list(pos), list(quat), physicsClientId=self.cid)

    def touching(self, body: int, candidates, tol: float = CONTACT_TOL_M) -> set:
        hits = set()
        for uid in candidates:
            pts = p.getClosestPoints(body, uid, tol, physicsClientId=self.cid)
            if pts and min(pt[8] for pt in pts) <= tol:
                hits.add(uid)
        return hits

    def drone_contacts(self, pos, quat, candidates=None) -> set:
        self.place_drone(pos, quat)
        if candidates is None:
            r = DRONE_REACH_M
            candidates = self.bodies_near(np.asarray(pos) - r, np.asarray(pos) + r)
        return self.touching(self.drone, candidates)

    def clearance_ok(self, pos, dist: float) -> bool:
        self.place_drone(pos, (0, 0, 0, 1))
        r = DRONE_REACH_M + dist
        for uid in self.bodies_near(np.asarray(pos) - r, np.asarray(pos) + r):
            if p.getClosestPoints(self.drone, uid, dist, physicsClientId=self.cid):
                return False
        return True

    def close(self):
        try:
            p.disconnect(self.cid)
        except Exception:
            pass


# =============================================================================
class RenderWorld:
    """Visual meshes merged per (tile, colour), with per-render tile culling."""

    def __init__(self, prims: list[Prim], ground_rgba):
        self.cid = p.connect(p.DIRECT)
        cid = self.cid
        g = GROUND_HALF_M
        gv = p.createVisualShape(p.GEOM_MESH, vertices=[[-g, -g, 0], [g, -g, 0], [g, g, 0], [-g, g, 0]],
                                 indices=[0, 1, 2, 0, 2, 3], rgbaColor=list(ground_rgba), physicsClientId=cid)
        self.ground = p.createMultiBody(0, -1, gv, physicsClientId=cid)
        groups = defaultdict(lambda: ([], []))
        for pr in prims:
            v, f = prim_mesh(pr)
            V, F = groups[(_tile_key(pr.pos), pr.color)]
            F.append(f + sum(len(x) for x in V))
            V.append(v)
        self.bodies, self.rgba, centers, halves = [], [], [], []
        self.n_triangles = 2
        for (key, color), (V, F) in sorted(groups.items()):
            V = np.concatenate(V)
            F = np.concatenate(F)
            vs = p.createVisualShape(p.GEOM_MESH, vertices=V.tolist(), indices=F.ravel().tolist(),
                                     rgbaColor=list(color), physicsClientId=cid)
            self.bodies.append(p.createMultiBody(0, -1, vs, physicsClientId=cid))
            self.rgba.append(list(color))
            lo, hi = V.min(0), V.max(0)
            centers.append((lo + hi) / 2)
            halves.append((hi - lo) / 2 + 0.05)
            self.n_triangles += len(F)
        self.centers = np.array(centers).reshape(-1, 3)
        self.halves = np.array(halves).reshape(-1, 3)
        self.visible = np.ones(len(self.bodies), bool)
        tv = p.createVisualShape(p.GEOM_BOX, halfExtents=TARGET_HALF, rgbaColor=list(TARGET_RGBA), physicsClientId=cid)
        self.target = p.createMultiBody(0, -1, tv, basePosition=[0, 0, -50], physicsClientId=cid)
        self.target_pose: TargetPose | None = None
        self.culling = True
        self.full_window = False        # debugging: render the detector image in full
        self.renders = 0

    def set_target(self, pose: TargetPose | None):
        self.target_pose = pose
        if pose is None:
            p.resetBasePositionAndOrientation(self.target, [0, 0, -50], [0, 0, 0, 1], physicsClientId=self.cid)
        else:
            p.resetBasePositionAndOrientation(self.target, pose.center, yaw_quat(pose.yaw), physicsClientId=self.cid)

    def _show(self, want: np.ndarray):
        if not self.culling:
            want = np.ones_like(self.visible)
        for k in np.flatnonzero(want != self.visible):
            col = self.rgba[k] if want[k] else self.rgba[k][:3] + [0.0]
            p.changeVisualShape(self.bodies[k], -1, rgbaColor=col, physicsClientId=self.cid)
        self.visible = want.copy()

    def _image(self, w, h, view, proj, flags):
        self.renders += 1
        return p.getCameraImage(w, h, view, proj, renderer=p.ER_TINY_RENDERER, shadow=0, flags=flags,
                                physicsClientId=self.cid)

    def render_depth(self, cam) -> np.ndarray:
        """Noise-free z-depth (m), shape (126, 224). Values >= 40 m mean nothing closer."""
        eye, fwd, up, right = cam
        W, H = S.DEPTH_W, S.DEPTH_H
        tang = S.window_tangents(0, W, 0, H, W, H)
        planes = S.frustum_planes(eye, fwd, up, right, tang, S.NEAR, S.DEPTH_FAR)
        self._show(S.aabb_in_frustum(planes, self.centers, self.halves))
        view = p.computeViewMatrix(list(eye), list(eye + fwd), list(up))
        proj = S.projection_for_window(0, W, 0, H, W, H, S.NEAR, S.DEPTH_FAR)
        img = self._image(W, H, view, proj, p.ER_NO_SEGMENTATION_MASK)
        buf = np.asarray(img[3], np.float32).reshape(H, W)
        return S.depth_from_buffer(buf.astype(np.float64), S.NEAR, S.DEPTH_FAR)

    def render_target_mask(self, cam):
        """Ground-truth visible-target mask in the 160 x 90 detector image.

        Returns (mask, (col0, row0), distance_to_box_m); mask is None when the
        target cannot be visible (behind, outside the image or beyond range)."""
        eye, fwd, up, right = cam
        pose = self.target_pose
        if pose is None:
            return None, (0, 0), math.inf
        dist = point_box_distance(eye, pose.center, TARGET_HALF, pose.yaw)
        if dist > S.DETECT_RANGE_M:
            return None, (0, 0), dist
        W, H = S.DET_W, S.DET_H
        corners = box_corners(pose.center, TARGET_HALF, pose.yaw)
        u, v, zc = project_points_safe(corners, eye, fwd, up, right, W, H)
        if np.all(zc <= S.NEAR):
            return None, (0, 0), dist
        if self.full_window or np.any(zc <= 2 * S.NEAR):
            i0, i1, j0, j1 = 0, W, 0, H
        else:
            i0 = max(0, int(math.floor(u.min())) - 1)
            i1 = min(W, int(math.ceil(u.max())) + 1)
            j0 = max(0, int(math.floor(v.min())) - 1)
            j1 = min(H, int(math.ceil(v.max())) + 1)
            if i0 >= i1 or j0 >= j1:
                return None, (0, 0), dist
        far = min(S.DET_FAR, float(zc.max()) + 1.0) if not self.full_window else S.DET_FAR
        tang = S.window_tangents(i0, i1, j0, j1, W, H)
        planes = S.frustum_planes(eye, fwd, up, right, tang, S.NEAR, far)
        self._show(S.aabb_in_frustum(planes, self.centers, self.halves))
        view = p.computeViewMatrix(list(eye), list(eye + fwd), list(up))
        proj = S.projection_for_window(i0, i1, j0, j1, W, H, S.NEAR, far)
        img = self._image(i1 - i0, j1 - j0, view, proj, 0)
        seg = np.asarray(img[4], np.int64).reshape(j1 - j0, i1 - i0)
        return seg == self.target, (i0, j0), dist

    def render_rgb(self, eye, target, up, w, h, fov_deg=60.0, far=400.0, ortho_half=None):
        """Plain colour render for previews (all tiles shown)."""
        self._show(np.ones_like(self.visible))
        view = p.computeViewMatrix(list(eye), list(target), list(up))
        if ortho_half is not None:
            a = w / h
            proj = _ortho(-ortho_half * a, ortho_half * a, -ortho_half, ortho_half, 1.0, far)
        else:
            proj = p.computeProjectionMatrixFOV(fov_deg, w / h, 0.1, far)
        img = self._image(w, h, view, proj, p.ER_NO_SEGMENTATION_MASK)
        return np.asarray(img[2], np.uint8).reshape(h, w, 4)[:, :, :3]

    def close(self):
        try:
            p.disconnect(self.cid)
        except Exception:
            pass


def _ortho(l, r, b, t, n, f):
    return [2 / (r - l), 0, 0, 0, 0, 2 / (t - b), 0, 0, 0, 0, -2 / (f - n), 0,
            -(r + l) / (r - l), -(t + b) / (t - b), -(f + n) / (f - n), 1]


def project_points_safe(points, eye, fwd, up, right, W, H):
    u, v, zc = S.project_points(points, eye, fwd, up, right, W, H)
    return np.nan_to_num(u), np.nan_to_num(v), zc


# =============================================================================
class PoseValidator:
    """Geometric checks for candidate target poses on a collision world."""

    SIDE_LEN = 25.0
    SIDE_Z = (0.3, 3.0)
    ABOVE_Z = (1.62, 60.0)
    OFFSETS = (0.0, 0.75, -0.75, 1.5, -1.5)

    def __init__(self, world: CollisionWorld):
        self.w = world
        cid = world.cid
        mk = lambda half: p.createMultiBody(0, p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=cid),
                                            world._dummy, basePosition=[0, 0, -500], physicsClientId=cid)
        self.box_half = TARGET_HALF
        self.box = mk(TARGET_HALF)
        hz = (self.SIDE_Z[1] - self.SIDE_Z[0]) / 2
        self.side_half = (self.SIDE_LEN / 2, 1.5, hz)
        self.side = mk(self.side_half)
        az = (self.ABOVE_Z[1] - self.ABOVE_Z[0]) / 2
        self.above_half = (1.5, 1.5, az)
        self.above = mk(self.above_half)
        self.probe_r = 0.45
        self.probe = p.createMultiBody(0, p.createCollisionShape(p.GEOM_SPHERE, radius=self.probe_r, physicsClientId=cid),
                                       world._dummy, basePosition=[0, 0, -500], physicsClientId=cid)

    def _hits(self, body, half, pos, quat, clearance=0.0, skip_bush=False) -> bool:
        lo, hi = world_aabb("box", half, pos, quat)
        p.resetBasePositionAndOrientation(body, list(pos), list(quat), physicsClientId=self.w.cid)
        for uid in self.w.bodies_near(lo - clearance - 0.05, hi + clearance + 0.05):
            if uid == self.w.target or (skip_bush and uid in self.w.bush_bodies):
                continue
            pts = p.getClosestPoints(body, uid, clearance, physicsClientId=self.w.cid)
            if pts and min(pt[8] for pt in pts) < clearance:
                return True
        return False

    def box_overlaps(self, pose: TargetPose, clearance=0.08) -> bool:
        """True if any non-bush prop is closer than ``clearance`` to the target box."""
        return self._hits(self.box, self.box_half, pose.center, yaw_quat(pose.yaw), clearance, skip_bush=True)

    def corridor(self, pose: TargetPose) -> str | None:
        """Name of a free ~3 m approach corridor, or None."""
        c, s = math.cos(pose.yaw), math.sin(pose.yaw)
        ux, uy = c, s
        vx, vy = -s, c
        zc = (self.ABOVE_Z[0] + self.ABOVE_Z[1]) / 2
        for o in self.OFFSETS:
            for t in (0.0, 1.0, -1.0):
                pos = (pose.x + ux * t + vx * o, pose.y + uy * t + vy * o, zc)
                if not self._hits(self.above, self.above_half, pos, yaw_quat(pose.yaw)):
                    return "above"
        zs = (self.SIDE_Z[0] + self.SIDE_Z[1]) / 2
        for k in range(8):
            a = pose.yaw + k * math.pi / 4
            dx, dy = math.cos(a), math.sin(a)
            nx, ny = -dy, dx
            for o in self.OFFSETS:
                pos = (pose.x + dx * self.SIDE_LEN / 2 + nx * o, pose.y + dy * self.SIDE_LEN / 2 + ny * o, zs)
                if not self._hits(self.side, self.side_half, pos, yaw_quat(a)):
                    return f"side{int(round(math.degrees(k * math.pi / 4)))}"
        return None

    def free_point(self, pos) -> bool:
        p.resetBasePositionAndOrientation(self.probe, list(pos), [0, 0, 0, 1], physicsClientId=self.w.cid)
        r = self.probe_r + 0.05
        for uid in self.w.bodies_near(np.asarray(pos) - r, np.asarray(pos) + r):
            if uid != self.w.target and p.getClosestPoints(self.probe, uid, 0.0, physicsClientId=self.w.cid):
                return False
        return abs(pos[0]) < 340 and abs(pos[1]) < 340

    @staticmethod
    def sample_points(pose: TargetPose) -> np.ndarray:
        hx, hy, hz = TARGET_HALF
        e = 0.06
        local = [(x, y, hz + e) for x in (-1.5, 0.0, 1.5) for y in (-0.5, 0.5)]
        local += [(hx + e, 0, 0), (-hx - e, 0, 0), (0, hy + e, 0), (0, -hy - e, 0),
                  (1.4, hy + e, 0.3), (-1.4, hy + e, 0.3), (1.4, -hy - e, 0.3), (-1.4, -hy - e, 0.3)]
        R = quat_to_matrix(yaw_quat(pose.yaw))
        return np.asarray(pose.center) + np.asarray(local) @ R.T

    def visible_rays(self, pose: TargetPose, viewpoints: np.ndarray) -> np.ndarray:
        """Number of unobstructed rays from each viewpoint to the box sample points."""
        pts = self.sample_points(pose)
        froms = np.repeat(viewpoints, len(pts), axis=0)
        tos = np.tile(pts, (len(viewpoints), 1))
        res = p.rayTestBatch(froms.tolist(), tos.tolist(), physicsClientId=self.w.cid)
        clear = np.array([r[0] < 0 or r[0] == self.w.target for r in res])
        return clear.reshape(len(viewpoints), len(pts)).sum(1)

    @staticmethod
    def viewpoints(pose: TargetPose, dist: float, alt: float, n: int = 8) -> np.ndarray:
        a = pose.yaw + np.arange(n) * 2 * math.pi / n
        return np.c_[pose.x + dist * np.cos(a), pose.y + dist * np.sin(a), np.full(n, alt)]
