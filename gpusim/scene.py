"""rescuesim terrains as GPU scenes (Warp): visual triangle mesh, collision primitives, target boxes.

Written for the rescuesim port of @sckoott's GPU engine (``GpuScenes`` keeps his multi-terrain upload:
one mesh per terrain type, per-face ids, offsets into concatenated tables).

Everything comes from rescuesim itself: the layout (props and the 48 target poses) is built by
``rescuesim.terrain.get_layout`` with the same layout seed, and the visual triangles are
``rescuesim.world.prim_mesh`` (the low-poly hulls rescuesim renders: icosahedra circumscribing
spheres, octagonal prisms circumscribing cylinders and capsules, boxes, flat decals), plus the
same 500 x 500 m ground quad. One scene per terrain *type*, shared by all of its poses; the
target box is not in the mesh, it is an oriented box per drone (intersected analytically).

Collisions use rescuesim's collision primitives exactly (boxes, spheres, cylinders and capsules
with their poses and sizes, decals excluded), not the visual hulls: they are uploaded as
primitive tables plus a Warp BVH over their bounding boxes (see ``gpusim.kernels``). By default
only the prims that actually collide in rescuesim's PyBullet world are solid (see
``rescuesim_collision_mask``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from rescuesim.geometry import quat_to_matrix, yaw_quat
from rescuesim.scene import GROUND_HALF_M, TARGET_HALF
from rescuesim.terrain import DEFAULT_LAYOUT_SEEDS, get_layout
from rescuesim.world import prim_mesh

SHAPE_CODES = {"box": 0, "sphere": 1, "cylinder": 2, "capsule": 3}
GROUND_ID = -2                 # face id of the ground quad (props use their prim index, >= 0)


@dataclass
class MeshData:
    vertices: np.ndarray       # (V, 3) float32
    faces: np.ndarray          # (F, 3) int32
    face_prim: np.ndarray      # (F,) int32: index into layout.prims, or GROUND_ID


@dataclass
class PrimTable:
    """Collision primitives in a GPU-friendly layout (one row per colliding rescuesim Prim)."""
    shape: np.ndarray          # (K,) int32, SHAPE_CODES
    pos: np.ndarray            # (K, 3) float32 centre
    rot: np.ndarray            # (K, 3, 3) float32 world-from-primitive
    size: np.ndarray           # (K, 3) float32: box half extents; sphere (r, 0, 0);
    #                            cylinder (r, half length, 0); capsule (r, half segment length, 0)
    lo: np.ndarray             # (K, 3) float32 world AABB
    hi: np.ndarray
    prim_index: np.ndarray     # (K,) int32 index into layout.prims


@dataclass
class TerrainScene:
    kind: str
    layout_seed: int
    layout: object             # rescuesim.terrain.Layout
    visual: MeshData
    prims: PrimTable
    boxes: np.ndarray          # (P, 15) float32 target OBB per pose v1..vP: centre, half extents, R row-major

    @property
    def n_poses(self) -> int:
        return len(self.boxes)

    def terrain_string(self, pose: int) -> str:
        """rescuesim's terrain string for seeding (RescueEnv.terrain_str)."""
        if self.layout_seed != DEFAULT_LAYOUT_SEEDS[self.kind]:
            return f"{self.kind}#L{self.layout_seed}@v{pose}"
        return f"{self.kind}@v{pose}"


def target_obb(pose) -> np.ndarray:
    R = quat_to_matrix(yaw_quat(pose.yaw))
    return np.concatenate([np.asarray(pose.center, float), np.asarray(TARGET_HALF, float), R.ravel()]).astype(np.float32)


def _visual_mesh(prims) -> MeshData:
    g = GROUND_HALF_M
    V = [np.array([[-g, -g, 0], [g, -g, 0], [g, g, 0], [-g, g, 0]], float)]
    F = [np.array([[0, 1, 2], [0, 2, 3]])]
    ids = [np.full(2, GROUND_ID)]
    n = 4
    for k, pr in enumerate(prims):
        v, f = prim_mesh(pr)
        V.append(v)
        F.append(np.asarray(f) + n)
        ids.append(np.full(len(f), k))
        n += len(v)
    return MeshData(np.concatenate(V).astype(np.float32), np.concatenate(F).astype(np.int32),
                    np.concatenate(ids).astype(np.int32))


def rescuesim_collision_mask(prims) -> np.ndarray:
    """Which prims really collide in rescuesim: read back from its own PyBullet CollisionWorld.

    PyBullet keeps at most 16 children per compound (MAX_COMPOUND_COLLISION_SHAPES); rescuesim
    chunks its compounds accordingly, so every collidable prim is present and this mask is all of
    them. Reading it back keeps gpusim exact if the reference world ever drops a shape again."""
    import pybullet as p
    from collections import Counter
    from rescuesim.world import CollisionWorld
    codes = {p.GEOM_BOX: "box", p.GEOM_SPHERE: "sphere", p.GEOM_CYLINDER: "cylinder", p.GEOM_CAPSULE: "capsule"}
    cw = CollisionWorld(list(prims))
    try:
        present = Counter()
        for uid in cw.prop_bodies:
            for d in p.getCollisionShapeData(uid, -1, physicsClientId=cw.cid):
                present[(codes.get(d[2]), tuple(np.round(d[5], 3)))] += 1
    finally:
        cw.close()
    mask = np.zeros(len(prims), bool)
    for k, pr in enumerate(prims):
        key = (pr.shape, tuple(np.round(pr.pos, 3)))
        if pr.collides and present[key] > 0:
            present[key] -= 1
            mask[k] = True
    return mask


def _prim_table(prims, mask=None) -> PrimTable:
    mask = np.array([pr.collides for pr in prims], bool) if mask is None else mask
    rows = [(k, pr) for k, pr in enumerate(prims) if mask[k]]
    K = max(1, len(rows))
    t = PrimTable(np.full(K, -1, np.int32), np.zeros((K, 3), np.float32), np.tile(np.eye(3, dtype=np.float32), (K, 1, 1)),
                  np.zeros((K, 3), np.float32), np.full((K, 3), -1001.0, np.float32), np.full((K, 3), -1000.0, np.float32),
                  np.full(K, -1, np.int32))
    for j, (k, pr) in enumerate(rows):
        t.shape[j] = SHAPE_CODES[pr.shape]
        t.pos[j] = pr.pos
        t.rot[j] = quat_to_matrix(pr.quat)
        if pr.shape == "box":
            t.size[j] = pr.size
        elif pr.shape == "sphere":
            t.size[j] = (pr.size[0], 0.0, 0.0)
        else:                                   # cylinder: total height; capsule: segment length
            t.size[j] = (pr.size[0], pr.size[1] / 2.0, 0.0)
        lo, hi = pr.aabb()
        t.lo[j], t.hi[j] = np.asarray(lo) - 1e-3, np.asarray(hi) + 1e-3
        t.prim_index[j] = k
    return t


_CACHE: dict = {}


def load_terrain_scene(kind: str, layout_seed: int | None = None, match_rescuesim: bool = True) -> TerrainScene:
    """GPU-ready description of one rescuesim terrain type (all its poses).

    ``match_rescuesim``: collide only with the prims that collide in rescuesim's PyBullet world
    (see ``rescuesim_collision_mask``); False makes every prim solid, as SPEC section 5 intends."""
    seed = DEFAULT_LAYOUT_SEEDS[kind] if layout_seed is None else int(layout_seed)
    key = (kind, seed, bool(match_rescuesim))
    if key not in _CACHE:
        lay = get_layout(kind, seed)
        boxes = np.stack([target_obb(p) for p in lay.poses]) if lay.poses else np.zeros((0, 15), np.float32)
        table = _prim_table(lay.prims, rescuesim_collision_mask(lay.prims) if match_rescuesim else None)
        _CACHE[key] = TerrainScene(kind, seed, lay, _visual_mesh(lay.prims), table, boxes)
    return _CACHE[key]


def obb_from_pose(x: float, y: float, yaw: float) -> np.ndarray:
    """Target OBB (15,) for an arbitrary target placement on the ground (tests)."""
    from rescuesim.scene import TargetPose
    return target_obb(TargetPose(float(x), float(y), float(math.atan2(math.sin(yaw), math.cos(yaw)))))


class GpuScenes:
    """Several TerrainScenes uploaded to one Warp device.

    Per terrain index t: ``meshes[t]`` (visual mesh id, for ray casting), ``bvhs[t]`` (BVH over its
    collision primitives) and ``prim_off[t]`` (offset of its rows in the concatenated primitive
    arrays ``shape``, ``ppos``, ``prot``, ``psize``)."""

    def __init__(self, scenes: list, device: str = "cuda:0"):
        import warp as wp
        self.scenes, self.device = list(scenes), device
        self._keep = []
        mesh_ids, bvh_ids, offs = [], [], []
        tables = []
        n = 0
        for sc in self.scenes:
            pts = wp.array(sc.visual.vertices, dtype=wp.vec3, device=device)
            idx = wp.array(sc.visual.faces.ravel(), dtype=wp.int32, device=device)
            mesh = wp.Mesh(points=pts, indices=idx)
            lo = wp.array(sc.prims.lo, dtype=wp.vec3, device=device)
            hi = wp.array(sc.prims.hi, dtype=wp.vec3, device=device)
            bvh = wp.Bvh(lo, hi)
            self._keep += [pts, idx, mesh, lo, hi, bvh]
            mesh_ids.append(mesh.id)
            bvh_ids.append(bvh.id)
            offs.append(n)
            n += len(sc.prims.shape)
            tables.append(sc.prims)
        self.meshes = wp.array(np.array(mesh_ids, np.uint64), dtype=wp.uint64, device=device)
        self.bvhs = wp.array(np.array(bvh_ids, np.uint64), dtype=wp.uint64, device=device)
        self.prim_off = wp.array(np.array(offs, np.int32), dtype=wp.int32, device=device)
        self.shape = wp.array(np.concatenate([t.shape for t in tables]), dtype=wp.int32, device=device)
        self.ppos = wp.array(np.concatenate([t.pos for t in tables]), dtype=wp.vec3, device=device)
        self.prot = wp.array(np.concatenate([t.rot for t in tables]), dtype=wp.mat33, device=device)
        self.psize = wp.array(np.concatenate([t.size for t in tables]), dtype=wp.vec3, device=device)
