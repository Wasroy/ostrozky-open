"""Terrain generation entry points: layouts, target poses, terrain strings."""
from __future__ import annotations

import math
import re
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace

import numpy as np

from .layouts import BUILDERS
from .scene import Prim, Scene, TargetPose

TERRAIN_TYPES = ("field", "forest", "village", "city", "rubble")
DEFAULT_LAYOUT_SEEDS = {"field": 1101, "forest": 2202, "village": 3303, "city": 4404, "rubble": 5505, "empty": 9909}
N_POSES = 48            # poses kept per layout (v1..v48)
MIN_POSES = 40
MIN_SPACING_M = 12.0
POSE_EXTENT_M = 140.0

_TERRAIN_RE = re.compile(r"^\s*([a-z]+)\s*@\s*v(\d+)\s*$")


def stable_hash(text: str) -> int:
    """Process-independent 32-bit hash (CRC-32 of the UTF-8 text)."""
    return zlib.crc32(text.encode("utf-8")) & 0xFFFFFFFF


def parse_terrain(spec: str) -> tuple[str, int]:
    """'forest@v3' -> ('forest', 3)."""
    m = _TERRAIN_RE.match(spec)
    if not m:
        raise ValueError(f"terrain must look like '<type>@v<N>', got {spec!r}")
    kind, n = m.group(1), int(m.group(2))
    if kind not in BUILDERS:
        raise ValueError(f"unknown terrain type {kind!r}; choose from {TERRAIN_TYPES}")
    if n < 1:
        raise ValueError("pose index N in '@v<N>' starts at 1")
    return kind, n


@dataclass(frozen=True)
class Layout:
    kind: str
    layout_seed: int
    prims: tuple
    poses: tuple
    ground_rgba: tuple
    stats: dict = field(default_factory=dict, compare=False, hash=False)


@dataclass(frozen=True)
class Terrain:
    """One world: a layout plus the selected target pose (1-based ``pose_index``)."""
    layout: Layout
    pose_index: int

    @property
    def kind(self) -> str:
        return self.layout.kind

    @property
    def layout_seed(self) -> int:
        return self.layout.layout_seed

    @property
    def prims(self) -> tuple:
        return self.layout.prims

    @property
    def poses(self) -> tuple:
        return self.layout.poses

    @property
    def target(self) -> TargetPose:
        return self.layout.poses[self.pose_index - 1]

    @property
    def name(self) -> str:
        return f"{self.kind}@v{self.pose_index}"


_CACHE: dict = {}


def get_layout(kind: str, layout_seed: int | None = None) -> Layout:
    if kind not in BUILDERS:
        raise ValueError(f"unknown terrain type {kind!r}; choose from {TERRAIN_TYPES}")
    seed = DEFAULT_LAYOUT_SEEDS[kind] if layout_seed is None else int(layout_seed)
    key = (kind, seed)
    if key not in _CACHE:
        _CACHE[key] = _build_layout(kind, seed)
    return _CACHE[key]


def generate_terrain(kind: str, layout_seed: int | None = None, pose_index: int = 1) -> Terrain:
    """Build (or fetch from the in-process cache) a terrain.

    ``kind``: one of TERRAIN_TYPES (plus ``"empty"`` for tests);
    ``layout_seed``: None for the fixed default layout of that type, any int for
    a new unseen layout; ``pose_index``: 1-based target pose (``v<N>``)."""
    lay = get_layout(kind, layout_seed)
    if not 1 <= pose_index <= len(lay.poses):
        raise ValueError(f"{kind} layout {lay.layout_seed} has poses v1..v{len(lay.poses)}, got v{pose_index}")
    return Terrain(lay, int(pose_index))


def _interleave(cands, rng):
    by_kind = defaultdict(list)
    for c in cands:
        by_kind[c.kind].append(c)
    kinds = sorted(by_kind)
    for k in kinds:
        idx = rng.permutation(len(by_kind[k]))
        by_kind[k] = [by_kind[k][i] for i in idx]
    out, depth = [], 0
    while any(depth < len(by_kind[k]) for k in kinds):
        for k in kinds:
            if depth < len(by_kind[k]):
                out.append(by_kind[k][depth])
        depth += 1
    return out


def _mask_any(render, eye, pose) -> bool:
    from .sensors import aimed_camera
    mask, _, _ = render.render_target_mask(aimed_camera(eye, pose.center))
    return mask is not None and bool(mask.any())


def _build_layout(kind: str, seed: int) -> Layout:
    from .world import CollisionWorld, PoseValidator, RenderWorld   # PyBullet only needed here

    builder, ground = BUILDERS[kind]
    rng = np.random.default_rng(seed)
    sc = Scene(rng, ground)
    builder(sc)
    prims = tuple(sc.prims)
    world = CollisionWorld(list(prims))
    val = PoseValidator(world)
    render = RenderWorld(list(prims), ground) if kind != "empty" else None
    stats = Counter(candidates=len(sc.candidates))
    accepted: list[TargetPose] = []
    need_hidden = kind != "empty"
    try:
        for cand in _interleave(sc.candidates, rng):
            if len(accepted) >= N_POSES:
                break
            if max(abs(cand.x), abs(cand.y)) > POSE_EXTENT_M:
                stats["out_of_extent"] += 1
                continue
            if any(math.hypot(cand.x - a.x, cand.y - a.y) < MIN_SPACING_M for a in accepted):
                stats["too_close"] += 1
                continue
            yaw = math.atan2(math.sin(cand.yaw), math.cos(cand.yaw))
            pose = TargetPose(round(cand.x, 3), round(cand.y, 3), round(yaw, 4), cand.kind)
            if val.box_overlaps(pose):
                stats["overlap"] += 1
                continue
            world.set_target(pose)
            low =np.concatenate([val.viewpoints(pose, 25.0, 2.0), val.viewpoints(pose, 25.0, 4.0)])
            low_free = np.array([val.free_point(v) for v in low])
            low_rays = val.visible_rays(pose, low)
            if not np.any(low_free & (low_rays >= 2)):
                stats["not_visible_low"] += 1
                continue
            if need_hidden:
                far = np.concatenate([val.viewpoints(pose, 60.0, 15.0), val.viewpoints(pose, 40.0, 25.0),
                                      [[pose.x + 0.01, pose.y, 45.0]]])
                far_free = np.array([val.free_point(v) for v in far])
                far_rays = val.visible_rays(pose, far)
                # rays sample the box sparsely: confirm with detector renders (exact pixels)
                render.set_target(pose)
                order = np.argsort(-low_rays * low_free)
                seen = any(_mask_any(render, low[i], pose) for i in order[:3] if low_free[i] and low_rays[i] >= 2)
                if not seen:
                    stats["not_visible_low"] += 1
                    continue
                pts = np.concatenate([low, far])
                free = np.concatenate([low_free, far_free])
                rays = np.concatenate([low_rays, far_rays])
                hidden = any(not _mask_any(render, pts[i], pose) for i in np.flatnonzero(free & (rays == 0))[:6])
                if not hidden:
                    stats["not_hidden"] += 1
                    continue
            corr = val.corridor(pose)
            if corr is None:
                stats["no_corridor"] += 1
                continue
            accepted.append(replace(pose, approach=corr))
    finally:
        world.close()
        if render is not None:
            render.close()
    stats["accepted"] = len(accepted)
    stats["kinds"] = dict(Counter(a.kind for a in accepted))
    if len(accepted) < MIN_POSES:
        raise RuntimeError(f"{kind} layout {seed}: only {len(accepted)} valid target poses ({dict(stats)})")
    return Layout(kind, seed, prims, tuple(accepted), tuple(ground), dict(stats))
