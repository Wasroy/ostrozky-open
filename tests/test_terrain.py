import itertools
import math

import pybullet as p
import pytest

from rescuesim import TERRAIN_TYPES, generate_terrain, parse_terrain, stable_hash
from rescuesim.scene import PROP_EXTENT_M
from rescuesim.terrain import get_layout
from rescuesim.world import CollisionWorld, PoseValidator


def test_parse_and_hash():
    assert parse_terrain("forest@v3") == ("forest", 3)
    with pytest.raises(ValueError):
        parse_terrain("forest")
    with pytest.raises(ValueError):
        parse_terrain("moon@v1")
    assert isinstance(stable_hash("forest@v3"), int)
    assert stable_hash("forest@v3") == stable_hash("forest@v3") != stable_hash("forest@v4")


@pytest.mark.parametrize("kind", TERRAIN_TYPES)
def test_pose_constraints(kind):
    lay = get_layout(kind)
    poses = lay.poses
    assert len(poses) >= 40
    for a, b in itertools.combinations(poses, 2):
        assert math.hypot(a.x - b.x, a.y - b.y) >= 12.0
    for pr in (q for q in lay.prims if q.collides):          # props (not ground decals)
        lo, hi = pr.aabb()
        assert max(abs(lo[0]), abs(hi[0]), abs(lo[1]), abs(hi[1])) <= PROP_EXTENT_M + 0.5
    world = CollisionWorld(list(lay.prims))
    val = PoseValidator(world)
    try:
        for pose in poses:
            world.set_target(pose)
            # no prop (except bushes) overlaps the box: independent PyBullet check
            for uid in world.prop_bodies:
                if uid in world.bush_bodies:
                    continue
                pts = p.getClosestPoints(world.target, uid, 0.0, physicsClientId=world.cid)
                assert not pts or min(q[8] for q in pts) >= -1e-4, (kind, pose)
            assert val.corridor(pose) is not None, (kind, pose)
            assert abs(pose.x) <= 140 and abs(pose.y) <= 140
    finally:
        world.close()


def test_layouts_are_deterministic_and_seedable():
    a = generate_terrain("rubble", None, 5)
    b = generate_terrain("rubble", None, 5)
    assert a.target == b.target and len(a.prims) == len(b.prims)
    c = generate_terrain("rubble", 12345, 5)
    assert c.layout_seed == 12345 and len(c.poses) >= 40
    assert c.prims != a.prims
    with pytest.raises(ValueError):
        generate_terrain("rubble", None, len(a.poses) + 1)


@pytest.mark.parametrize("kind", TERRAIN_TYPES)
def test_targets_hidden_somewhere_but_visible_from_a_low_viewpoint(kind):
    """Render-based check on the first poses: the detector mask is non-empty from at
    least one of 16 low viewpoints 25 m away (aimed at the box) and empty from at
    least one viewpoint of a wider set."""
    from rescuesim.diagnostics import pose_visibility
    lay = get_layout(kind)
    for k in range(6):
        vis = pose_visibility(lay, k + 1)
        assert vis["low_visible"] >= 1, (kind, k + 1, vis)
        assert vis["hidden_any"] >= 1, (kind, k + 1, vis)
