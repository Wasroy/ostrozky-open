import math

import numpy as np
import pytest

from conftest import fly
from rescuesim import sensors as S
from rescuesim.geometry import quat_from_euler, quat_to_matrix
from rescuesim.terrain import generate_terrain
from rescuesim.world import RenderWorld


def test_camera_geometry_constants():
    assert S.V_FOV_DEG == pytest.approx(69.6, abs=0.1)
    assert (S.DEPTH_W, S.DEPTH_H) == (224, 126) and (S.DET_W, S.DET_H) == (160, 90)


def test_depth_shape_and_sky_value(empty_env_cams):
    h = fly(empty_env_cams, [0, 0, 0, 0], 5, pos=(0, -200, 20), yaw=0.0)
    d = h[-1][0]["depth_m"]
    assert d.shape == (126, 224) and d.dtype == np.float32
    # level camera looks 25 deg up: everything is beyond 40 m -> clipped, then noise
    assert d.mean() == pytest.approx(40.0, abs=0.05)
    assert d.std() == pytest.approx(1.2, abs=0.05)
    assert d.max() > 41.0                           # no re-clip after the noise


def test_depth_clip_then_noise():
    rng = np.random.default_rng(0)
    z = np.full((126, 224), 100.0)
    d = S.noisy_depth(z, rng)
    assert d.mean() == pytest.approx(40.0, abs=0.03) and d.std() == pytest.approx(1.2, abs=0.03)
    d = S.noisy_depth(np.full((126, 224), 0.5), rng)
    assert d.std() == pytest.approx(0.03, abs=0.003)
    d = S.noisy_depth(np.full((126, 224), 10.0), rng)
    assert d.std() == pytest.approx(0.3, abs=0.01)


def test_depth_is_z_depth_of_a_wall():
    from rescuesim.scene import Prim
    rw = RenderWorld([Prim("box", (0.5, 20.0, 20.0), (10.5, 0.0, 10.0))], (0.4, 0.5, 0.3, 1))
    try:
        # pitch the body 25 deg nose down so the camera axis is horizontal, eye at x=0
        R = quat_to_matrix(quat_from_euler(0.0, math.radians(25), 0.0))
        pos = np.array([0.0, 0.0, 10.0]) - R[:, 0] * S.CAM_OFFSET_M
        z = rw.render_depth(S.camera_frame(pos, R))
        # a plane perpendicular to the axis has the same z-depth everywhere
        assert np.allclose(z, 10.0, atol=0.01)
    finally:
        rw.close()


def test_detector_miss_rate_range_and_jitter():
    rng = np.random.default_rng(1)
    mask = np.zeros((90, 160), bool)
    mask[40:50, 60:100] = True
    hits, boxes = 0, []
    for _ in range(4000):
        det, box = S.detect_from_mask(mask, (0, 0), 100.0, rng)
        hits += det
        if det:
            boxes.append(box)
    assert hits / 4000 == pytest.approx(0.95, abs=0.012)
    b = np.array(boxes)
    assert np.allclose(b.mean(0), [60 / 160, 40 / 90, 100 / 160, 50 / 90], atol=0.002)
    assert b[:, 0].std() * 160 == pytest.approx(0.5, rel=0.1)        # sigma = max(0.5 px, 1% of 40 px)
    assert np.all(b[:, 0] <= b[:, 2]) and np.all((b >= 0) & (b <= 1))
    assert all(S.detect_from_mask(mask, (0, 0), 250.5, rng)[0] == 0 for _ in range(200))
    assert S.detect_from_mask(np.zeros((5, 5), bool), (0, 0), 10.0, rng)[0] == 0


def test_detector_in_scene_range_and_rate(empty_env_cams):
    env = empty_env_cams
    t = env.target
    def run(dist):
        pos = (t.x - dist - 2.0, t.y, 1.0)
        h = fly(env, [0, 0, 0, 0], 200, pos=pos, yaw=0.0)
        return np.array([o["detected"] for o, _ in h[1:]]), h
    det, h = run(30.0)
    assert det.mean() == pytest.approx(0.95, abs=0.04)
    boxes = np.array([o["box"] for o, _ in h[1:] if o["detected"]])
    assert abs(boxes[:, [0, 2]].mean() - 0.5) < 0.02          # box centred horizontally
    det, _ = run(262.0)
    assert det.sum() == 0                                       # beyond 250 m from the camera


def test_vio_anchoring_and_drift(empty_env_cams):
    h = fly(empty_env_cams, [0, 0.3, 0, 0.05], 150, pos=(10, -230, 20), yaw=math.pi / 2, speed=5.0)
    v0 = h[0][0]["vio_xyz_m"]
    assert v0[0] == pytest.approx(10.0, abs=1e-4) and v0[1] == pytest.approx(-230.0, abs=1e-4)
    err = np.array([o["vio_xyz_m"][:2] - i["pos"][:2] for o, i in h])
    assert 0.0 < np.abs(err[-1]).max() < 3.0
    alt_err = np.array([o["vio_xyz_m"][2] - i["pos"][2] for o, i in h])
    assert abs(alt_err.mean()) < 0.02 and alt_err.std() == pytest.approx(0.05, rel=0.2)


@pytest.mark.parametrize("kind", ["forest", "city"])
def test_culled_cropped_render_equals_full_render(kind):
    ter = generate_terrain(kind, None, 3)
    rw = RenderWorld(list(ter.prims), ter.layout.ground_rgba)
    rng = np.random.default_rng(0)
    try:
        for k in range(15):
            tgt = ter.poses[k]
            rw.set_target(tgt)
            a, d = rng.uniform(0, 2 * math.pi), rng.uniform(5, 60)
            pos = np.array([tgt.x + d * math.cos(a), tgt.y + d * math.sin(a), rng.uniform(1.5, 15)])
            look = math.atan2(tgt.y - pos[1], tgt.x - pos[0]) + rng.normal(0, 0.3)
            cam = S.camera_frame(pos, quat_to_matrix(quat_from_euler(rng.normal(0, 0.2), rng.uniform(0, 0.9), look)))
            rw.culling, rw.full_window = True, False
            m1, off, _ = rw.render_target_mask(cam)
            z1 = rw.render_depth(cam)
            rw.culling, rw.full_window = False, True
            m2, _, _ = rw.render_target_mask(cam)
            z2 = rw.render_depth(cam)
            full1 = np.zeros((90, 160), bool)
            if m1 is not None:
                full1[off[1]:off[1] + m1.shape[0], off[0]:off[0] + m1.shape[1]] = m1
            assert np.array_equal(full1, m2 if m2 is not None else np.zeros((90, 160), bool))
            assert np.allclose(np.minimum(z1, 40), np.minimum(z2, 40), atol=1e-3)
    finally:
        rw.close()


@pytest.mark.parametrize("frac,expect_col,expect_row", [(0.3, 0, 0), (0.7, 1, 1)])
def test_pixels_sample_their_centres(frac, expect_col, expect_row):
    """A wall edge at (112 + frac) px covers column 112 only if 112.5 is past the edge;
    a horizontal edge at (63 + frac) px covers row 63 only if 63.5 is below it."""
    from rescuesim.scene import Prim
    D, W, H = 10.0, S.DEPTH_W, S.DEPTH_H
    y_edge = -(2 * (112 + frac) / W - 1) * S.TAN_H * D          # camera right is -y
    z_edge = (1 - 2 * (63 + frac) / H) * S.TAN_V * D
    prims = [Prim("box", (0.5, 20.0, 30.0), (D + 0.5, y_edge - 20.0, 5.0)),       # right half, vertical edge
             Prim("box", (0.5, 20.0, 30.0), (D + 0.5, 25.0, 5.0 + z_edge - 30.0))]  # left side, top edge
    rw = RenderWorld(prims, (0.4, 0.5, 0.3, 1))
    try:
        z = rw.render_depth(S.aimed_camera((0, 0, 5), (1, 0, 5)))
        assert np.flatnonzero(z[20] < 20).min() == 112 + expect_col
        assert np.flatnonzero(z[:, 20] < 20).min() == 63 + expect_row
    finally:
        rw.close()
