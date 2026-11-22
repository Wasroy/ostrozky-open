"""gpusim against rescuesim: quick dynamics, rendering, starts, outcomes and API checks.

The Warp tests are skipped when warp-lang or a CUDA device is missing (``pip install -e .[gpu]``).
"""
import math

import numpy as np
import pytest

from rescuesim import RescueEnv
from rescuesim import sensors as S
from rescuesim.geometry import quat_from_euler, quat_to_matrix
from gpusim.dynamics import BatchQuad


def _cuda_ok() -> bool:
    try:
        import warp as wp
        wp.config.quiet = True
        wp.init()
        return wp.is_cuda_available()
    except Exception:
        return False


needs_gpu = pytest.mark.skipif(not _cuda_ok(), reason="needs warp-lang and a CUDA device")
KINDS = ("field", "forest", "village", "city", "rubble")
START = {"pos": (180.0, 190.0, 60.0), "yaw": 0.4, "speed": 5.0}
STICKS = np.array([[0, 0.5, 0, 0.3], [0.6, 0, 0.5, 0.2], [0.3, -1, -1, 1], [-1, 1, 0, -0.5]], float)


def _rescuesim_open_loop(actions, steps):
    env = RescueEnv("empty@v1", render_depth=False, render_detector=False)
    out = []
    for a in actions:
        env.reset(seed=0, options={"start_pos": START["pos"], "start_yaw": START["yaw"], "start_speed": START["speed"]})
        for _ in range(steps):
            env.step(a)
        out.append((env.quad.state.pos.copy(), np.array(env.quad.state.quat)))
    env.close()
    return out


def test_numpy_dynamics_matches_rescuesim():
    ref = _rescuesim_open_loop(STICKS, 60)
    q = BatchQuad(len(STICKS))
    q.reset(np.arange(len(STICKS)), np.tile(START["pos"], (len(STICKS), 1)), START["yaw"], START["speed"])
    for _ in range(60):
        q.step(STICKS)
    for j, (p, quat) in enumerate(ref):
        assert np.allclose(q.pos[j], p, atol=1e-9) and np.allclose(q.quat[j], quat, atol=1e-9)


@pytest.fixture(scope="module")
def env5():
    from gpusim.fast import FastBatchEnv
    return FastBatchEnv(5, terrains=KINDS, obs="full", noise=False)


@needs_gpu
def test_warp_dynamics_matches_rescuesim():
    from gpusim.fast import FastBatchEnv
    ref = _rescuesim_open_loop(STICKS, 60)
    m = len(STICKS)
    env = FastBatchEnv(m, terrains=("empty",), noise=False)
    env.reset(terrains=["empty@v1"] * m, runs=[0] * m, starts=[START] * m)
    for _ in range(60):
        env.step(STICKS)
    tr = env.truth()
    for j, (p, quat) in enumerate(ref):
        assert np.linalg.norm(tr["pos"][j] - p) < 0.01                       # < 1 cm after 2 s
        cosang = abs(float(np.dot(tr["quat"][j], quat)))
        assert math.degrees(2 * math.acos(min(1.0, cosang))) < 0.1


@needs_gpu
def test_starts_and_first_observation_match_rescuesim(env5):
    terr = [f"{k}@v3" for k in KINDS]
    obs = env5.reset(terrains=terr, runs=[2] * 5)
    for i, t in enumerate(terr):
        env = RescueEnv(t)
        o, info = env.reset(seed=0, options={"run": 2})
        s = env.start
        assert np.allclose(env5.start[i], [s["x"], s["y"], s["z"], s["yaw"], s["speed"]], atol=1e-9)
        assert np.allclose(obs["vio_xyz_m"][i], info["pos"], atol=1e-4)      # noise off: anchored at the start
        assert np.allclose(obs["accel_m_s2"][i], [0, 0, 9.81], atol=1e-4)
        env.close()


@needs_gpu
def test_render_matches_rescuesim(env5):
    from rescuesim.world import RenderWorld
    from gpusim.render import Renderer
    ren = Renderer(env5.gpu)
    rng = np.random.default_rng(3)
    for ti in (1, 3):
        sc = env5.scenes[ti]
        lay = sc.layout
        rw = RenderWorld(list(lay.prims), lay.ground_rgba)
        for v in (1, 5, 9):
            pose = lay.poses[v - 1]
            a = rng.uniform(-math.pi, math.pi)
            pos = np.array([pose.x + 25 * math.cos(a), pose.y + 25 * math.sin(a), 12.0])
            q = quat_from_euler(0.0, 0.2, a + math.pi)
            rw.set_target(pose)
            cam = S.camera_frame(pos, quat_to_matrix(q))
            z = np.clip(rw.render_depth(cam), 0, 40)
            mask, off, _ = rw.render_target_mask(cam)
            gd, gb = ren.render([ti], pos[None], np.array(q)[None], sc.boxes[v - 1][None])
            assert np.median(np.abs(gd[0] - z)) < 0.01
            assert np.mean(np.abs(gd[0] - z) > 0.05) < 0.05          # silhouette pixels may differ
            if mask is not None and mask.any():
                cols, rows = np.flatnonzero(mask.any(0)), np.flatnonzero(mask.any(1))
                ref = [off[0] + cols[0], off[1] + rows[0], off[0] + cols[-1] + 1, off[1] + rows[-1] + 1]
                assert np.abs(gb[0] - ref).max() <= 1
            else:
                assert gb[0, 0] < 0
        rw.close()


@needs_gpu
def test_api_shapes_and_single(env5):
    from gpusim.fast import FLYING
    obs = env5.reset()
    assert obs["depth_m"].shape == (5, 126, 224) and obs["depth_pool"].shape == (5, 9, 16)
    obs, done, outcome, info = env5.step(np.zeros((5, 4)))
    assert done.shape == (5,) and (outcome == FLYING).all() and info["distance_m"].shape == (5,)
    env = RescueEnv("village@v1")
    o_ref, _ = env.reset(seed=0)
    env.close()
    o = env5.single(obs, 2)
    assert set(o) == set(o_ref)
    for k in o_ref:
        assert np.asarray(o[k]).dtype == np.asarray(o_ref[k]).dtype and np.shape(o[k]) == np.shape(o_ref[k])
    # pooled = min over 14 x 14 blocks of the depth image
    assert np.allclose(obs["depth_pool"][0], obs["depth_m"][0].reshape(9, 14, 16, 14).min(axis=(1, 3)))


def _outcome_both(terrain, start, action, steps, timeout_s=120.0):
    """Outcome and step count of the same scripted flight in rescuesim and gpusim."""
    from gpusim.fast import FastBatchEnv
    env = RescueEnv(terrain, max_time_s=timeout_s)
    env.reset(seed=0, options={"start_pos": start["pos"], "start_yaw": start["yaw"], "start_speed": start["speed"]})
    ref = (None, 0)
    for k in range(steps):
        _, _, term, trunc, info = env.step(action)
        if term or trunc:
            ref = (info["outcome"], k + 1)
            break
    env.close()
    kind = terrain.split("@")[0]
    g = FastBatchEnv(1, terrains=(kind,), noise=False, timeout_s=timeout_s)
    g.reset(terrains=[terrain], runs=[0], starts=[start])
    got = (None, 0)
    for k in range(steps):
        _, done, outcome, _ = g.step(np.asarray(action, float)[None])
        if done[0]:
            got = (g.outcome_name(outcome[0]), k + 1)
            break
    return ref, got


@needs_gpu
def test_outcomes_match_rescuesim():
    from gpusim.scene import load_terrain_scene
    sc = load_terrain_scene("field")
    v = 1 + next(i for i, p in enumerate(sc.layout.poses) if p.approach == "above")
    c = sc.boxes[v - 1, :3]
    # descend onto the box top through its free shaft -> success
    ref, got = _outcome_both(f"field@v{v}", {"pos": (c[0], c[1], c[2] + 2.0), "yaw": 0.0, "speed": 0.0}, [0, 0, 0, -0.4], 90)
    assert ref[0] == "success" and got == ref
    # fly into a prop -> collision
    city = load_terrain_scene("city")
    k = next(j for j in range(len(city.prims.shape)) if city.prims.shape[j] == 0 and city.prims.size[j, 2] > 5)
    ctr, R, h = city.prims.pos[k], city.prims.rot[k], city.prims.size[k]
    side = R[:, 0]
    p0 = ctr + side * (h[0] + 6.0)
    p0[2] = 3.0
    yaw = math.atan2(-side[1], -side[0])
    ref, got = _outcome_both("city@v1", {"pos": tuple(p0), "yaw": yaw, "speed": 8.0}, [0, 0.2, 0, 0.1], 60)
    assert ref[0] == "collision" and got == ref
    # ground -> collision; arena edge -> out_of_arena; time limit -> timeout
    ref, got = _outcome_both("field@v1", {"pos": (200.0, 200.0, 1.0), "yaw": 0.0, "speed": 0.0}, [0, 0, 0, -1], 30)
    assert ref[0] == "collision" and got == ref
    ref, got = _outcome_both("field@v1", {"pos": (349.0, 0.0, 20.0), "yaw": 0.0, "speed": 10.0}, [0, 0, 0, 0], 30)
    assert ref[0] == "out_of_arena" and got == ref
    ref, got = _outcome_both("field@v1", {"pos": (200.0, 200.0, 20.0), "yaw": 0.0, "speed": 0.0}, [0, 0, 0, 0], 30, 0.5)
    assert ref[0] == "timeout" and got == ref


@needs_gpu
def test_invalid_action_and_frozen_flights(env5):
    from gpusim.fast import ERROR, FLYING
    env5.reset()
    act = np.zeros((5, 4))
    act[1, 0] = np.nan
    act[3, 3] = 1.5
    obs, done, outcome, _ = env5.step(act)
    assert list(outcome) == [FLYING, ERROR, FLYING, ERROR, FLYING]
    before = env5.truth()["pos"][1].copy()
    obs2, done, outcome, _ = env5.step(np.zeros((5, 4)))
    assert np.allclose(env5.truth()["pos"][1], before) and outcome[1] == ERROR
    assert np.array_equal(obs2["depth_m"][1], obs["depth_m"][1])
    env5.reset(idx=[1, 3])
    assert (env5.truth()["outcome"] == FLYING).all()


@needs_gpu
def test_noise_is_reproducible_per_flight():
    from gpusim.fast import FastBatchEnv
    a = FastBatchEnv(2, terrains=("field", "city"), obs="pooled")
    b = FastBatchEnv(3, terrains=("field", "city"), obs="pooled")
    oa = a.reset(terrains=["city@v4", "field@v2"], runs=[5, 6])
    ob = b.reset(terrains=["field@v9", "field@v2", "city@v4"], runs=[1, 6, 5])
    for _ in range(3):
        oa, *_ = a.step(np.tile([0, 0.2, 0, 0.1], (2, 1)))
        ob, *_ = b.step(np.tile([0, 0.2, 0, 0.1], (3, 1)))
    for k in ("depth_pool", "gyro_rad_s", "accel_m_s2", "vio_xyz_m"):
        assert np.allclose(oa[k][0], ob[k][2], atol=1e-5) and np.allclose(oa[k][1], ob[k][1], atol=1e-5)
    assert np.isfinite(oa["depth_pool"]).all()


@needs_gpu
def test_rescuesim_noise_mode_reproduces_observations():
    from gpusim.fast import FastBatchEnv
    g = FastBatchEnv(1, terrains=("village",), obs="full", noise="rescuesim")
    og = g.reset(terrains=["village@v2"], runs=[1])
    env = RescueEnv("village@v2")
    o, _ = env.reset(seed=0, options={"run": 1})
    for step in range(20):
        b = g.single(og, 0)
        assert o["detected"] == b["detected"] and np.allclose(o["box"], b["box"], atol=1e-5)
        assert np.median(np.abs(o["depth_m"] - b["depth_m"])) < 1e-3
        for k in ("gyro_rad_s", "accel_m_s2", "vio_xyz_m"):
            assert np.allclose(o[k], b[k], atol=2e-3)
        o, *_ = env.step([0.1, 0.4, 0.2, 0.2])
        og, *_ = g.step(np.array([[0.1, 0.4, 0.2, 0.2]]))
    env.close()
