import math

import numpy as np
import pytest

from conftest import fly
from rescuesim import RescueEnv


def test_success_on_box(empty_env):
    t = empty_env.target
    h = fly(empty_env, [0, 0, 0, -1], 120, pos=(t.x, t.y, 3.0), yaw=0.0)
    _, info = h[-1]
    assert info["outcome"] == "success" and info["is_success"] and not info["collision"]
    assert info["distance_m"] < 0.1


def test_success_reward_and_flags(empty_env):
    t = empty_env.target
    empty_env.reset(seed=0, options={"start_pos": (t.x, t.y, 3.0)})
    rewards = []
    while True:
        _, r, term, trunc, info = empty_env.step([0, 0, 0, -1])
        rewards.append(r)
        if term or trunc:
            break
    assert term and not trunc and rewards[-1] == 1.0 and sum(rewards) == 1.0


def test_ground_collision(empty_env):
    h = fly(empty_env, [0, 0, 0, -1], 120, pos=(0, -200, 3.0), yaw=0.0)
    info = h[-1][1]
    assert info["outcome"] == "collision" and info["collision"]
    assert info["pos"][2] < 0.2


@pytest.fixture(scope="module")
def village():
    env = RescueEnv("village@v1", render_depth=False, render_detector=False)
    yield env
    env.close()


def test_collision_with_a_prop(village):
    house = next(p for p in village.terrain.prims if p.tag == "house")
    top = house.pos[2] + house.size[2]
    h = fly(village, [0, 0, 0, -1], 200, pos=(house.pos[0], house.pos[1], top + 8.0), yaw=0.0)
    info = h[-1][1]
    assert info["outcome"] == "collision" and info["pos"][2] > top - 0.5   # stopped on the roof


def test_flying_into_a_wall_is_a_collision(village):
    house = next(p for p in village.terrain.prims if p.tag == "house")
    # start 6 m in front of the wall centre at half height, facing it, and pitch forward
    yaw = 2 * math.atan2(house.quat[2], house.quat[3])
    hx = house.size[0]
    sx = house.pos[0] + math.cos(yaw) * (hx + 6.0)
    sy = house.pos[1] + math.sin(yaw) * (hx + 6.0)
    h = fly(village, [0, 0.3, 0, 0.1], 150, pos=(sx, sy, 2.0), yaw=yaw + math.pi)
    assert h[-1][1]["outcome"] == "collision"


def test_out_of_arena(empty_env):
    h = fly(empty_env, [0, 0.6, 0, 0.3], 300, pos=(340.0, 0.0, 20.0), yaw=0.0, speed=10.0)
    info = h[-1][1]
    assert info["outcome"] == "out_of_arena" and info["pos"][0] >= 350.0


def test_timeout():
    env = RescueEnv("empty@v1", render_depth=False, render_detector=False, max_time_s=2.0)
    try:
        env.reset(seed=0)
        n = 0
        while True:
            _, _, term, trunc, info = env.step([0, 0, 0, 0])
            n += 1
            if term or trunc:
                break
        assert info["outcome"] == "timeout" and trunc and not term and n == 60
    finally:
        env.close()


@pytest.mark.parametrize("bad", [[0, 0, 0], [0, 0, 0, float("nan")], [0, 1.5, 0, 0], [0, 0, float("inf"), 0], "abcd", None])
def test_invalid_action_is_error(empty_env, bad):
    empty_env.reset(seed=0)
    _, r, term, trunc, info = empty_env.step(bad)
    assert info["outcome"] == "error" and term and r == 0.0


def test_reproducible_from_seed():
    a = RescueEnv("village@v2")
    b = RescueEnv("village@v2")
    try:
        oa, ia = a.reset(seed=7)
        ob, ib = b.reset(seed=7)
        assert a.start == b.start
        acts = np.random.default_rng(3).uniform(-0.3, 0.3, (25, 4))
        for act in acts:
            oa, *_ = a.step(act)
            ob, *_ = b.step(act)
            for k in oa:
                assert np.array_equal(np.asarray(oa[k]), np.asarray(ob[k])), k
        a.reset(seed=8)
        b.reset(seed=7, options={"run": 1})
        c_start = dict(b.start)
        b.reset(seed=7)
        assert a.start != b.start and c_start != b.start
    finally:
        a.close()
        b.close()


def test_start_state_distribution(empty_env):
    for s in range(30):
        _, info = empty_env.reset(seed=s)
        st = empty_env.start
        r = math.hypot(st["x"], st["y"])
        assert 225 <= r <= 275 and 5 <= st["z"] <= 25 and 0 <= st["speed"] <= 10
        to_centre = math.atan2(-st["y"], -st["x"])
        err = (st["yaw"] - to_centre + math.pi) % (2 * math.pi) - math.pi
        assert abs(err) <= math.radians(20) + 1e-9
        assert np.allclose(info["rpy"][:2], 0.0)
