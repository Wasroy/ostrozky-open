import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from rescuesim import RescueEnv  # noqa: E402


@pytest.fixture(scope="session")
def root():
    return ROOT


@pytest.fixture(scope="module")
def empty_env():
    """Flat world without props, cameras off (fast dynamics tests)."""
    env = RescueEnv("empty@v1", render_depth=False, render_detector=False)
    yield env
    env.close()


@pytest.fixture(scope="module")
def empty_env_cams():
    env = RescueEnv("empty@v1")
    yield env
    env.close()


def fly(env, action, steps, **start):
    opts = {}
    if "pos" in start:
        opts["start_pos"] = start["pos"]
    if "yaw" in start:
        opts["start_yaw"] = start["yaw"]
    if "speed" in start:
        opts["start_speed"] = start["speed"]
    obs, info = env.reset(seed=start.get("seed", 0), options=opts)
    hist = [(obs, info)]
    for _ in range(steps):
        obs, r, term, trunc, info = env.step(action)
        hist.append((obs, info))
        if term or trunc:
            break
    return hist
