import math

import numpy as np
import pytest

from conftest import fly
from rescuesim.dynamics import QuadParams, allocate, motor_wrench

START = dict(pos=(0.0, -200.0, 30.0), yaw=0.0, speed=0.0)


def test_pitch_positive_accelerates_forward(empty_env):
    h = fly(empty_env, [0, 0.5, 0, 0], 30, **START)
    info = h[-1][1]
    assert info["vel"][0] > 3.0                     # forward = +x for yaw 0
    assert abs(info["vel"][1]) < 0.05
    assert info["rpy"][1] > math.radians(25)        # pitch+ (nose down)


def test_pitch_forward_follows_heading(empty_env):
    h = fly(empty_env, [0, 0.5, 0, 0], 30, pos=(0, -200, 30), yaw=math.pi / 2, speed=0.0)
    v = h[-1][1]["vel"]
    assert v[1] > 3.0 and abs(v[0]) < 0.05          # heading +y -> moves along +y


def test_roll_positive_accelerates_right(empty_env):
    h = fly(empty_env, [0.5, 0, 0, 0], 30, **START)
    v = h[-1][1]["vel"]
    assert v[1] < -3.0                              # right of +x heading is -y
    assert h[-1][1]["rpy"][0] > math.radians(25)


def test_yaw_positive_turns_clockwise(empty_env):
    h = fly(empty_env, [0, 0, 0.5, 0], 30, **START)
    yaw = h[-1][1]["rpy"][2]
    assert -math.radians(110) < yaw < -math.radians(70)     # ~100 deg/s clockwise
    gyro_z = np.mean([o["gyro_rad_s"][2] for o, _ in h[10:]])
    assert gyro_z == pytest.approx(-math.radians(100), rel=0.1)


def test_throttle_zero_holds_height_when_level(empty_env):
    h = fly(empty_env, [0, 0, 0, 0], 90, **START)
    z = [i["pos"][2] for _, i in h]
    assert max(abs(np.array(z) - 30.0)) < 0.02


def test_tilt_at_throttle_zero_loses_height(empty_env):
    h = fly(empty_env, [0, 0.5, 0, 0], 60, **START)
    assert h[-1][1]["pos"][2] < 29.0


def test_hover_accelerometer_reads_gravity(empty_env):
    h = fly(empty_env, [0, 0, 0, 0], 60, **START)
    acc = np.array([o["accel_m_s2"] for o, _ in h])
    assert np.allclose(acc.mean(0), [0, 0, 9.81], atol=0.02)
    assert 0.02 < acc[:, 0].std() < 0.045           # N(0, 0.03) per axis
    gyro = np.array([o["gyro_rad_s"] for o, _ in h])
    assert 0.001 < gyro[:, 0].std() < 0.003


def test_expected_feel_and_top_speed(empty_env):
    h = fly(empty_env, [0, 0.5, 0, 0], 45, **START)
    t30 = next(i["t"] for _, i in h if i["rpy"][1] >= math.radians(29))
    t4 = next(i["t"] for _, i in h if i["vel"][0] >= 4.0)
    assert t30 < 0.7 and 0.7 < t4 < 1.3
    h = fly(empty_env, [0, 1, 0, 1], 25 * 30, pos=(-300, 0, 60), yaw=0.0, speed=0.0)
    v = h[-1][1]["vel"]
    assert 31.5 < math.hypot(v[0], v[1]) < 35.0
    assert abs(v[2]) < 0.3                          # level flight at 60 deg needs throttle +1


def test_allocation_priorities():
    p = QuadParams()
    # hover with a large yaw request: thrust and roll/pitch kept, yaw reduced
    f = allocate(p, p.hover_thrust, np.array([0.0, 0.0, 5.0]))
    T, tau = motor_wrench(p, f)
    assert abs(T - p.hover_thrust) < 1e-6
    assert 0 < tau[2] < 5.0
    assert np.all(f >= p.motor_min - 1e-9) and np.all(f <= p.motor_max + 1e-9)
    # zero-thrust request is clipped to idle
    f = allocate(p, 0.0, np.zeros(3))
    assert np.allclose(f, p.motor_min)
    # roll torque preserved at low thrust by shifting the collective
    f = allocate(p, 0.0, np.array([0.5, 0.0, 0.0]))
    assert abs(motor_wrench(p, f)[1][0] - 0.5) < 1e-6
