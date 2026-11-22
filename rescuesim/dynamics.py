"""Quadrotor rigid-body model and the fixed angle-mode flight controller.

Everything here is plain NumPy; PyBullet is only used elsewhere for collision
queries and rendering. The body frame is FLU (x forward, y left, z up).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .geometry import quat_conj, quat_from_euler, quat_mul, quat_normalize, quat_to_matrix

GRAVITY = 9.81


@dataclass
class QuadParams:
    mass: float = 1.68                      # kg
    thrust_to_weight: float = 5.64
    arm_xy: float = 0.078                   # motor x/y offset (m): ~220 mm motor-to-motor diagonal
    inertia: tuple = (0.0030, 0.0030, 0.0055)   # kg m^2 (5-inch class, heavy battery)
    yaw_moment_coeff: float = 0.016         # reaction torque per newton of thrust (m)
    drag_k: float = 0.0154                  # quadratic drag, F = -m k |v| v (1/m)
    idle_fraction: float = 0.05             # minimum motor command (fraction of max)
    max_tilt_rad: float = math.radians(60.0)
    max_yaw_rate_rad_s: float = math.radians(200.0)
    k_att: float = 6.0                      # attitude P gain (1/s)
    k_rate: tuple = (30.0, 30.0, 10.0)      # rate P gains (1/s)
    collision_half_extents: tuple = (0.14, 0.14, 0.04)

    @property
    def hover_thrust(self) -> float:
        return self.mass * GRAVITY

    @property
    def motor_max(self) -> float:
        return self.thrust_to_weight * self.hover_thrust / 4.0

    @property
    def motor_min(self) -> float:
        return self.idle_fraction * self.motor_max


# Motor layout (FLU): front-left, front-right, rear-right, rear-left.
_MX = np.array([1.0, 1.0, -1.0, -1.0])
_MY = np.array([1.0, -1.0, -1.0, 1.0])
_SPIN = np.array([1.0, -1.0, 1.0, -1.0])   # sign of the reaction torque about +z


@dataclass
class QuadState:
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    quat: tuple = (0.0, 0.0, 0.0, 1.0)      # world-from-body
    omega: np.ndarray = field(default_factory=lambda: np.zeros(3))   # body rates (FLU)

    def copy(self) -> "QuadState":
        return QuadState(self.pos.copy(), self.vel.copy(), tuple(self.quat), self.omega.copy())


def allocate(params: QuadParams, thrust: float, torque: np.ndarray) -> np.ndarray:
    """Motor thrusts for a collective thrust and body torque request.

    Priorities: roll/pitch torque is kept (scaled down only if it alone exceeds
    the motor range), the collective is shifted as little as possible to make
    room for it, and yaw only gets the authority that is left.
    """
    fmin, fmax = params.motor_min, params.motor_max
    a = params.arm_xy
    rp = (torque[0] * _MY * a - torque[1] * _MX * a) / (4 * a * a)
    yw = torque[2] * _SPIN / (4 * params.yaw_moment_coeff)
    spread = rp.max() - rp.min()
    if spread > fmax - fmin:
        rp = rp * ((fmax - fmin) / spread)
    base = thrust / 4.0
    lo, hi = fmin - rp.min(), fmax - rp.max()
    base = min(max(base, lo), hi)
    f = base + rp
    s = 1.0
    for i in range(4):
        if yw[i] > 1e-12:
            s = min(s, (fmax - f[i]) / yw[i])
        elif yw[i] < -1e-12:
            s = min(s, (fmin - f[i]) / yw[i])
    s = max(0.0, s)
    return np.clip(f + s * yw, fmin, fmax)


def motor_wrench(params: QuadParams, f: np.ndarray) -> tuple:
    a = params.arm_xy
    thrust = float(f.sum())
    tau = np.array([
        a * float(np.dot(_MY, f)),
        -a * float(np.dot(_MX, f)),
        params.yaw_moment_coeff * float(np.dot(_SPIN, f)),
    ])
    return thrust, tau


class Quadrotor:
    """Rigid body + angle-mode controller, integrated at the physics rate."""

    def __init__(self, params: QuadParams | None = None, wind=(0.0, 0.0, 0.0), thrust_scale: float = 1.0):
        self.p = params or QuadParams()
        self.wind = np.asarray(wind, float)
        self.thrust_scale = float(thrust_scale)
        self.I = np.asarray(self.p.inertia, float)
        self.k_rate = np.asarray(self.p.k_rate, float)
        self.state = QuadState()
        self.last_motor = np.zeros(4)

    def reset(self, pos, yaw: float, speed: float = 0.0):
        v = np.array([math.cos(yaw), math.sin(yaw), 0.0]) * speed
        self.state = QuadState(np.asarray(pos, float).copy(), v, quat_from_euler(0.0, 0.0, yaw), np.zeros(3))

    def control(self, action) -> np.ndarray:
        """Angle-mode cascade: stick -> attitude -> rates -> torque -> motors."""
        roll_c, pitch_c, yaw_c, thr_c = action
        st = self.state
        R = quat_to_matrix(st.quat)
        yaw_now = math.atan2(R[1, 0], R[0, 0])
        q_des = quat_from_euler(roll_c * self.p.max_tilt_rad, pitch_c * self.p.max_tilt_rad, yaw_now)
        qe = quat_mul(quat_conj(st.quat), q_des)          # body-frame error rotation
        if qe[3] < 0:
            qe = (-qe[0], -qe[1], -qe[2], -qe[3])
        vn = math.sqrt(qe[0] ** 2 + qe[1] ** 2 + qe[2] ** 2)
        if vn > 1e-12:
            ang = 2.0 * math.atan2(vn, qe[3])
            err = np.array(qe[:3]) * (ang / vn)
        else:
            err = np.zeros(3)
        rate_sp = np.array([
            self.p.k_att * err[0],
            self.p.k_att * err[1],
            -self.p.max_yaw_rate_rad_s * yaw_c,
        ])
        alpha = self.k_rate * (rate_sp - st.omega)
        w = st.omega
        torque = self.I * alpha + np.cross(w, self.I * w)
        thrust = self.p.hover_thrust * (1.0 + thr_c)
        return allocate(self.p, thrust, torque)

    def substep(self, action, dt: float):
        f = self.control(action) * self.thrust_scale
        self.last_motor = f
        thrust, tau = motor_wrench(self.p, f)
        st = self.state
        R = quat_to_matrix(st.quat)
        v_air = st.vel - self.wind
        drag = -self.p.mass * self.p.drag_k * np.linalg.norm(v_air) * v_air
        acc = (R[:, 2] * thrust + drag) / self.p.mass
        acc[2] -= GRAVITY
        st.vel = st.vel + acc * dt
        st.pos = st.pos + st.vel * dt
        w = st.omega
        wdot = (tau - np.cross(w, self.I * w)) / self.I
        st.omega = w + wdot * dt
        # body-frame quaternion increment
        wn = float(np.linalg.norm(st.omega))
        if wn > 1e-12:
            h = 0.5 * wn * dt
            s = math.sin(h) / wn
            dq = (st.omega[0] * s, st.omega[1] * s, st.omega[2] * s, math.cos(h))
            st.quat = quat_normalize(quat_mul(st.quat, dq))
