"""Batched NumPy version of rescuesim's quadrotor model and angle-mode controller.

Same equations, numbers and order of operations as ``rescuesim.dynamics.Quadrotor`` (SPEC section 3),
vectorised over N drones in float64. It is the reference for the Warp step kernel in ``gpusim.fast``
(tests and ``gpusim.validate`` compare them) and a CPU fallback for dynamics-only work without Warp.

State per drone: pos (3), vel (3) world frame; quat (x, y, z, w) world-from-body; omega (3) body
rates (FLU).
"""
from __future__ import annotations

import numpy as np

from rescuesim.dynamics import GRAVITY, QuadParams

PHYSICS_HZ, CONTROL_HZ = 240, 30
SUBSTEPS = PHYSICS_HZ // CONTROL_HZ
DT_PHYS = 1.0 / PHYSICS_HZ
DT_CTRL = 1.0 / CONTROL_HZ

# Motor layout (FLU): front-left, front-right, rear-right, rear-left (as rescuesim.dynamics)
MX = np.array([1.0, 1.0, -1.0, -1.0])
MY = np.array([1.0, -1.0, -1.0, 1.0])
SPIN = np.array([1.0, -1.0, 1.0, -1.0])


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """(N, 4) xyzw -> (N, 3, 3) world-from-body."""
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def quat_from_euler(roll, pitch, yaw) -> np.ndarray:
    cr, sr = np.cos(np.asarray(roll) / 2), np.sin(np.asarray(roll) / 2)
    cp, sp = np.cos(np.asarray(pitch) / 2), np.sin(np.asarray(pitch) / 2)
    cy, sy = np.cos(np.asarray(yaw) / 2), np.sin(np.asarray(yaw) / 2)
    return np.stack([sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
                     cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy], -1)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz], -1)


def euler_from_matrix(R: np.ndarray) -> np.ndarray:
    roll = np.arctan2(R[..., 2, 1], R[..., 2, 2])
    pitch = np.arcsin(np.clip(-R[..., 2, 0], -1.0, 1.0))
    yaw = np.arctan2(R[..., 1, 0], R[..., 0, 0])
    return np.stack([roll, pitch, yaw], -1)


def allocate(p: QuadParams, thrust: np.ndarray, torque: np.ndarray) -> np.ndarray:
    """(N,) collective, (N, 3) torque -> (N, 4) motor thrusts; rescuesim.dynamics.allocate batched."""
    fmin, fmax, a = p.motor_min, p.motor_max, p.arm_xy
    rp = (torque[:, :1] * MY * a - torque[:, 1:2] * MX * a) / (4 * a * a)
    yw = torque[:, 2:3] * SPIN / (4 * p.yaw_moment_coeff)
    spread = rp.max(1) - rp.min(1)
    big = spread > fmax - fmin
    rp = np.where(big[:, None], rp * ((fmax - fmin) / np.where(big, spread, 1.0))[:, None], rp)
    base = thrust / 4.0
    lo, hi = fmin - rp.min(1), fmax - rp.max(1)
    base = np.minimum(np.maximum(base, lo), hi)
    f = base[:, None] + rp
    with np.errstate(divide="ignore", invalid="ignore"):
        up = np.where(yw > 1e-12, (fmax - f) / yw, np.inf)
        dn = np.where(yw < -1e-12, (fmin - f) / yw, np.inf)
    s = np.maximum(0.0, np.minimum(1.0, np.minimum(up, dn).min(1)))
    return np.clip(f + s[:, None] * yw, fmin, fmax)


class BatchQuad:
    """N quadrotors with rescuesim's dynamics and controller, stepped together."""

    def __init__(self, n: int, params: QuadParams | None = None, wind=(0.0, 0.0, 0.0), thrust_scale=1.0):
        self.n = n
        self.p = params or QuadParams()
        self.I = np.asarray(self.p.inertia, float)
        self.k_rate = np.asarray(self.p.k_rate, float)
        self.wind = np.broadcast_to(np.asarray(wind, float), (n, 3)).copy()
        self.thrust_scale = np.broadcast_to(np.asarray(thrust_scale, float), (n,)).copy()
        self.pos = np.zeros((n, 3))
        self.vel = np.zeros((n, 3))
        self.quat = np.tile([0.0, 0.0, 0.0, 1.0], (n, 1))
        self.omega = np.zeros((n, 3))

    def reset(self, idx, pos, yaw, speed):
        idx = np.atleast_1d(idx)
        yaw = np.broadcast_to(np.asarray(yaw, float), idx.shape)
        speed = np.broadcast_to(np.asarray(speed, float), idx.shape)
        self.pos[idx] = pos
        self.vel[idx] = np.c_[np.cos(yaw), np.sin(yaw), np.zeros_like(yaw)] * speed[:, None]
        self.quat[idx] = quat_from_euler(np.zeros_like(yaw), np.zeros_like(yaw), yaw)
        self.omega[idx] = 0.0

    def control(self, action: np.ndarray, sel=slice(None)) -> np.ndarray:
        p = self.p
        q, w = self.quat[sel], self.omega[sel]
        R = quat_to_matrix(q)
        yaw_now = np.arctan2(R[:, 1, 0], R[:, 0, 0])
        q_des = quat_from_euler(action[:, 0] * p.max_tilt_rad, action[:, 1] * p.max_tilt_rad, yaw_now)
        qc = q * np.array([-1.0, -1.0, -1.0, 1.0])
        qe = quat_mul(qc, q_des)
        qe = np.where(qe[:, 3:4] < 0, -qe, qe)
        vn = np.linalg.norm(qe[:, :3], axis=1)
        ang = 2.0 * np.arctan2(vn, qe[:, 3])
        err = np.where((vn > 1e-12)[:, None], qe[:, :3] * (ang / np.where(vn > 1e-12, vn, 1.0))[:, None], 0.0)
        rate_sp = np.c_[p.k_att * err[:, 0], p.k_att * err[:, 1], -p.max_yaw_rate_rad_s * action[:, 2]]
        alpha = self.k_rate * (rate_sp - w)
        torque = self.I * alpha + np.cross(w, self.I * w)
        thrust = p.hover_thrust * (1.0 + action[:, 3])
        return allocate(p, thrust, torque)

    def substep(self, action: np.ndarray, dt: float = DT_PHYS, sel=slice(None)):
        p, a = self.p, self.p.arm_xy
        f = self.control(action, sel) * self.thrust_scale[sel][:, None]
        thrust = f.sum(1)
        tau = np.c_[a * (f @ MY), -a * (f @ MX), p.yaw_moment_coeff * (f @ SPIN)]
        R = quat_to_matrix(self.quat[sel])
        v_air = self.vel[sel] - self.wind[sel]
        drag = -p.mass * p.drag_k * np.linalg.norm(v_air, axis=1, keepdims=True) * v_air
        acc = (R[:, :, 2] * thrust[:, None] + drag) / p.mass
        acc[:, 2] -= GRAVITY
        vel = self.vel[sel] + acc * dt
        self.vel[sel] = vel
        self.pos[sel] = self.pos[sel] + vel * dt
        w = self.omega[sel]
        wdot = (tau - np.cross(w, self.I * w)) / self.I
        w = w + wdot * dt
        self.omega[sel] = w
        wn = np.linalg.norm(w, axis=1)
        h = 0.5 * wn * dt
        s = np.where(wn > 1e-12, np.sin(h) / np.where(wn > 1e-12, wn, 1.0), 0.0)
        dq = np.c_[w * s[:, None], np.cos(h)]
        q = quat_mul(self.quat[sel], dq)
        q = q / np.linalg.norm(q, axis=1, keepdims=True)
        self.quat[sel] = np.where((wn > 1e-12)[:, None], q, self.quat[sel])

    def step(self, action: np.ndarray, substeps: int = SUBSTEPS):
        """One control step (8 physics sub-steps at 240 Hz) for every drone."""
        action = np.asarray(action, float).reshape(self.n, 4)
        for _ in range(substeps):
            self.substep(action)

    @property
    def rpy(self) -> np.ndarray:
        return euler_from_matrix(quat_to_matrix(self.quat))


__all__ = ["BatchQuad", "allocate", "quat_to_matrix", "quat_from_euler", "quat_mul", "euler_from_matrix",
           "SUBSTEPS", "DT_PHYS", "DT_CTRL", "GRAVITY", "QuadParams"]
