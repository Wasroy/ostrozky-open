"""Small rotation and mesh helpers (no PyBullet dependency).

Conventions: quaternions are (x, y, z, w) like PyBullet. Euler angles are
(roll, pitch, yaw) with R = Rz(yaw) @ Ry(pitch) @ Rx(roll), right-handed
rotations about the world axes. For the FLU body used here that means
roll+ = right wing down, pitch+ = nose down, yaw+ = counter-clockwise from above.
"""
from __future__ import annotations

import math

import numpy as np

IDENTITY_QUAT = (0.0, 0.0, 0.0, 1.0)


def quat_from_euler(roll: float, pitch: float, yaw: float) -> tuple:
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def quat_mul(a, b) -> tuple:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def quat_conj(q) -> tuple:
    return (-q[0], -q[1], -q[2], q[3])


def quat_normalize(q) -> tuple:
    n = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    return (q[0] / n, q[1] / n, q[2] / n, q[3] / n)


def quat_from_axis_angle(axis, angle: float) -> tuple:
    ax = np.asarray(axis, float)
    ax = ax / np.linalg.norm(ax)
    s = math.sin(angle / 2)
    return (ax[0] * s, ax[1] * s, ax[2] * s, math.cos(angle / 2))


def quat_to_matrix(q) -> np.ndarray:
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def euler_from_matrix(R: np.ndarray) -> tuple:
    roll = math.atan2(R[2, 1], R[2, 2])
    pitch = math.asin(max(-1.0, min(1.0, -R[2, 0])))
    yaw = math.atan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw


def yaw_quat(yaw: float) -> tuple:
    return (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))


# ----------------------------------------------------------------------------
# Low-poly meshes. Each returns (vertices (N,3), triangles (M,3)) in the local
# frame of the primitive. Round shapes are *circumscribing* hulls of the
# collision primitive, so the rendered surface always contains the collision
# surface (depth never shows free space where there is none).
# ----------------------------------------------------------------------------

_BOX_V = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
_BOX_F = np.array(
    [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
     [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]]
)

_T = (1 + 5 ** 0.5) / 2
_ICO_V = np.array(
    [[-1, _T, 0], [1, _T, 0], [-1, -_T, 0], [1, -_T, 0], [0, -1, _T], [0, 1, _T],
     [0, -1, -_T], [0, 1, -_T], [_T, 0, -1], [_T, 0, 1], [-_T, 0, -1], [-_T, 0, 1]], float
)
_ICO_V /= np.linalg.norm(_ICO_V[0])
_ICO_F = np.array(
    [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4],
     [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8],
     [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]]
)
# inradius / circumradius of the regular icosahedron
_ICO_IN = 0.7946544722917661


def box_mesh(half) -> tuple:
    return _BOX_V * np.asarray(half, float), _BOX_F


def sphere_mesh(r: float) -> tuple:
    return _ICO_V * (r / _ICO_IN), _ICO_F


def prism_mesh(r: float, length: float, n: int = 8, caps: bool = True) -> tuple:
    """n-sided prism along local z whose faces are tangent to a cylinder of radius r."""
    rc = r / math.cos(math.pi / n)
    a = np.arange(n) * 2 * math.pi / n + math.pi / n
    ring = np.stack([rc * np.cos(a), rc * np.sin(a)], 1)
    h = length / 2
    v = np.concatenate([np.c_[ring, np.full(n, -h)], np.c_[ring, np.full(n, h)]])
    f = []
    for i in range(n):
        j = (i + 1) % n
        f += [[i, j, n + j], [i, n + j, n + i]]
    if caps:
        for i in range(1, n - 1):
            f.append([0, i + 1, i])
            f.append([n, n + i, n + i + 1])
    return v, np.array(f)


def quad_mesh(hx: float, hy: float) -> tuple:
    v = np.array([[-hx, -hy, 0], [hx, -hy, 0], [hx, hy, 0], [-hx, hy, 0]], float)
    return v, np.array([[0, 1, 2], [0, 2, 3]])


def local_half_extents(shape: str, size) -> np.ndarray:
    """Half extents of the local bounding box of a primitive."""
    if shape == "box":
        return np.asarray(size, float)
    if shape == "sphere":
        return np.full(3, size[0])
    if shape == "cylinder":
        return np.array([size[0], size[0], size[1] / 2])
    if shape == "capsule":
        return np.array([size[0], size[0], size[1] / 2 + size[0]])
    if shape == "quad":
        return np.array([size[0], size[1], 0.0])
    raise ValueError(shape)


def world_aabb(shape: str, size, pos, quat) -> tuple:
    R = quat_to_matrix(quat)
    h = np.abs(R) @ local_half_extents(shape, size)
    c = np.asarray(pos, float)
    return c - h, c + h


def point_box_distance(point, center, half, yaw: float) -> float:
    """Distance from a point to the surface of a yaw-rotated box (0 if inside)."""
    d = np.asarray(point, float) - np.asarray(center, float)
    c, s = math.cos(yaw), math.sin(yaw)
    local = np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1], d[2]])
    q = np.abs(local) - np.asarray(half, float)
    outside = np.linalg.norm(np.maximum(q, 0.0))
    return float(outside)


def box_corners(center, half, yaw: float) -> np.ndarray:
    R = quat_to_matrix(yaw_quat(yaw))
    return np.asarray(center, float) + (_BOX_V * np.asarray(half, float)) @ R.T
