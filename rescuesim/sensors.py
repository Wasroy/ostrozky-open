"""Sensor models: camera geometry, synthetic detector, depth noise, IMU, VIO."""
from __future__ import annotations

import math

import numpy as np

# Camera geometry shared by the depth image and the detector.
H_FOV_DEG = 102.0
ASPECT = 16.0 / 9.0
TAN_H = math.tan(math.radians(H_FOV_DEG / 2))
TAN_V = TAN_H / ASPECT
V_FOV_DEG = math.degrees(2 * math.atan(TAN_V))      # about 69.6 deg
CAM_OFFSET_M = 0.30
CAM_TILT_RAD = math.radians(25.0)

DEPTH_W, DEPTH_H = 224, 126
DET_W, DET_H = 160, 90
DEPTH_MAX_M = 40.0
DETECT_RANGE_M = 250.0
MISS_PROB = 0.05

NEAR = 0.05
DEPTH_FAR = 41.0          # beyond the 40 m clip: anything not hit reads > 40 and clips to 40
DET_FAR = 265.0

GYRO_SIGMA = 0.002
ACCEL_SIGMA = 0.03
VIO_ALT_SIGMA = 0.05
VIO_XY_BASE = 0.003
VIO_XY_PROP = 0.01


def camera_frame(pos, R):
    """Return (eye, forward, up, right) in world coordinates for body pose (pos, R)."""
    c, s = math.cos(CAM_TILT_RAD), math.sin(CAM_TILT_RAD)
    eye = pos + R[:, 0] * CAM_OFFSET_M
    fwd = R @ np.array([c, 0.0, s])
    up = R @ np.array([-s, 0.0, c])
    right = np.cross(fwd, up)
    return eye, fwd, up, right


def aimed_camera(eye, target):
    """(eye, forward, up, right) for a camera at ``eye`` looking at ``target`` (roll-free)."""
    eye = np.asarray(eye, float)
    fwd = np.asarray(target, float) - eye
    fwd = fwd / np.linalg.norm(fwd)
    ref = np.array([0.0, 0.0, 1.0]) if abs(fwd[2]) < 0.95 else np.array([1.0, 0.0, 0.0])
    right = np.cross(fwd, ref)
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return eye, fwd, up, right


def frustum_matrix(l, r, b, t, n, f) -> list:
    """OpenGL glFrustum matrix, column-major, as PyBullet expects."""
    return [
        2 * n / (r - l), 0.0, 0.0, 0.0,
        0.0, 2 * n / (t - b), 0.0, 0.0,
        (r + l) / (r - l), (t + b) / (t - b), -(f + n) / (f - n), -1.0,
        0.0, 0.0, -2 * f * n / (f - n), 0.0,
    ]


# TinyRenderer samples each pixel at a corner of its cell (left edge in x, lower
# edge in y, measured empirically), not at its centre. The projection window is
# therefore shifted by half a pixel so that pixel (i, j) samples the direction of
# its centre ((i + 0.5) / W, (j + 0.5) / H), as the spec requires.
RASTER_SHIFT_X = 0.5        # pixels, towards +x (right)
RASTER_SHIFT_Y = 0.5        # pixels, towards the top of the image


def window_tangents(i0, i1, j0, j1, W, H, raster_fix: bool = True):
    """Tangent-space bounds (left, right, bottom, top) of the projection that renders
    the pixel window [i0,i1) x [j0,j1) of the W x H image."""
    sx = RASTER_SHIFT_X if raster_fix else 0.0
    sy = RASTER_SHIFT_Y if raster_fix else 0.0
    L = -TAN_H + 2 * TAN_H * (i0 + sx) / W
    Rr = -TAN_H + 2 * TAN_H * (i1 + sx) / W
    T = TAN_V - 2 * TAN_V * (j0 - sy) / H
    B = TAN_V - 2 * TAN_V * (j1 - sy) / H
    return L, Rr, B, T


def projection_for_window(i0, i1, j0, j1, W, H, near, far) -> list:
    L, Rr, B, T = window_tangents(i0, i1, j0, j1, W, H)
    return frustum_matrix(L * near, Rr * near, B * near, T * near, near, far)


def frustum_planes(eye, fwd, up, right, tangents, near, far) -> np.ndarray:
    """Inward plane equations (n, d) with n.x + d >= 0 inside; shape (6, 4)."""
    L, Rr, B, T = tangents
    normals_cam = np.array([
        [1.0, 0.0, -L],
        [-1.0, 0.0, Rr],
        [0.0, 1.0, -B],
        [0.0, -1.0, T],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ])
    basis = np.stack([right, up, fwd])            # rows: camera x, y, z axes in world
    nw = normals_cam @ basis
    d = -(nw @ eye)
    d[4] -= near
    d[5] += far
    return np.c_[nw, d]


def aabb_in_frustum(planes: np.ndarray, centers: np.ndarray, halves: np.ndarray) -> np.ndarray:
    n = planes[:, :3]
    s = centers @ n.T + halves @ np.abs(n).T + planes[:, 3]
    return np.all(s >= 0.0, axis=1)


def project_points(points, eye, fwd, up, right, W, H):
    """Pixel coordinates (u right, v down, continuous) and forward depth of world points."""
    d = points - eye
    xc, yc, zc = d @ right, d @ up, d @ fwd
    with np.errstate(divide="ignore", invalid="ignore"):
        u = (xc / zc / TAN_H + 1.0) * 0.5 * W
        v = (1.0 - yc / zc / TAN_V) * 0.5 * H
    return u, v, zc


def depth_from_buffer(buf: np.ndarray, near: float, far: float) -> np.ndarray:
    """Convert an OpenGL depth buffer to distance along the camera axis (z-depth)."""
    return far * near / (far - (far - near) * buf)


def noisy_depth(z: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Clip to [0, 40] m first, then add Gaussian noise sigma = max(0.03, 0.03 d); no re-clip."""
    d = np.clip(z, 0.0, DEPTH_MAX_M)
    sigma = np.maximum(0.03, 0.03 * d)
    return (d + rng.standard_normal(d.shape) * sigma).astype(np.float32)


def detect_from_mask(mask: np.ndarray | None, offset, distance_m: float, rng: np.random.Generator,
                     W: int = DET_W, H: int = DET_H):
    """Synthetic detector from a ground-truth target mask.

    ``mask`` is a boolean array covering the pixel window starting at ``offset``
    (column, row) of the W x H detection image; ``None`` means nothing visible.
    Returns (detected, box[4] float32).
    """
    zero = np.zeros(4, np.float32)
    if mask is None or distance_m > DETECT_RANGE_M:
        return 0, zero
    cols = np.flatnonzero(mask.any(axis=0))
    if cols.size == 0:
        return 0, zero
    rows = np.flatnonzero(mask.any(axis=1))
    if rng.random() < MISS_PROB:
        return 0, zero
    x1, x2 = offset[0] + cols[0], offset[0] + cols[-1] + 1.0
    y1, y2 = offset[1] + rows[0], offset[1] + rows[-1] + 1.0
    sx = max(0.5, 0.01 * (x2 - x1))
    sy = max(0.5, 0.01 * (y2 - y1))
    j = rng.standard_normal(4)
    x1, x2 = x1 + j[0] * sx, x2 + j[1] * sx
    y1, y2 = y1 + j[2] * sy, y2 + j[3] * sy
    bx = np.clip(np.array([min(x1, x2) / W, min(y1, y2) / H, max(x1, x2) / W, max(y1, y2) / H]), 0.0, 1.0)
    return 1, bx.astype(np.float32)


class IMU:
    def __init__(self, rng: np.random.Generator, dt: float):
        self.rng = rng
        self.dt = dt
        self.prev_vel = None

    def reset(self, vel):
        self.prev_vel = np.array(vel, float)

    def read(self, vel, R, omega, first: bool = False):
        acc = np.zeros(3) if first else (vel - self.prev_vel) / self.dt
        self.prev_vel = np.array(vel, float)
        specific = R.T @ (acc + np.array([0.0, 0.0, 9.81]))
        accel = specific + self.rng.normal(0.0, ACCEL_SIGMA, 3)
        gyro = np.asarray(omega) + self.rng.normal(0.0, GYRO_SIGMA, 3)
        return gyro.astype(np.float32), accel.astype(np.float32)


class VIO:
    """Drifting horizontal position, drift-free noisy altitude."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.xy = np.zeros(2)
        self.prev = np.zeros(2)

    def reset(self, pos):
        self.xy = np.array(pos[:2], float)
        self.prev = self.xy.copy()

    def read(self, pos, update: bool = True):
        if update:
            d = np.asarray(pos[:2], float) - self.prev
            sigma = VIO_XY_BASE + VIO_XY_PROP * float(np.hypot(d[0], d[1]))
            self.xy = self.xy + d + self.rng.normal(0.0, sigma, 2)
            self.prev = np.array(pos[:2], float)
        alt = pos[2] + self.rng.normal(0.0, VIO_ALT_SIGMA)
        return np.array([self.xy[0], self.xy[1], alt], np.float32)
