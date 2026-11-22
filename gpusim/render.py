"""Batched depth and detector images by GPU ray casting (NVIDIA Warp), with rescuesim's cameras.

Part of the GPU engine originally written by @sckoott (OSTROZKY team); ported to rescuesim.

Camera (SPEC section 4, constants from ``rescuesim.sensors``): pinhole, 102 deg horizontal FOV,
16:9, 0.30 m ahead of the body origin, optical axis tilted 25 deg up; pixel (i, j) samples the
direction of its centre ((i + 0.5)/W, (j + 0.5)/H); depth is z-depth along the camera axis. As in
rescuesim's renders, nothing nearer than the 0.05 m near plane is seen (rays start on it) and the
depth image only looks 41 m far (then clipped to 40 m).

One ray per pixel per drone against the terrain's visual mesh (the same triangles rescuesim
renders) and the drone's own target box (analytic). The depth kernel adds the depth noise on the
GPU (clip to 40 m, then sigma = max(0.03, 0.03 d), not re-clipped) and min-pools it to a 9 x 16
grid; the detector kernel returns the pixel bounding box of the visible target (rays test the box
first; only box pixels check occlusion).
"""
import numpy as np
import warp as wp

from rescuesim import sensors as S
from gpusim.kernels import mesh_ray_front, ray_obb

TAN_H, TAN_V = S.TAN_H, S.TAN_V
MOUNT = np.array([S.CAM_OFFSET_M, 0.0, 0.0], dtype=np.float32)
_c, _s = np.cos(S.CAM_TILT_RAD), np.sin(S.CAM_TILT_RAD)
TILT = np.array([[_c, 0.0, -_s],
                 [0.0, 1.0, 0.0],
                 [_s, 0.0, _c]], dtype=np.float32)   # columns: camera forward, left, up in the body frame
NEAR = S.NEAR
DEPTH_FAR = S.DEPTH_FAR
DET_FAR = S.DET_FAR
DW, DH, CW, CH = S.DEPTH_W, S.DEPTH_H, S.DET_W, S.DET_H
POOL_H, POOL_W = 9, 16                      # 126 / 9 = 14 rows, 224 / 16 = 14 columns per cell
NO_BOX = np.array([10 ** 6, 10 ** 6, -1, -1], dtype=np.int32)


def pixel_rays(width, height):
    """Camera-frame rays (forward component 1, left, up), one per pixel centre, row-major (top row first)."""
    u = (np.arange(width) + 0.5) / width
    v = (np.arange(height) + 0.5) / height
    uu, vv = np.meshgrid(u, v)
    return np.stack([np.ones_like(uu), -(2 * uu - 1) * TAN_H, (1 - 2 * vv) * TAN_V], -1).reshape(-1, 3).astype(np.float32)


@wp.func
def quat_rot(q: wp.quat):
    """World-from-body matrix of an (x, y, z, w) quaternion (rescuesim.geometry.quat_to_matrix)."""
    x = q[0]
    y = q[1]
    z = q[2]
    w = q[3]
    return wp.mat33(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w),
                    2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w),
                    2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y))


@wp.kernel
def pool_reset_kernel(mask: wp.array(dtype=wp.int32), pool: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    if mask[tid // (POOL_H * POOL_W)] != 0:
        pool[tid] = 1.0e9


@wp.kernel
def depth_kernel(pos: wp.array(dtype=wp.vec3), quat: wp.array(dtype=wp.quat), terrain: wp.array(dtype=wp.int32),
                 mask: wp.array(dtype=wp.int32), meshes: wp.array(dtype=wp.uint64),
                 box_c: wp.array(dtype=wp.vec3), box_h: wp.array(dtype=wp.vec3), box_R: wp.array(dtype=wp.mat33),
                 rays: wp.array(dtype=wp.vec3), mount: wp.vec3, tilt: wp.mat33, w: int, h: int,
                 seeds: wp.array(dtype=wp.int32), noise: float, noise_buf: wp.array(dtype=wp.float32), use_buf: int,
                 near: float, far: float, clip: float,
                 depth: wp.array(dtype=wp.float32), pool: wp.array(dtype=wp.float32), write_full: int):
    tid = wp.tid()
    n_pix = w * h
    d = tid // n_pix
    if mask[d] == 0:
        return
    k = tid - d * n_pix
    R = quat_rot(quat[d])
    r_world = (R * tilt) * rays[k]
    length = wp.length(r_world)
    direction = r_world / length
    t0 = near * length                           # start on the near plane (z = near)
    origin = pos[d] + R * mount + direction * t0
    t_max = (far - near) * length
    t = mesh_ray_front(meshes[terrain[d]], origin, direction, t_max)
    t = wp.min(t, ray_obb(origin, direction, box_c[d], box_h[d], box_R[d], t_max))
    z = wp.min(near + t / length, clip)
    if use_buf != 0:                             # standard normals drawn on the host (noise="rescuesim")
        z = z + noise_buf[tid] * wp.max(0.03, 0.03 * z)
    elif noise > 0.0:
        rng = wp.rand_init(seeds[d], k)
        z = z + wp.randn(rng) * wp.max(0.03, 0.03 * z) * noise
    if write_full != 0:
        depth[tid] = z
    row = k // w
    col = k - row * w
    wp.atomic_min(pool, d * (POOL_H * POOL_W) + (row // (h // POOL_H)) * POOL_W + col // (w // POOL_W), z)


@wp.kernel
def detect_kernel(pos: wp.array(dtype=wp.vec3), quat: wp.array(dtype=wp.quat), terrain: wp.array(dtype=wp.int32),
                  meshes: wp.array(dtype=wp.uint64), active: wp.array(dtype=wp.int32),
                  box_c: wp.array(dtype=wp.vec3), box_h: wp.array(dtype=wp.vec3), box_R: wp.array(dtype=wp.mat33),
                  rays: wp.array(dtype=wp.vec3), mount: wp.vec3, tilt: wp.mat33, w: int, h: int,
                  near: float, far: float, bbox: wp.array(dtype=wp.int32)):
    """Pixel box (col0, row0, col1, row1 inclusive) of the visible target."""
    tid = wp.tid()
    n_pix = w * h
    d = tid // n_pix
    if active[d] == 0:
        return
    k = tid - d * n_pix
    R = quat_rot(quat[d])
    r_world = (R * tilt) * rays[k]
    length = wp.length(r_world)
    direction = r_world / length
    t0 = near * length
    origin = pos[d] + R * mount + direction * t0
    t_max = (far - near) * length
    tb = ray_obb(origin, direction, box_c[d], box_h[d], box_R[d], t_max)
    if tb >= t_max:
        return
    if mesh_ray_front(meshes[terrain[d]], origin, direction, tb) < tb:
        return                                    # something closer hides this box pixel
    row = k // w
    col = k - row * w
    wp.atomic_min(bbox, d * 4 + 0, col)
    wp.atomic_min(bbox, d * 4 + 1, row)
    wp.atomic_max(bbox, d * 4 + 2, col)
    wp.atomic_max(bbox, d * 4 + 3, row)


def camera_eye(pos: np.ndarray, R: np.ndarray) -> np.ndarray:
    """(N, 3) camera centres for body positions (N, 3) and world-from-body rotations (N, 3, 3)."""
    return pos + R[:, :, 0] * float(MOUNT[0])


class Renderer:
    """Noise-free renders of arbitrary drone poses on uploaded scenes (validation, debugging).

    ``scenes`` is a ``gpusim.scene.GpuScenes``. ``render(terrain_ids, pos, quat, obb)`` returns the
    z-depth (N, 126, 224), clipped at 40 m, and the detector pixel boxes (N, 4) as
    (x1, y1, x2, y2) pixel edges (x2, y2 exclusive), -1 where the target is not visible. The
    250 m detection range is not applied here."""

    def __init__(self, scenes, device="cuda:0"):
        self.s, self.device = scenes, device
        self.rays_d = wp.array(pixel_rays(DW, DH), dtype=wp.vec3, device=device)
        self.rays_c = wp.array(pixel_rays(CW, CH), dtype=wp.vec3, device=device)
        self.mount, self.tilt = wp.vec3(*MOUNT.tolist()), wp.mat33(*TILT.ravel().tolist())

    def render(self, terrain_ids, pos, quat, obb):
        dev, n = self.device, len(pos)
        arr = lambda a, t: wp.array(np.ascontiguousarray(a, dtype=np.float32 if t is not wp.int32 else np.int32), dtype=t, device=dev)
        P, Q, T = arr(pos, wp.vec3), arr(quat, wp.quat), arr(terrain_ids, wp.int32)
        obb = np.asarray(obb, np.float32)
        bc, bh, bR = arr(obb[:, :3], wp.vec3), arr(obb[:, 3:6], wp.vec3), arr(obb[:, 6:].reshape(-1, 3, 3), wp.mat33)
        ones = arr(np.ones(n), wp.int32)
        depth = wp.zeros(n * DW * DH, dtype=wp.float32, device=dev)
        pool = wp.zeros(n * POOL_H * POOL_W, dtype=wp.float32, device=dev)
        wp.launch(depth_kernel, dim=n * DW * DH, device=dev,
                  inputs=[P, Q, T, ones, self.s.meshes, bc, bh, bR, self.rays_d, self.mount, self.tilt, DW, DH,
                          ones, 0.0, depth, 0, NEAR, DEPTH_FAR, S.DEPTH_MAX_M, depth, pool, 1])
        bbox = wp.array(np.tile(NO_BOX, n), dtype=wp.int32, device=dev)
        wp.launch(detect_kernel, dim=n * CW * CH, device=dev,
                  inputs=[P, Q, T, self.s.meshes, ones, bc, bh, bR, self.rays_c, self.mount, self.tilt, CW, CH,
                          NEAR, DET_FAR, bbox])
        b = bbox.numpy().reshape(n, 4)
        seen = b[:, 2] >= 0
        out = np.where(seen[:, None], b + np.array([0, 0, 1, 1]), -1)
        return depth.numpy().reshape(n, DH, DW), out
