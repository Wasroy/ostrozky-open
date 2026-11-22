"""Warp geometry functions: ray vs per-drone target box, and drone contact queries.

Part of the GPU engine originally written by @sckoott (OSTROZKY team); ported to rescuesim.

The props of a terrain are shared by all of its poses; each drone also has its own target box (an
oriented box: centre, half extents, world-from-box rotation), handled analytically.

Contacts reproduce rescuesim's collision test (``rescuesim.world.CollisionWorld.drone_contacts``:
the drone is a 0.28 x 0.28 x 0.08 m box, a body is touched when the closest distance is <= 1 mm).
Broad phase: a Warp BVH over the bounding boxes of the terrain's collision primitives. Narrow phase,
against the exact rescuesim primitives (sharp edges, as PyBullet reports them):
  box      separating-axis test between the two oriented boxes (exact);
  sphere   closest point of the drone box to the centre (exact);
  capsule  segment-to-box distance by golden-section search along the axis (exact, convex);
  cylinder exact signed distance of points sampled on the drone box surface (2.5 cm grid, corners
           and edges included), plus points of the rim arcs facing the drone (error < ~1 cm in rare
           rim-into-face configurations).
"""
import numpy as np
import warp as wp

from rescuesim.dynamics import QuadParams
from rescuesim.scene import GROUND_HALF_M

DRONE_HALF = np.asarray(QuadParams().collision_half_extents, np.float32)
CONTACT_TOL_M = 1.0e-3          # rescuesim.world.CONTACT_TOL_M
GROUND_HALF = wp.constant(float(GROUND_HALF_M))
SHAPE_BOX, SHAPE_SPHERE, SHAPE_CYLINDER, SHAPE_CAPSULE = 0, 1, 2, 3


def drone_surface_points(spacing: float = 0.025) -> np.ndarray:
    """Points on the surface of the drone collision box (body frame), corners and edges included."""
    hx, hy, hz = DRONE_HALF.astype(float)
    ax = [np.linspace(-h, h, max(2, int(np.ceil(2 * h / spacing)) + 1)) for h in (hx, hy, hz)]
    X, Y, Z = np.meshgrid(*ax, indexing="ij")
    P = np.stack([X, Y, Z], -1).reshape(-1, 3)
    on = np.isclose(np.abs(P), [hx, hy, hz]).any(1)
    return P[on].astype(np.float32)


DRONE_POINTS = drone_surface_points()


@wp.func
def ray_obb(o: wp.vec3, d: wp.vec3, c: wp.vec3, h: wp.vec3, R: wp.mat33, t_max: float):
    """Distance along the ray to the oriented box's front faces, or t_max if missed (from inside the box
    nothing is seen: rescuesim's renderer culls back faces)."""
    ol = wp.transpose(R) * (o - c)
    dl = wp.transpose(R) * d
    t0 = float(-1.0e30)
    t1 = float(1.0e30)
    for i in range(3):
        if wp.abs(dl[i]) < 1.0e-12:
            if ol[i] < -h[i] or ol[i] > h[i]:
                return t_max
        else:
            a = (-h[i] - ol[i]) / dl[i]
            b = (h[i] - ol[i]) / dl[i]
            t0 = wp.max(t0, wp.min(a, b))
            t1 = wp.min(t1, wp.max(a, b))
    if t1 < wp.max(t0, 0.0):
        return t_max
    if t0 > 0.0:
        return wp.min(t0, t_max)
    return t_max


@wp.func
def mesh_ray_front(mesh: wp.uint64, o: wp.vec3, d: wp.vec3, t_max: float):
    """Distance to the first front-facing triangle along the ray, or t_max (back faces are culled,
    as in rescuesim's TinyRenderer images: a camera inside a prop sees through its walls).

    The ground (z = 0, |x|, |y| <= GROUND_HALF) is also intersected analytically, because a prop's
    bottom face is coplanar with it and skipping that back face could skip the ground too."""
    t_g = t_max
    if d[2] < 0.0 and o[2] >= 0.0:
        tg = -o[2] / d[2]
        hx = o[0] + d[0] * tg
        hy = o[1] + d[1] * tg
        if tg < t_max and wp.abs(hx) <= GROUND_HALF and wp.abs(hy) <= GROUND_HALF:
            t_g = tg
    return wp.min(t_g, _mesh_front(mesh, o, d, t_max))


@wp.func
def _mesh_front(mesh: wp.uint64, o: wp.vec3, d: wp.vec3, t_max: float):
    t_acc = float(0.0)
    for _i in range(8):
        q = wp.mesh_query_ray(mesh, o + d * t_acc, d, t_max - t_acc)
        if not q.result:
            return t_max
        tb = t_acc + q.t
        if q.sign >= 0.0:
            return tb
        # a back face: look for a front face coplanar with it (e.g. a box resting on another one),
        # which the BVH may have returned second; re-query from a few points just before the hit
        for j in range(3):
            delta = 0.01 * float(1 + 3 * j)
            if tb - delta > t_acc:
                q2 = wp.mesh_query_ray(mesh, o + d * (tb - delta), d, 2.0 * delta)
                if q2.result and q2.sign >= 0.0 and wp.abs(q2.t - delta) < 2.0e-3:
                    return tb - delta + q2.t
        t_acc = tb + 1.0e-4
    return t_max


@wp.func
def point_obb_dist(p: wp.vec3, c: wp.vec3, h: wp.vec3, R: wp.mat33):
    l = wp.transpose(R) * (p - c)
    dx = wp.max(wp.abs(l[0]) - h[0], 0.0)
    dy = wp.max(wp.abs(l[1]) - h[1], 0.0)
    dz = wp.max(wp.abs(l[2]) - h[2], 0.0)
    return wp.sqrt(dx * dx + dy * dy + dz * dz)


@wp.func
def obb_obb_touch(ca: wp.vec3, Ra: wp.mat33, ha: wp.vec3, cb: wp.vec3, Rb: wp.mat33, hb: wp.vec3, tol: float):
    """Separating-axis test with tolerance: False when some axis separates by more than tol."""
    Rm = wp.transpose(Ra) * Rb
    t = wp.transpose(Ra) * (cb - ca)
    A = wp.mat33()
    for i in range(3):
        for j in range(3):
            A[i, j] = wp.abs(Rm[i, j]) + 1.0e-6
    for i in range(3):
        rb = hb[0] * A[i, 0] + hb[1] * A[i, 1] + hb[2] * A[i, 2]
        if wp.abs(t[i]) > ha[i] + rb + tol:
            return False
    for j in range(3):
        ra = ha[0] * A[0, j] + ha[1] * A[1, j] + ha[2] * A[2, j]
        s = t[0] * Rm[0, j] + t[1] * Rm[1, j] + t[2] * Rm[2, j]
        if wp.abs(s) > ra + hb[j] + tol:
            return False
    for i in range(3):
        i1 = (i + 1) % 3
        i2 = (i + 2) % 3
        for j in range(3):
            j1 = (j + 1) % 3
            j2 = (j + 2) % 3
            ln = wp.sqrt(wp.max(1.0 - Rm[i, j] * Rm[i, j], 0.0))
            if ln > 1.0e-4:
                ra = ha[i1] * A[i2, j] + ha[i2] * A[i1, j]
                rb = hb[j1] * A[i, j2] + hb[j2] * A[i, j1]
                s = wp.abs(t[i2] * Rm[i1, j] - t[i1] * Rm[i2, j])
                if s > ra + rb + tol * ln:
                    return False
    return True


@wp.func
def capsule_touch(p: wp.vec3, R: wp.mat33, hd: wp.vec3, c: wp.vec3, Rc: wp.mat33, r: float, hl: float, tol: float):
    ax = wp.vec3(Rc[0, 2], Rc[1, 2], Rc[2, 2])
    a = c - ax * hl
    ab = ax * (2.0 * hl)
    lo = float(0.0)
    hi = float(1.0)
    g = float(0.6180339887)
    x1 = hi - g * (hi - lo)
    x2 = lo + g * (hi - lo)
    f1 = point_obb_dist(a + ab * x1, p, hd, R)
    f2 = point_obb_dist(a + ab * x2, p, hd, R)
    for _it in range(32):
        if f1 <= r + tol or f2 <= r + tol:
            return True
        if f1 < f2:
            hi = x2
            x2 = x1
            f2 = f1
            x1 = hi - g * (hi - lo)
            f1 = point_obb_dist(a + ab * x1, p, hd, R)
        else:
            lo = x1
            x1 = x2
            f1 = f2
            x2 = lo + g * (hi - lo)
            f2 = point_obb_dist(a + ab * x2, p, hd, R)
    fe = wp.min(point_obb_dist(a, p, hd, R), point_obb_dist(a + ab, p, hd, R))
    return wp.min(wp.min(f1, f2), fe) <= r + tol


@wp.func
def cylinder_sdf(x: wp.vec3, r: float, hh: float):
    qx = wp.sqrt(x[0] * x[0] + x[1] * x[1]) - r
    qz = wp.abs(x[2]) - hh
    ox = wp.max(qx, 0.0)
    oz = wp.max(qz, 0.0)
    return wp.sqrt(ox * ox + oz * oz) + wp.min(wp.max(qx, qz), 0.0)


@wp.func
def cylinder_touch(p: wp.vec3, R: wp.mat33, hd: wp.vec3, c: wp.vec3, Rc: wp.mat33, r: float, hh: float,
                   pts: wp.array(dtype=wp.vec3), n_pts: int, tol: float):
    RcT = wp.transpose(Rc)
    for k in range(n_pts):
        x = RcT * (p + R * pts[k] - c)
        if cylinder_sdf(x, r, hh) <= tol:
            return True
    dl = RcT * (p - c)
    phi0 = wp.atan2(dl[1], dl[0])
    span = wp.min(3.14159265, 0.25 / wp.max(r, 1.0e-3))
    for side in range(2):
        z = hh * (2.0 * float(side) - 1.0)
        for m in range(33):
            phi = phi0 + span * (float(m) / 16.0 - 1.0)
            x = c + Rc * wp.vec3(r * wp.cos(phi), r * wp.sin(phi), z)
            if point_obb_dist(x, p, hd, R) <= tol:
                return True
    return False


@wp.func
def prim_touch(code: int, p: wp.vec3, R: wp.mat33, hd: wp.vec3, c: wp.vec3, Rc: wp.mat33, s: wp.vec3,
               pts: wp.array(dtype=wp.vec3), n_pts: int, tol: float):
    if code == 0:
        return obb_obb_touch(p, R, hd, c, Rc, s, tol)
    if code == 1:
        return point_obb_dist(c, p, hd, R) <= s[0] + tol
    if code == 2:
        return cylinder_touch(p, R, hd, c, Rc, s[0], s[1], pts, n_pts, tol)
    if code == 3:
        return capsule_touch(p, R, hd, c, Rc, s[0], s[1], tol)
    return False


@wp.func
def drone_touch(p: wp.vec3, R: wp.mat33, hd: wp.vec3, bvh: wp.uint64, off: int,
                shape: wp.array(dtype=wp.int32), ppos: wp.array(dtype=wp.vec3), prot: wp.array(dtype=wp.mat33),
                psize: wp.array(dtype=wp.vec3), bc: wp.vec3, bh: wp.vec3, bR: wp.mat33,
                pts: wp.array(dtype=wp.vec3), n_pts: int, tol: float):
    """Bit 1: the drone box touches the target box; bit 2: it touches any prop."""
    res = int(0)
    if obb_obb_touch(p, R, hd, bc, bR, bh, tol):
        res = 1
    ext = wp.vec3(wp.abs(R[0, 0]) * hd[0] + wp.abs(R[0, 1]) * hd[1] + wp.abs(R[0, 2]) * hd[2],
                  wp.abs(R[1, 0]) * hd[0] + wp.abs(R[1, 1]) * hd[1] + wp.abs(R[1, 2]) * hd[2],
                  wp.abs(R[2, 0]) * hd[0] + wp.abs(R[2, 1]) * hd[1] + wp.abs(R[2, 2]) * hd[2]) + wp.vec3(tol, tol, tol)
    q = wp.bvh_query_aabb(bvh, p - ext, p + ext)
    i = int(0)
    while wp.bvh_query_next(q, i):
        k = off + i
        code = shape[k]
        if code >= 0:
            if prim_touch(code, p, R, hd, ppos[k], prot[k], psize[k], pts, n_pts, tol):
                res = res | 2
                return res
    return res


@wp.kernel
def contact_kernel(pos: wp.array(dtype=wp.vec3), rot: wp.array(dtype=wp.mat33), terrain: wp.array(dtype=wp.int32),
                   bvhs: wp.array(dtype=wp.uint64), prim_off: wp.array(dtype=wp.int32),
                   shape: wp.array(dtype=wp.int32), ppos: wp.array(dtype=wp.vec3), prot: wp.array(dtype=wp.mat33),
                   psize: wp.array(dtype=wp.vec3),
                   box_c: wp.array(dtype=wp.vec3), box_h: wp.array(dtype=wp.vec3), box_R: wp.array(dtype=wp.mat33),
                   pts: wp.array(dtype=wp.vec3), n_pts: int, hd: wp.vec3, tol: float, out: wp.array(dtype=wp.int32)):
    """Contact flags (bit 1 target, bit 2 props) of drone boxes at arbitrary poses, within ``tol``."""
    d = wp.tid()
    t = terrain[d]
    out[d] = drone_touch(pos[d], rot[d], hd, bvhs[t], prim_off[t], shape, ppos, prot, psize,
                         box_c[d], box_h[d], box_R[d], pts, n_pts, tol)


def obb_distance(points, obb):
    """Distance from points (N, K, 3) to each drone's box (N, 15) -> (N, K), 0 inside."""
    c, h, R = obb[:, None, :3], obb[:, None, 3:6], obb[:, 6:].reshape(-1, 3, 3)
    local = np.einsum("nji,nkj->nki", R, points - c)
    return np.linalg.norm(np.maximum(np.abs(local) - h, 0), axis=-1)
