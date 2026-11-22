"""Camera geometry shared by every agent: pixels <-> rays, depth -> 3D points, detection -> world point.

Both onboard cameras (detector 160 x 90, depth 224 x 126) use the same model, as specified in
SPEC.md section 4. These are sensor specs, identical on every map:
- pinhole, 102 deg horizontal FOV, 16:9 (vertical FOV ~69.6 deg)
- tilted 25 deg up from the body x axis, mounted 0.30 m ahead of the body origin
- depth is z-depth along the camera axis (not range along each ray), clipped at 40 m
So a normalised detector box maps directly onto depth pixels.

Frames: world = VIO frame (x, y from the arena centre, z up); body FLU (x forward, y left, z up);
camera (forward, left, up). Normalised image coords: u right, v down, both in [0, 1].
`state` is a dict with "pos" (world, 3) and "R" (world-from-body rotation, 3 x 3), as returned by
`handin.core.estimator.Estimator`.
"""
import numpy as np

HFOV = np.radians(102.0)
ASPECT = 16 / 9
TAN_H = np.tan(HFOV / 2)
TAN_V = TAN_H / ASPECT
UPTILT = np.radians(25.0)
MOUNT = np.array([0.30, 0.0, 0.0])
#### Columns: camera forward, left, up, expressed in the body frame
TILT = np.array([[np.cos(UPTILT), 0.0, -np.sin(UPTILT)],
                 [0.0, 1.0, 0.0],
                 [np.sin(UPTILT), 0.0, np.cos(UPTILT)]])
DEPTH_MAX = 40.0
#### The kit clips depth at 40 m THEN adds noise (sigma = 0.03 d = 1.2 m there), so "nothing within 40 m"
#### reads 40 +- 1.2 m: 34% of sky pixels read below 39.5 m (measured, median 38.85 m). 36 m = 3 sigma
#### under the cap, so only real returns count as depth (tools/check_localise.py: p90 error at 40-80 m 24 -> 6.5 m)
DEPTH_VALID = 36.0
DETECTION_RANGE = 250.0
BOX_SIZE = np.array([4.0, 1.65, 1.55])
BOX_CENTER_Z = BOX_SIZE[2] / 2


def camera_pose(state):
    """Camera eye (world) and rotation (columns: camera forward, left, up, in world)."""
    R = state["R"]
    return state["pos"] + R @ MOUNT, R @ TILT


def pixel_ray_cam(u, v):
    """Normalised image coords -> camera-frame ray scaled so that its forward component is 1
    (so `eye + depth * R_cam @ ray` is the 3D point of a depth reading)."""
    return np.array([1.0, -(2 * u - 1) * TAN_H, (1 - 2 * v) * TAN_V])


def cam_to_pixel(d_cam):
    """Camera-frame direction -> normalised image coords (u, v), or None if behind the camera.
    The result may lie outside [0, 1] (out of view)."""
    if d_cam[0] <= 1e-3:
        return None
    return 0.5 - d_cam[1] / d_cam[0] / (2 * TAN_H), 0.5 - d_cam[2] / d_cam[0] / (2 * TAN_V)


def world_to_pixel(point, state):
    """World point -> normalised image coords, or None if behind the camera."""
    eye, R_cam = camera_pose(state)
    return cam_to_pixel(R_cam.T @ (np.asarray(point) - eye))


def ray_grid(h, w):
    """Camera-frame rays (forward = 1) at every pixel centre of an h x w image, shape (h, w, 3)."""
    u = (np.arange(w) + 0.5) / w
    v = (np.arange(h) + 0.5) / h
    uu, vv = np.meshgrid(u, v)
    return np.stack([np.ones_like(uu), -(2 * uu - 1) * TAN_H, (1 - 2 * vv) * TAN_V], axis=-1)


_RAYS = {}


def depth_points(depth, state, step=2, exclude_box=None, min_z=None, max_depth=DEPTH_MAX - 0.5,
                 exclude_band=None):
    """Depth map -> world points (N, 3) of everything seen closer than `max_depth`.

    step: keep one pixel in `step` along each axis (2 -> 63 x 112 = 7k points).
    exclude_box: normalised (x1, y1, x2, y2) whose pixels are dropped, e.g. the detected target.
    exclude_band: optional (d_lo, d_hi): inside exclude_box, drop only pixels whose depth is in this band
                  (the target itself); nearer pixels (a bush or bale in front of it) stay obstacles.
    min_z: drop points below this world height (e.g. 0.35 to drop the ground).
    Clipped pixels (40 m, "nothing closer") are never returned as points.
    """
    d = np.asarray(depth)[::step, ::step]
    key = d.shape
    if key not in _RAYS:
        _RAYS[key] = ray_grid(*key)
    rays = _RAYS[key]
    mask = d < max_depth
    if exclude_box is not None:
        h, w = d.shape
        x1, y1, x2, y2 = exclude_box
        pad = 0.02
        rows = slice(max(int((y1 - pad) * h), 0), int(np.ceil((y2 + pad) * h)))
        cols = slice(max(int((x1 - pad) * w), 0), int(np.ceil((x2 + pad) * w)))
        if exclude_band is None:
            mask[rows, cols] = False
        else:
            sub = d[rows, cols]
            mask[rows, cols] &= ~((sub >= exclude_band[0]) & (sub <= exclude_band[1]))
    eye, R_cam = camera_pose(state)
    pts = eye + (rays[mask] * d[mask][:, None]) @ R_cam.T
    if min_z is not None:
        pts = pts[pts[:, 2] > min_z]
    return pts


def box_depth(box, depth):
    """Median valid z-depth inside the inner half of a normalised detection box, or None."""
    x1, y1, x2, y2 = box
    h, w = depth.shape
    cx1, cx2 = int((x1 + 0.25 * (x2 - x1)) * w), int(np.ceil((x2 - 0.25 * (x2 - x1)) * w))
    cy1, cy2 = int((y1 + 0.25 * (y2 - y1)) * h), int(np.ceil((y2 - 0.25 * (y2 - y1)) * h))
    patch = depth[max(cy1, 0):max(cy2, cy1 + 1), max(cx1, 0):max(cx2, cx1 + 1)]
    valid = patch[patch < DEPTH_VALID]
    return float(np.median(valid)) if valid.size >= 2 else None


def occluded_points(points, z_cam, depth, state, margin=1.5):
    """For world points with camera z-depths `z_cam`: is something within depth range in front of them?

    Looks up the depth pixel each point projects to. A reading shorter than the point (or than DEPTH_VALID for
    points beyond it) by more than `margin` means an occluder. Points outside the image count as occluded."""
    eye, R_cam = camera_pose(state)
    d = (np.asarray(points) - eye) @ R_cam
    fwd = np.maximum(d[:, 0], 1e-3)
    h, w = depth.shape
    u = 0.5 - d[:, 1] / fwd / (2 * TAN_H)
    v = 0.5 - d[:, 2] / fwd / (2 * TAN_V)
    ok = (d[:, 0] > 0) & (u >= 0) & (u < 1) & (v >= 0) & (v < 1)
    col = np.clip((u * w).astype(int), 0, w - 1)
    row = np.clip((v * h).astype(int), 0, h - 1)
    reading = np.asarray(depth)[row, col]
    return ~ok | (reading < np.minimum(z_cam, DEPTH_VALID) - margin)


def localise(box, depth, state):
    """One detection -> (world point on the box, sigma_m, kind).

    kind 'depth':   the median depth inside the box gives the range (box within ~40 m). The point
                    is on the visible face (up to 2 m from the box centre, which is a good aim
                    point for touching it), at box-centre height.
    kind 'ground':  no valid depth; the ray through the box centre meets the box-centre height
                    (the box rests on the ground at z = 0). sigma grows with range^2 / altitude.
    kind 'bearing': the ray points at or above the horizon; returns a point 200 m along it,
                    sigma 1000 (direction only).
    """
    x1, y1, x2, y2 = box
    u, v = (x1 + x2) / 2, (y1 + y2) / 2
    eye, R_cam = camera_pose(state)
    ray_w = R_cam @ pixel_ray_cam(u, v)
    h, w = depth.shape
    #### Depth inside the inner half of the box (its edges may overlap the background)
    cx1, cx2 = int((x1 + 0.25 * (x2 - x1)) * w), int(np.ceil((x2 - 0.25 * (x2 - x1)) * w))
    cy1, cy2 = int((y1 + 0.25 * (y2 - y1)) * h), int(np.ceil((y2 - 0.25 * (y2 - y1)) * h))
    patch = depth[max(cy1, 0):max(cy2, cy1 + 1), max(cx1, 0):max(cx2, cx1 + 1)]
    valid = patch[patch < DEPTH_VALID]
    if valid.size >= 2:
        z = float(np.median(valid))
        point = eye + z * ray_w
        point[2] = BOX_CENTER_Z
        return point, 0.3 + 0.05 * z, "depth"
    if ray_w[2] < -0.02:
        s = (BOX_CENTER_Z - eye[2]) / ray_w[2]
        point = eye + s * ray_w
        rng = np.linalg.norm(point - eye)
        return point, max(2.0, 0.05 * rng * rng / max(eye[2], 1.0)), "ground"
    return eye + 200.0 * ray_w / np.linalg.norm(ray_w), 1e3, "bearing"
