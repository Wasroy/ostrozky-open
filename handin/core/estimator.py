"""State estimation from the observation only: gyro, accelerometer and VIO.

- Attitude: quaternion integration of the gyro (trapezoidal), starting level. The simulator's gyro
  has white noise and no bias drift, so roll and pitch stay accurate; a weak accelerometer
  correction (gravity direction, minus our own estimated acceleration) bounds the integration error.
- Heading: the unknown at start. The drone starts moving along its heading, so the direction of its
  first VIO displacement is the heading; `HeadingInit` handles that, with a forward push if slow.
- Position and velocity: one Kalman filter per world axis, state [p, v], driven by the accelerometer
  rotated to the world frame, corrected by VIO.
- Heading correction: whenever the drone accelerates, the velocity change seen by VIO alone and the
  one integrated from the accelerometer (rotated by our heading) must point the same way; the angle
  between them is the heading error, removed a fraction at a time.

Frames: world x, y as VIO (metres from the arena centre), z up; body FLU (x forward, y left, z up).
"""
import numpy as np

G = 9.8


def quat_mult(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def quat_from_rotvec(v):
    angle = np.linalg.norm(v)
    if angle < 1e-12:
        return np.array([1.0, *(0.5 * v)])
    axis = v / angle
    return np.array([np.cos(angle / 2), *(np.sin(angle / 2) * axis)])


def quat_to_matrix(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def quat_from_yaw(yaw):
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def rpy_from_matrix(R):
    roll = np.arctan2(R[2, 1], R[2, 2])
    pitch = np.arcsin(np.clip(-R[2, 0], -1, 1))
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return np.array([roll, pitch, yaw])


class Estimator:
    def __init__(self, dt=1 / 30, accel_gain=0.02, vio_sigma=(0.05, 0.05, 0.05), accel_sigma=0.3,
                 yaw_gain=0.5, yaw_window=15, yaw_min_dv=1.5):
        self.dt = dt
        self.yaw_gain, self.yaw_window, self.yaw_min_dv = yaw_gain, yaw_window, yaw_min_dv
        self.accel_gain = accel_gain
        self.R_meas = np.square(np.asarray(vio_sigma))
        self.q_acc = accel_sigma
        self.reset()

    def reset(self):
        self.q = np.array([1.0, 0, 0, 0])
        self.prev_gyro = None
        self.x = None                       # (3, 2): per axis [p, v]
        self.P = None
        self.acc_world = np.zeros(3)
        self.vio_hist, self.vimu_hist = [], []           # horizontal VIO positions; integrated IMU velocity
        self.v_imu = np.zeros(2)
        self.heading_fixes = 0

    def set_yaw(self, yaw):
        """Rotate the attitude about world z so its heading becomes `yaw` (keeps roll and pitch)."""
        current = rpy_from_matrix(quat_to_matrix(self.q))[2]
        self.q = quat_mult(quat_from_yaw(yaw - current), self.q)
        self.q /= np.linalg.norm(self.q)

    def update(self, obs):
        dt = self.dt
        gyro = np.asarray(obs["gyro_rad_s"], dtype=float)
        f_body = np.asarray(obs["accel_m_s2"], dtype=float)
        vio = np.asarray(obs["vio_xyz_m"], dtype=float)
        #### Attitude: trapezoidal gyro integration (body rates, right-multiplied)
        w = gyro if self.prev_gyro is None else 0.5 * (gyro + self.prev_gyro)
        self.prev_gyro = gyro
        self.q = quat_mult(self.q, quat_from_rotvec(w * dt))
        self.q /= np.linalg.norm(self.q)
        R = quat_to_matrix(self.q)
        #### Weak tilt correction: measured specific force vs the one our attitude and acceleration predict
        if self.accel_gain > 0 and self.x is not None:
            f_pred = R.T @ (self.acc_world + np.array([0, 0, G]))
            err = np.cross(f_body / max(np.linalg.norm(f_body), 1e-6), f_pred / max(np.linalg.norm(f_pred), 1e-6))
            err[2] = 0.0                    # gravity says nothing about heading
            self.q = quat_mult(self.q, quat_from_rotvec(-self.accel_gain * err))
            self.q /= np.linalg.norm(self.q)
            R = quat_to_matrix(self.q)
        #### Position and velocity: KF per world axis, accelerometer as input
        a_world = R @ f_body - np.array([0, 0, G])
        if self.x is None:
            self.x = np.stack([vio, np.zeros(3)], axis=1)
            self.P = np.array([np.diag([0.01, 25.0]) for _ in range(3)])   # speed unknown at start (0-10 m/s)
        F = np.array([[1, dt], [0, 1]])
        B = np.array([0.5 * dt * dt, dt])
        Q = np.outer(B, B) * self.q_acc ** 2
        for i in range(3):
            x = F @ self.x[i] + B * a_world[i]
            P = F @ self.P[i] @ F.T + Q
            S = P[0, 0] + self.R_meas[i]
            K = P[:, 0] / S
            x = x + K * (vio[i] - x[0])
            P = P - np.outer(K, P[0])
            self.x[i], self.P[i] = x, P
        self.acc_world = 0.8 * self.acc_world + 0.2 * a_world
        self._correct_heading(vio[:2], a_world[:2])
        return self.state()

    def _correct_heading(self, vio_xy, a_xy):
        """Compares 0.5 s average velocities W frames apart, from VIO alone and from the IMU."""
        W = self.yaw_window
        self.v_imu = self.v_imu + a_xy * self.dt
        self.vio_hist.append(vio_xy.copy())
        self.vimu_hist.append(self.v_imu.copy())
        if len(self.vio_hist) > 3 * W + 1:
            self.vio_hist.pop(0); self.vimu_hist.pop(0)
        if len(self.vio_hist) < 3 * W + 1 or self.yaw_gain <= 0:
            return
        h, m = np.array(self.vio_hist), np.array(self.vimu_hist)
        T = W * self.dt
        dv_vio = (h[-1] - h[-1 - W]) / T - (h[W] - h[0]) / T
        dv_imu = m[-W:].mean(axis=0) - m[1:W + 1].mean(axis=0)
        if np.linalg.norm(dv_vio) < self.yaw_min_dv or np.linalg.norm(dv_imu) < self.yaw_min_dv:
            return
        err = np.arctan2(dv_imu[1], dv_imu[0]) - np.arctan2(dv_vio[1], dv_vio[0])
        err = (err + np.pi) % (2 * np.pi) - np.pi
        if abs(err) < np.radians(45):                   # ignore outliers (e.g. a collision)
            self.q = quat_mult(quat_from_yaw(-self.yaw_gain * err), self.q)
            self.q /= np.linalg.norm(self.q)
            self.heading_fixes += 1
        #### One correction per independent window: the history was built with the old heading
        self.vio_hist, self.vimu_hist, self.v_imu = [], [], np.zeros(2)

    def state(self):
        R = quat_to_matrix(self.q)
        return {"pos": self.x[:, 0].copy(), "vel": self.x[:, 1].copy(), "R": R,
                "rpy": rpy_from_matrix(R), "acc": self.acc_world.copy()}


class HeadingInit:
    """Finds the starting heading from VIO motion.

    The drone starts flying along its heading (0-10 m/s). With level sticks it keeps that direction
    (drag only slows it), so the VIO displacement over the first frames gives the heading. If it
    moved less than `min_disp_m`, we push forward (pitch) and measure again: motion is then along
    body x. The yaw rotation integrated since the start is accounted for.
    """

    def __init__(self, coast_frames=6, push_frames=15, min_disp_m=0.8, push_pitch=0.25):
        self.coast_frames, self.push_frames = coast_frames, push_frames
        self.min_disp, self.push_pitch = min_disp_m, push_pitch
        self.reset()

    def reset(self):
        self.frame, self.start_xy, self.phase, self.yaw = 0, None, "coast", None

    def step(self, obs, est_rpy_yaw):
        """Returns sticks to fly during initialisation, or None once the heading is known
        (then `self.yaw` holds the heading at this frame)."""
        xy = np.asarray(obs["vio_xyz_m"][:2], dtype=float)
        if self.start_xy is None:
            self.start_xy, self.yaw0_int = xy, est_rpy_yaw
        self.frame += 1
        limit = self.coast_frames if self.phase == "coast" else self.push_frames
        if self.frame >= limit:
            d = xy - self.start_xy
            if np.linalg.norm(d) >= self.min_disp:
                #### Heading at the start = motion direction; add the yaw turned since (gyro)
                self.yaw = float(np.arctan2(d[1], d[0]) + (est_rpy_yaw - self.yaw0_int))
                return None
            if self.phase == "coast":
                self.phase, self.frame, self.start_xy, self.yaw0_int = "push", 0, xy, est_rpy_yaw
            else:                                       # still slow: push longer
                self.frame = 0
        return np.array([0.0, self.push_pitch if self.phase == "push" else 0.0, 0.0, 0.05])
