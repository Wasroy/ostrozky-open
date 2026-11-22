"""Scripted demo agent (no learning, no obstacle avoidance).

1. Transit towards the arena centre at about 15 m altitude.
2. Search: fly an expanding square spiral around the centre.
3. Approach: when the detector fires, yaw towards the box and descend along
   the line of sight (pure pursuit) until the drone touches it.

State estimation uses only the observation: heading from the integrated gyro,
corrected slowly by the VIO course; altitude and climb rate from an
alpha-beta filter on VIO altitude with the accelerometer as input.
"""
import math

import numpy as np

DT = 1.0 / 30.0
MAX_TILT = math.radians(60.0)
CAM_TILT = math.radians(25.0)
TAN_H = math.tan(math.radians(51.0))
TAN_V = TAN_H * 9.0 / 16.0
CRUISE_ALT = 15.0


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def square_spiral(step=32.0, extent=140.0):
    pts, x, y, leg, d = [(0.0, 0.0)], 0.0, 0.0, step, 0
    dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    while max(abs(x), abs(y)) < extent:
        for _ in range(2):
            dx, dy = dirs[d % 4]
            x, y = x + dx * leg, y + dy * leg
            pts.append((max(-extent, min(extent, x)), max(-extent, min(extent, y))))
            d += 1
        leg += step
    return pts


class Agent:
    def reset(self):
        self.t = 0.0
        self.psi = None
        self.prev_xy = None
        self.vel = np.zeros(2)
        self.alt = None
        self.vz = 0.0
        self.cmd = np.zeros(4)
        self.mode = "transit"
        self.waypoints = square_spiral()
        self.wp = 0
        self.last_det_t = -1e9
        self.los = None             # (yaw error, target elevation) at last detection

    # ---------------------------------------------------------------- estimation
    def _estimate(self, obs):
        x, y, alt = (float(v) for v in obs["vio_xyz_m"])
        gyro = obs["gyro_rad_s"]
        xy = np.array([x, y])
        if self.psi is None:
            self.psi = math.atan2(-y, -x)          # start heading points roughly at the centre
            self.prev_xy = xy
            self.alt = alt
        # body z-rate -> heading rate (roll is always commanded 0 here)
        self.psi = wrap(self.psi + float(gyro[2]) * DT / math.cos(self.cmd[1] * MAX_TILT))
        v = (xy - self.prev_xy) / DT
        self.prev_xy = xy
        self.vel = 0.85 * self.vel + 0.15 * v
        speed = float(np.hypot(*self.vel))
        if speed > 3.0 and abs(float(gyro[2])) < 0.4:
            course = math.atan2(self.vel[1], self.vel[0])
            self.psi = wrap(self.psi + 0.05 * wrap(course - self.psi))
        # vertical: accelerometer (rotated with the commanded attitude) + alpha-beta on VIO altitude
        roll, pitch = self.cmd[0] * MAX_TILT, self.cmd[1] * MAX_TILT
        a = obs["accel_m_s2"]
        # world-z row of R = Rz(yaw) Ry(pitch) Rx(roll) applied to the body specific force
        az = (-math.sin(pitch) * a[0] + math.cos(pitch) * math.sin(roll) * a[1]
              + math.cos(pitch) * math.cos(roll) * a[2]) - 9.81
        self.vz += az * DT
        self.alt += self.vz * DT
        r = alt - self.alt
        self.alt += 0.12 * r
        self.vz += 0.3 * r
        return x, y, speed

    # ---------------------------------------------------------------- control
    def _throttle(self, vz_des, pitch_cmd):
        ff = 1.0 / max(0.3, math.cos(pitch_cmd * MAX_TILT)) - 1.0
        return float(np.clip(ff + 0.18 * (vz_des - self.vz), -0.6, 0.9))

    def _fly_to(self, x, y, tx, ty, alt_des, pitch_max=0.45):
        err = wrap(math.atan2(ty - y, tx - x) - self.psi)
        yaw = float(np.clip(-1.2 * err, -0.8, 0.8))
        pitch = pitch_max * max(0.0, math.cos(err)) ** 2
        vz_des = float(np.clip(0.8 * (alt_des - self.alt), -3.0, 3.0))
        return [0.0, pitch, yaw, self._throttle(vz_des, pitch)]

    def act(self, obs):
        self.t += DT
        x, y, speed = self._estimate(obs)
        if int(obs["detected"]):
            x1, y1, x2, y2 = (float(v) for v in obs["box"])
            cx, cy = (x1 + x2) / 2, y1 + 0.3 * (y2 - y1)     # aim at the upper part of the box
            yaw_err = math.atan((cx - 0.5) * 2 * TAN_H)               # + = target to the right
            el_cam = math.atan((0.5 - cy) * 2 * TAN_V)                # + = above the optical axis
            elev = CAM_TILT - self.cmd[1] * MAX_TILT + el_cam         # target elevation vs horizon
            self.los = (yaw_err, elev, max(x2 - x1, y2 - y1))
            self.last_det_t = self.t
            self.mode = "approach"
        if self.mode == "approach" and self.t - self.last_det_t > 3.0:
            self.mode = "search"
            self.wp = int(np.argmin([math.hypot(px - x, py - y) for px, py in self.waypoints]))

        if self.mode == "approach":
            yaw_err, elev, size = self.los
            fresh = self.t - self.last_det_t < 0.2
            yaw = float(np.clip(1.6 * yaw_err, -0.8, 0.8)) if fresh else 0.0
            pitch = 0.32 if size < 0.25 else 0.2
            if abs(yaw_err) > 0.5:
                pitch = 0.08
            vh = max(speed, 3.0)
            vz_des = float(np.clip(vh * math.tan(elev), -5.0, 2.0))
            if self.alt < 2.0 and size < 0.3:
                vz_des = max(vz_des, 0.8 * (2.0 - self.alt))
            cmd = [0.0, pitch, yaw, self._throttle(vz_des, pitch)]
        elif self.mode == "transit":
            if math.hypot(x, y) < 25.0:
                self.mode = "search"
                self.wp = 1
            cmd = self._fly_to(x, y, 0.0, 0.0, CRUISE_ALT)
        else:
            tx, ty = self.waypoints[self.wp % len(self.waypoints)]
            if math.hypot(tx - x, ty - y) < 10.0:
                self.wp = (self.wp + 1) % len(self.waypoints)
            cmd = self._fly_to(x, y, tx, ty, CRUISE_ALT, pitch_max=0.35)
        self.cmd = np.clip(np.asarray(cmd, float), -1.0, 1.0)
        return [float(v) for v in self.cmd]
