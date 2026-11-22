"""Stick conventions as a controller: desired world velocity + heading -> sticks (angle mode).

    a_des = Kp (v_des - v) + Ki * integral + drag feedforward     (world frame, m/s^2)
    specific force F = a_des + g z                                 (per unit mass)
    tilt from F in the heading frame -> roll and pitch sticks      (full stick = 60 deg)
    throttle = |F| / g - 1                                         (0 = hover thrust, 1 = 2 x hover)
    yaw stick from the heading error                               (full stick = 200 deg/s, + = clockwise)

Sticks (measured in this simulator): roll + = right, pitch + = forward, yaw + = clockwise,
throttle + = up. Drag: the simulator's body drag makes 119.5 km/h the top speed at 60 deg pitch, so
drag acceleration = g tan(60 deg) / v_top^2 * |v| v; we feed it forward so cruise speed holds.
"""
import numpy as np

G = 9.8
MAX_STICK_TILT = np.radians(60.0)
MAX_YAW_RATE = np.radians(200.0)
DRAG_K = G * np.tan(np.radians(60.0)) / (119.5 / 3.6) ** 2      # 1/m


class VelocityController:
    def __init__(self, kp_xy=1.5, kp_z=2.0, ki_xy=0.3, ki_z=0.8, max_tilt_deg=45.0,
                 kp_yaw=2.5, max_acc_z=6.0):
        self.kp = np.array([kp_xy, kp_xy, kp_z])
        self.ki = np.array([ki_xy, ki_xy, ki_z])
        self.max_tilt = np.radians(max_tilt_deg)
        self.kp_yaw, self.max_acc_z = kp_yaw, max_acc_z
        self.reset()

    def reset(self):
        self.integral = np.zeros(3)

    def sticks(self, state, v_des, yaw_des, dt=1 / 30, max_tilt_deg=None):
        v = state["vel"]
        err = np.asarray(v_des, dtype=float) - v
        self.integral = np.clip(self.integral + self.ki * err * dt, [-2, -2, -3], [2, 2, 3])
        a_des = self.kp * err + self.integral + DRAG_K * np.linalg.norm(v) * v
        a_des[2] = np.clip(a_des[2], -self.max_acc_z, self.max_acc_z)
        F = a_des + np.array([0.0, 0.0, G])
        F[2] = max(F[2], 0.3 * G)
        yaw = state["rpy"][2]
        cy, sy = np.cos(yaw), np.sin(yaw)
        f_fwd, f_left = F[0] * cy + F[1] * sy, -F[0] * sy + F[1] * cy
        #### Limit the tilt by scaling the horizontal part, keeping its direction
        max_tilt = self.max_tilt if max_tilt_deg is None else np.radians(max_tilt_deg)
        h = np.hypot(f_fwd, f_left)
        if h > F[2] * np.tan(max_tilt):
            scale = F[2] * np.tan(max_tilt) / h
            f_fwd, f_left = f_fwd * scale, f_left * scale
        pitch = np.arctan2(f_fwd, F[2])                                 # + = nose down = forward
        roll = np.arctan2(-f_left, np.hypot(f_fwd, F[2]))               # + = right
        thrust = np.sqrt(f_fwd ** 2 + f_left ** 2 + F[2] ** 2)
        yaw_err = (yaw_des - yaw + np.pi) % (2 * np.pi) - np.pi
        yaw_rate = np.clip(self.kp_yaw * yaw_err, -MAX_YAW_RATE, MAX_YAW_RATE)
        out = np.array([roll / MAX_STICK_TILT, pitch / MAX_STICK_TILT, -yaw_rate / MAX_YAW_RATE, thrust / G - 1.0])
        return np.clip(out, -1.0, 1.0)
