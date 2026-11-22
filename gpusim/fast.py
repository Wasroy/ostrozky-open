"""GPU-resident batched rescuesim: N independent flights stepped together, everything heavy on the GPU.

GPU engine originally written by @sckoott (OSTROZKY team) for the EDTH Paris 2026 hackathon; ported to rescuesim.

    env = FastBatchEnv(n=1024, terrains=("field", "forest", "village", "city", "rubble"), obs="pooled")
    obs = env.reset()
    obs, done, outcome, info = env.step(actions)          # actions (N, 4), rescuesim sticks

One fused kernel per step runs, for every drone, the 8 physics sub-steps (rescuesim's angle-mode
controller, motor allocation and rigid-body integration, see ``gpusim.dynamics`` for the NumPy
reference) with the contact test at every sub-step, then the outcome rules. The depth camera is
ray-cast with its noise added on the GPU and min-pooled to a 9 x 16 grid there; the detector box
is found on the GPU. Only small arrays come back to the host, where the IMU, VIO, detector misses
and jitter are applied.

Behaviour follows SPEC.md and rescuesim (same starts, sensors, noise models and outcomes):
* each flight has rescuesim's identity ``(seed, terrain string, run)``; its start is drawn exactly
  as ``RescueEnv`` draws it (same SeedSequence, same order of draws), so a gpusim flight starts
  where the rescuesim flight with the same identity starts;
* the sensor noise of a flight comes from a counter-based generator keyed by the same identity and
  the step index: reproducible per flight whatever the batch, but not the same samples as
  rescuesim's (which draws them from one sequential Generator);
* a flight ends at the first contact (success if only the box is touched), altitude < 0.2 m,
  |x| or |y| >= 350 m, the time limit, or an invalid action (``error``). Finished flights are frozen
  (no physics, no rendering, their last observation is kept) until ``reset(idx)``.

obs="pooled": ``depth_pool`` (N, 9, 16), the min over each 14 x 14 block of the noisy depth.
obs="full": also the full ``depth_m`` (N, 126, 224) (slower; needed to fly rescuesim agents).
``noise=False`` turns every noise source off (tests, validation). ``noise="rescuesim"`` draws every
noise sample from each flight's own NumPy Generator in rescuesim's order (depth image, detector,
IMU, VIO), so a flight gets exactly rescuesim's noise samples: slow (the depth noise is drawn on the
host), meant for validation, where it separates simulator differences from noise.
"""
from __future__ import annotations

import math

import numpy as np
import warp as wp

from rescuesim import sensors as S
from rescuesim.dynamics import GRAVITY, QuadParams
from rescuesim.env import MIN_ALTITUDE_M
from rescuesim.scene import ARENA_HALF_M
from rescuesim.terrain import TERRAIN_TYPES, stable_hash
from gpusim import dynamics as D
from gpusim.kernels import CONTACT_TOL_M, DRONE_HALF, DRONE_POINTS, contact_kernel, drone_touch, obb_distance
from gpusim.render import (CH, CW, DET_FAR, DEPTH_FAR, DH, DW, MOUNT, NEAR, NO_BOX, POOL_H, POOL_W, TILT,
                           depth_kernel, detect_kernel, pixel_rays, pool_reset_kernel, quat_rot)
from gpusim.scene import GpuScenes, load_terrain_scene

TERRAINS = TERRAIN_TYPES
OUTCOMES = ("flying", "success", "collision", "out_of_arena", "timeout", "error")
FLYING, SUCCESS, COLLISION, OUT_OF_ARENA, TIMEOUT, ERROR = range(6)


@wp.struct
class QuadConsts:
    mass: float
    g: float
    hover: float
    fmin: float
    fmax: float
    arm: float
    kyaw: float
    drag_k: float
    max_tilt: float
    max_yaw: float
    k_att: float
    k_rate: wp.vec3
    inertia: wp.vec3
    drone_half: wp.vec3


def quad_consts(p: QuadParams) -> QuadConsts:
    c = QuadConsts()
    c.mass, c.g, c.hover = p.mass, GRAVITY, p.hover_thrust
    c.fmin, c.fmax, c.arm, c.kyaw = p.motor_min, p.motor_max, p.arm_xy, p.yaw_moment_coeff
    c.drag_k, c.max_tilt, c.max_yaw, c.k_att = p.drag_k, p.max_tilt_rad, p.max_yaw_rate_rad_s, p.k_att
    c.k_rate, c.inertia = wp.vec3(*p.k_rate), wp.vec3(*p.inertia)
    c.drone_half = wp.vec3(*p.collision_half_extents)
    return c


#### Rotation helpers with rescuesim.geometry's conventions (quaternions x, y, z, w) ###########
@wp.func
def qmul(a: wp.quat, b: wp.quat):
    return wp.quat(a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
                   a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
                   a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
                   a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2])


@wp.func
def quat_euler(roll: float, pitch: float, yaw: float):
    cr = wp.cos(roll * 0.5)
    sr = wp.sin(roll * 0.5)
    cp = wp.cos(pitch * 0.5)
    sp = wp.sin(pitch * 0.5)
    cy = wp.cos(yaw * 0.5)
    sy = wp.sin(yaw * 0.5)
    return wp.quat(sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
                   cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy)


@wp.func
def allocate(C: QuadConsts, thrust: float, tq: wp.vec3):
    """rescuesim.dynamics.allocate: roll/pitch kept, collective shifted, yaw gets what is left."""
    a = C.arm
    k = 1.0 / (4.0 * a * a)
    # motors FL, FR, RR, RL: MX = (1, 1, -1, -1), MY = (1, -1, -1, 1), SPIN = (1, -1, 1, -1)
    rp = wp.vec4((tq[0] * a - tq[1] * a) * k, (-tq[0] * a - tq[1] * a) * k,
                 (-tq[0] * a + tq[1] * a) * k, (tq[0] * a + tq[1] * a) * k)
    yv = tq[2] / (4.0 * C.kyaw)
    yw = wp.vec4(yv, -yv, yv, -yv)
    rmax = wp.max(wp.max(rp[0], rp[1]), wp.max(rp[2], rp[3]))
    rmin = wp.min(wp.min(rp[0], rp[1]), wp.min(rp[2], rp[3]))
    spread = rmax - rmin
    if spread > C.fmax - C.fmin:
        rp = rp * ((C.fmax - C.fmin) / spread)
        rmax = wp.max(wp.max(rp[0], rp[1]), wp.max(rp[2], rp[3]))
        rmin = wp.min(wp.min(rp[0], rp[1]), wp.min(rp[2], rp[3]))
    base = wp.min(wp.max(thrust * 0.25, C.fmin - rmin), C.fmax - rmax)
    f = wp.vec4(base, base, base, base) + rp
    s = float(1.0)
    for i in range(4):
        if yw[i] > 1.0e-12:
            s = wp.min(s, (C.fmax - f[i]) / yw[i])
        elif yw[i] < -1.0e-12:
            s = wp.min(s, (C.fmin - f[i]) / yw[i])
    s = wp.max(0.0, s)
    out = wp.vec4()
    for i in range(4):
        out[i] = wp.clamp(f[i] + s * yw[i], C.fmin, C.fmax)
    return out


#### The fused step kernel ###################################################################
@wp.kernel
def step_kernel(pos: wp.array(dtype=wp.vec3), vel: wp.array(dtype=wp.vec3), quat: wp.array(dtype=wp.quat),
                omega: wp.array(dtype=wp.vec3), outcome: wp.array(dtype=wp.int32), steps: wp.array(dtype=wp.int32),
                act: wp.array(dtype=wp.vec4), wind: wp.array(dtype=wp.vec3), tscale: wp.array(dtype=wp.float32),
                terrain: wp.array(dtype=wp.int32), bvhs: wp.array(dtype=wp.uint64), prim_off: wp.array(dtype=wp.int32),
                shape: wp.array(dtype=wp.int32), ppos: wp.array(dtype=wp.vec3), prot: wp.array(dtype=wp.mat33),
                psize: wp.array(dtype=wp.vec3),
                box_c: wp.array(dtype=wp.vec3), box_h: wp.array(dtype=wp.vec3), box_R: wp.array(dtype=wp.mat33),
                pts: wp.array(dtype=wp.vec3), n_pts: int, C: QuadConsts, dt: float, n_sub: int, tol: float,
                min_alt: float, arena: float, max_steps: int):
    d = wp.tid()
    if outcome[d] != 0:
        return
    p = pos[d]
    v = vel[d]
    q = quat[d]
    w = omega[d]
    a = act[d]
    t = terrain[d]
    touch = int(0)
    for _s in range(n_sub):
        #### Angle-mode controller (rescuesim.dynamics.Quadrotor.control)
        R = quat_rot(q)
        yaw_now = wp.atan2(R[1, 0], R[0, 0])
        qd = quat_euler(a[0] * C.max_tilt, a[1] * C.max_tilt, yaw_now)
        qe = qmul(wp.quat(-q[0], -q[1], -q[2], q[3]), qd)
        if qe[3] < 0.0:
            qe = wp.quat(-qe[0], -qe[1], -qe[2], -qe[3])
        vn = wp.sqrt(qe[0] * qe[0] + qe[1] * qe[1] + qe[2] * qe[2])
        err = wp.vec3(0.0, 0.0, 0.0)
        if vn > 1.0e-12:
            err = wp.vec3(qe[0], qe[1], qe[2]) * (2.0 * wp.atan2(vn, qe[3]) / vn)
        rate_sp = wp.vec3(C.k_att * err[0], C.k_att * err[1], -C.max_yaw * a[2])
        alpha = wp.cw_mul(C.k_rate, rate_sp - w)
        torque = wp.cw_mul(C.inertia, alpha) + wp.cross(w, wp.cw_mul(C.inertia, w))
        f = allocate(C, C.hover * (1.0 + a[3]), torque) * tscale[d]
        #### Rigid body (rescuesim.dynamics.Quadrotor.substep): forces, then semi-implicit Euler
        thrust = f[0] + f[1] + f[2] + f[3]
        tau = wp.vec3(C.arm * (f[0] - f[1] - f[2] + f[3]), -C.arm * (f[0] + f[1] - f[2] - f[3]),
                      C.kyaw * (f[0] - f[1] + f[2] - f[3]))
        v_air = v - wind[d]
        drag = v_air * (-C.mass * C.drag_k * wp.length(v_air))
        acc = (wp.vec3(R[0, 2], R[1, 2], R[2, 2]) * thrust + drag) / C.mass - wp.vec3(0.0, 0.0, C.g)
        v = v + acc * dt
        p = p + v * dt
        wdot = wp.cw_div(tau - wp.cross(w, wp.cw_mul(C.inertia, w)), C.inertia)
        w = w + wdot * dt
        wn = wp.length(w)
        if wn > 1.0e-12:
            h = 0.5 * wn * dt
            sn = wp.sin(h) / wn
            q = wp.normalize(qmul(q, wp.quat(w[0] * sn, w[1] * sn, w[2] * sn, wp.cos(h))))
        #### Contacts at every sub-step; the state stays at the first touching sub-step
        touch = drone_touch(p, quat_rot(q), C.drone_half, bvhs[t], prim_off[t], shape, ppos, prot, psize,
                            box_c[d], box_h[d], box_R[d], pts, n_pts, tol)
        if touch != 0:
            break
    code = int(0)
    if touch == 1:
        code = 1
    elif touch != 0 or p[2] < min_alt:
        code = 2
    elif wp.abs(p[0]) >= arena or wp.abs(p[1]) >= arena:
        code = 3
    pos[d] = p
    vel[d] = v
    quat[d] = q
    omega[d] = w
    steps[d] = steps[d] + 1
    if code == 0 and steps[d] >= max_steps:
        code = 4
    outcome[d] = code


#### Counter-based host noise (reproducible per flight, independent of the batch) ##############
_M1, _M2, _GOLD = np.uint64(0xBF58476D1CE4E5B9), np.uint64(0x94D049BB133111EB), np.uint64(0x9E3779B97F4A7C15)


def _mix(z: np.ndarray) -> np.ndarray:
    z = (z ^ (z >> np.uint64(30))) * _M1
    z = (z ^ (z >> np.uint64(27))) * _M2
    return z ^ (z >> np.uint64(31))


def _counter_bits(key: np.ndarray, step: np.ndarray, n: int) -> np.ndarray:
    base = _mix(key ^ _mix(step.astype(np.uint64) * _GOLD + np.uint64(1)))
    return _mix(base[:, None] + np.arange(1, n + 1, dtype=np.uint64)[None, :] * _GOLD)


def counter_uniforms(key: np.ndarray, step: np.ndarray, n: int) -> np.ndarray:
    """(len(key), n) uniforms in (0, 1) from (flight key, step index, slot)."""
    return ((_counter_bits(key, step, n) >> np.uint64(11)).astype(np.float64) + 0.5) * 2.0 ** -53


def counter_normals(key: np.ndarray, step: np.ndarray, n: int) -> np.ndarray:
    m = n + (n & 1)
    u = counter_uniforms(key, step + np.uint64(1 << 40), m)
    r = np.sqrt(-2.0 * np.log(u[:, 0::2]))
    a = 2.0 * np.pi * u[:, 1::2]
    return np.concatenate([r * np.cos(a), r * np.sin(a)], 1)[:, :n]


def flight_key(seed: int, terrain_str: str, run: int) -> np.uint64:
    st = np.random.SeedSequence([int(seed), stable_hash(terrain_str), int(run)], spawn_key=(1,)).generate_state(2, np.uint32)
    return np.uint64((int(st[0]) << 32) | int(st[1]))


#### The batched environment ####################################################################
class FastBatchEnv:
    """N independent rescuesim flights on the GPU.

    terrains: terrain types the random resets choose from (each type is one shared scene).
    seed: rescuesim's flight seed (``RescueEnv.reset(seed)``) and the seed of the random terrain/pose
        choices of ``reset`` without ``terrains``.
    obs: "pooled" or "full" (see the module docstring); noise: True, False (no noise) or "rescuesim"
        (rescuesim's own noise samples, slow, for validation).
    timeout_s: flight time limit; layout_seeds: {type: layout_seed} for unseen layouts;
    wind (3,) or (N, 3) and thrust_scale scalar or (N,): SPEC's robustness options (default off).
    """

    def __init__(self, n, terrains=TERRAINS, seed=0, device="cuda:0", obs="pooled", noise=True, timeout_s=120.0,
                 layout_seeds=None, wind=(0.0, 0.0, 0.0), thrust_scale=1.0, params: QuadParams | None = None):
        if obs not in ("pooled", "full"):
            raise ValueError("obs must be 'pooled' or 'full'")
        wp.init()
        if noise not in (True, False, "rescuesim"):
            raise ValueError("noise must be True, False or 'rescuesim'")
        self.n, self.device, self.obs_mode = int(n), device, obs
        self.exact_noise = noise == "rescuesim"
        self.noise = bool(noise)
        self.gens = [None] * int(n)
        self.seed = int(seed)
        self.timeout_s = float(timeout_s)
        self.max_steps = int(math.ceil(self.timeout_s * D.CONTROL_HZ - 1e-6))
        self.params = params or QuadParams()
        self.consts = quad_consts(self.params)
        self.rng = np.random.default_rng(self.seed)
        self.names = list(terrains)
        layout_seeds = layout_seeds or {}
        self.scenes = [load_terrain_scene(t, layout_seeds.get(t)) for t in self.names]
        self.gpu = GpuScenes(self.scenes, device)
        n, dev = self.n, device
        z3 = lambda: wp.zeros(n, dtype=wp.vec3, device=dev)
        zi = lambda: wp.zeros(n, dtype=wp.int32, device=dev)
        #### Per-drone state (GPU)
        self.pos, self.vel, self.omega = z3(), z3(), z3()
        self.quat = wp.array(np.tile([0, 0, 0, 1.0], (n, 1)).astype(np.float32), dtype=wp.quat, device=dev)
        self.outcome, self.steps, self.terrain = zi(), zi(), zi()
        self.act = wp.zeros(n, dtype=wp.vec4, device=dev)
        self.wind = wp.array(np.broadcast_to(np.asarray(wind, np.float32), (n, 3)).copy(), dtype=wp.vec3, device=dev)
        self.tscale = wp.array(np.broadcast_to(np.asarray(thrust_scale, np.float32), (n,)).copy(), dtype=wp.float32, device=dev)
        self.box_c, self.box_h = z3(), z3()
        self.box_R = wp.zeros(n, dtype=wp.mat33, device=dev)
        self.pts = wp.array(DRONE_POINTS, dtype=wp.vec3, device=dev)
        self.rays_d = wp.array(pixel_rays(DW, DH), dtype=wp.vec3, device=dev)
        self.rays_c = wp.array(pixel_rays(CW, CH), dtype=wp.vec3, device=dev)
        self.depth = wp.full(n * DW * DH if obs == "full" else 1, S.DEPTH_MAX_M, dtype=wp.float32, device=dev)
        self.pool = wp.full(n * POOL_H * POOL_W, S.DEPTH_MAX_M, dtype=wp.float32, device=dev)
        self.bbox = wp.zeros(n * 4, dtype=wp.int32, device=dev)
        self.mask, self.det_active, self.seeds = zi(), zi(), zi()
        self.noise_buf = wp.zeros(n * DW * DH if self.exact_noise else 1, dtype=wp.float32, device=dev)
        self.mount, self.tilt = wp.vec3(*MOUNT.tolist()), wp.mat33(*TILT.ravel().tolist())
        #### Host-side flight bookkeeping and sensor state
        self.h_terrain = np.zeros(n, np.int32)
        self.h_pose = np.ones(n, np.int64)
        self.h_run = np.zeros(n, np.int64)
        self.h_obb = np.zeros((n, 15), np.float32)
        self.h_outcome = np.full(n, ERROR, np.int32)          # nothing flies before reset()
        self.key = np.zeros(n, np.uint64)
        self.obs_k = np.zeros(n, np.uint64)
        self.start = np.zeros((n, 5))                          # x, y, z, yaw, speed
        self.prev_vel = np.zeros((n, 3))
        self.vio_xy = np.zeros((n, 2))
        self.vio_prev = np.zeros((n, 2))
        self._next_run = 0
        self._obs = {"detected": np.zeros(n, np.int32), "box": np.zeros((n, 4), np.float32),
                     "gyro_rad_s": np.zeros((n, 3), np.float32), "accel_m_s2": np.zeros((n, 3), np.float32),
                     "vio_xyz_m": np.zeros((n, 3), np.float32)}
        self._state = None

    #### Host <-> device ###########################################################################
    def _host_state(self):
        if self._state is None:
            self._state = (self.pos.numpy().astype(np.float64), self.vel.numpy().astype(np.float64),
                           self.quat.numpy().astype(np.float64), self.omega.numpy().astype(np.float64))
        return self._state

    def _upload_state(self, pos, vel, quat, omega):
        for arr, val in ((self.pos, pos), (self.vel, vel), (self.quat, quat), (self.omega, omega)):
            arr.assign(np.ascontiguousarray(val, dtype=np.float32))
        self._state = None

    def set_state(self, idx, pos=None, vel=None, quat=None, omega=None):
        """Overwrite the true state of drones ``idx`` (tests and validation)."""
        idx = np.atleast_1d(idx)
        P, V, Q, W = (x.copy() for x in self._host_state())
        for arr, val in ((P, pos), (V, vel), (Q, quat), (W, omega)):
            if val is not None:
                arr[idx] = val
        self._upload_state(P, V, Q, W)

    def _contacts(self, terrain_ids, pos, R, obb, tol):
        """Contact flags (bit 1 box, bit 2 props) of drone boxes at arbitrary poses."""
        m, dev = len(pos), self.device
        out = wp.zeros(m, dtype=wp.int32, device=dev)
        f32 = lambda a: np.ascontiguousarray(a, np.float32)
        wp.launch(contact_kernel, dim=m, device=dev,
                  inputs=[wp.array(f32(pos), dtype=wp.vec3, device=dev), wp.array(f32(R), dtype=wp.mat33, device=dev),
                          wp.array(np.asarray(terrain_ids, np.int32), dtype=wp.int32, device=dev),
                          self.gpu.bvhs, self.gpu.prim_off, self.gpu.shape, self.gpu.ppos, self.gpu.prot, self.gpu.psize,
                          wp.array(f32(obb[:, :3]), dtype=wp.vec3, device=dev), wp.array(f32(obb[:, 3:6]), dtype=wp.vec3, device=dev),
                          wp.array(f32(obb[:, 6:].reshape(-1, 3, 3)), dtype=wp.mat33, device=dev),
                          self.pts, len(DRONE_POINTS), wp.vec3(*DRONE_HALF.tolist()), float(tol), out])
        return out.numpy()

    #### Reset #####################################################################################
    def terrain_string(self, i: int) -> str:
        return self.scenes[self.h_terrain[i]].terrain_string(int(self.h_pose[i]))

    def reset(self, idx=None, terrains=None, runs=None, starts=None):
        """Start new flights for drones ``idx`` (default: all) and return the batch observation.

        terrains: one "<type>@v<N>" per drone (type among this env's terrains); default random.
        runs: rescuesim run index per drone (flight seed = (seed, terrain, run)); default: a new
            run number per flight. starts: optional dicts {"pos", "yaw", "speed"} overriding the
            drawn start (like RescueEnv's start_* options)."""
        idx = np.arange(self.n) if idx is None else np.atleast_1d(np.asarray(idx, np.int64))
        m = len(idx)
        for j, i in enumerate(idx):
            if terrains is not None:
                name, _, v = terrains[j].partition("@v")
                self.h_terrain[i], self.h_pose[i] = self.names.index(name.strip()), int(v)
            else:
                self.h_terrain[i] = self.rng.integers(len(self.names))
                self.h_pose[i] = 1 + self.rng.integers(self.scenes[self.h_terrain[i]].n_poses)
            sc = self.scenes[self.h_terrain[i]]
            if not 1 <= self.h_pose[i] <= sc.n_poses:
                raise ValueError(f"{sc.kind} has poses v1..v{sc.n_poses}")
            self.h_obb[i] = sc.boxes[self.h_pose[i] - 1]
            if runs is not None:
                self.h_run[i] = int(runs[j])
            else:
                self.h_run[i] = self._next_run
                self._next_run += 1
            self.key[i] = flight_key(self.seed, self.terrain_string(i), self.h_run[i])
        xyz, yaw, speed = self._draw_starts(idx)
        if starts is not None:
            for j, s in enumerate(starts):
                if s is None:
                    continue
                xyz[j] = s.get("pos", xyz[j])
                yaw[j] = s.get("yaw", yaw[j])
                speed[j] = s.get("speed", speed[j])
        P, V, Q, W = (x.copy() for x in self._host_state())
        P[idx] = xyz
        V[idx] = np.c_[np.cos(yaw), np.sin(yaw), np.zeros(m)] * speed[:, None]
        Q[idx] = D.quat_from_euler(np.zeros(m), np.zeros(m), yaw)
        W[idx] = 0.0
        self._upload_state(P, V, Q, W)
        self.start[idx] = np.c_[xyz, yaw, speed]
        self.h_outcome[idx] = FLYING
        st = self.steps.numpy()
        st[idx] = 0
        self.steps.assign(st)
        self.outcome.assign(self.h_outcome)
        self.box_c.assign(self.h_obb[:, :3])
        self.box_h.assign(self.h_obb[:, 3:6])
        self.box_R.assign(self.h_obb[:, 6:].reshape(-1, 3, 3))
        self.terrain.assign(self.h_terrain)
        self.prev_vel[idx] = V[idx]
        self.vio_xy[idx] = P[idx, :2]
        self.vio_prev[idx] = P[idx, :2]
        self.obs_k[idx] = 0
        mask = np.zeros(self.n, bool)
        mask[idx] = True
        return self._sense(mask, first=mask)

    def _draw_starts(self, idx):
        """rescuesim's RescueEnv._draw_start for every drone, with its own flight Generator."""
        m = len(idx)
        rngs = [np.random.default_rng(np.random.SeedSequence([self.seed, stable_hash(self.terrain_string(i)),
                                                               int(self.h_run[i])])) for i in idx]
        xyz, yaw, speed = np.zeros((m, 3)), np.zeros(m), np.zeros(m)
        todo = np.arange(m)
        for _ in range(1000):
            for j in todo:
                g = rngs[j]
                r = g.uniform(225.0, 275.0)
                bearing = g.uniform(-math.pi, math.pi)
                z = g.uniform(5.0, 25.0)
                heading = bearing + math.pi + g.uniform(-math.radians(20), math.radians(20))
                speed[j] = g.uniform(0.0, 10.0)
                xyz[j] = (r * math.cos(bearing), r * math.sin(bearing), z)
                yaw[j] = math.atan2(math.sin(heading), math.cos(heading))
            sel = idx[todo]
            hits = self._contacts(self.h_terrain[sel], xyz[todo], np.tile(np.eye(3), (len(todo), 1, 1)),
                                  self.h_obb[sel], 2.0)
            todo = todo[hits != 0]
            if not len(todo):
                for j, i in enumerate(idx):
                    self.gens[i] = rngs[j]          # the flight's Generator continues with its sensor noise
                return xyz, yaw, speed
        raise RuntimeError("could not find a free start position")

    #### Step ######################################################################################
    def step(self, actions):
        """Advance every flying drone by one control step (1/30 s).

        Returns (obs, done (N,) bool, outcome (N,) int codes into OUTCOMES, info). A drone that is
        not flying ignores its action; an invalid action (non-finite or outside [-1, 1]) ends that
        flight with ``error``."""
        a = np.asarray(actions, dtype=np.float64).reshape(self.n, 4)
        flying = self.h_outcome == FLYING
        ok = np.all(np.isfinite(a), 1) & np.all(np.abs(a) <= 1.0, 1)
        bad = flying & ~ok
        if bad.any():
            self.h_outcome[bad] = ERROR
            self.outcome.assign(self.h_outcome)
        stepped = flying & ok
        self.act.assign(np.where(ok[:, None], a, 0.0).astype(np.float32))
        wp.launch(step_kernel, dim=self.n, device=self.device,
                  inputs=[self.pos, self.vel, self.quat, self.omega, self.outcome, self.steps, self.act, self.wind,
                          self.tscale, self.terrain, self.gpu.bvhs, self.gpu.prim_off, self.gpu.shape, self.gpu.ppos,
                          self.gpu.prot, self.gpu.psize, self.box_c, self.box_h, self.box_R, self.pts, len(DRONE_POINTS),
                          self.consts, D.DT_PHYS, D.SUBSTEPS, CONTACT_TOL_M, MIN_ALTITUDE_M, ARENA_HALF_M, self.max_steps])
        self._state = None
        self.h_outcome = self.outcome.numpy()
        self.obs_k[stepped] += np.uint64(1)
        obs = self._sense(stepped)
        return obs, self.h_outcome != FLYING, self.h_outcome.copy(), self.info()

    def info(self) -> dict:
        pos = self._host_state()[0]
        return {"distance_m": obb_distance(pos[:, None], self.h_obb)[:, 0],
                "t": self.steps.numpy() / float(D.CONTROL_HZ)}

    #### Sensors ###################################################################################
    def _sense(self, mask, first=None):
        n, dev = self.n, self.device
        first = np.zeros(n, bool) if first is None else first
        pos, vel, quat, omega = self._host_state()
        R = D.quat_to_matrix(quat)
        key, k = self.key, self.obs_k
        if mask.any():
            self.mask.assign(mask.astype(np.int32))
            #### Depth (noise and pooling on the GPU)
            seeds = (_counter_bits(key, k, 1)[:, 0] >> np.uint64(33)).astype(np.int32)
            self.seeds.assign(seeds)
            if self.exact_noise:
                buf = self.noise_buf.numpy().reshape(n, DH * DW)
                for i in np.flatnonzero(mask):
                    buf[i] = self.gens[i].standard_normal((DH, DW)).ravel()
                self.noise_buf.assign(buf.ravel())
            wp.launch(pool_reset_kernel, dim=n * POOL_H * POOL_W, device=dev, inputs=[self.mask, self.pool])
            wp.launch(depth_kernel, dim=n * DW * DH, device=dev,
                      inputs=[self.pos, self.quat, self.terrain, self.mask, self.gpu.meshes, self.box_c, self.box_h,
                              self.box_R, self.rays_d, self.mount, self.tilt, DW, DH, self.seeds,
                              1.0 if self.noise else 0.0, self.noise_buf, 1 if self.exact_noise else 0,
                              NEAR, DEPTH_FAR, S.DEPTH_MAX_M, self.depth, self.pool,
                              1 if self.obs_mode == "full" else 0])
            #### Detector: GPU pixel box, host misses and jitter
            eye = pos + R[:, :, 0] * float(MOUNT[0])
            dist = obb_distance(eye[:, None], self.h_obb)[:, 0]
            active = mask & (dist <= S.DETECT_RANGE_M)
            sel = np.flatnonzero(mask)
            b = np.tile(NO_BOX, (len(sel), 1)).astype(np.float64)
            if active.any():
                self.det_active.assign(active.astype(np.int32))
                self.bbox.assign(np.tile(NO_BOX, n))
                wp.launch(detect_kernel, dim=n * CW * CH, device=dev,
                          inputs=[self.pos, self.quat, self.terrain, self.gpu.meshes, self.det_active, self.box_c,
                                  self.box_h, self.box_R, self.rays_c, self.mount, self.tilt, CW, CH, NEAR, DET_FAR,
                                  self.bbox])
                b = self.bbox.numpy().reshape(n, 4)[sel].astype(np.float64)
            visible = active[sel] & (b[:, 2] >= 0)
            u, g = self._host_noise(sel, k[sel], visible, first[sel])
            seen = visible & (u >= S.MISS_PROB)
            x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2] + 1.0, b[:, 3] + 1.0
            sx = np.maximum(0.5, 0.01 * (x2 - x1))
            sy = np.maximum(0.5, 0.01 * (y2 - y1))
            x1, x2 = x1 + g[:, 0] * sx, x2 + g[:, 1] * sx
            y1, y2 = y1 + g[:, 2] * sy, y2 + g[:, 3] * sy
            bx = np.clip(np.c_[np.minimum(x1, x2) / CW, np.minimum(y1, y2) / CH,
                               np.maximum(x1, x2) / CW, np.maximum(y1, y2) / CH], 0.0, 1.0)
            det = seen.astype(np.int32)
            box = np.where(seen[:, None], bx, 0.0).astype(np.float32)
            self._obs["detected"][sel] = det
            self._obs["box"][sel] = box
            #### IMU and VIO (host, small arrays)
            fs = first[sel]
            acc = np.where(fs[:, None], 0.0, (vel[sel] - self.prev_vel[sel]) * D.CONTROL_HZ)
            self.prev_vel[sel] = vel[sel]
            Rt = np.transpose(R[sel], (0, 2, 1))
            specific = np.einsum("nij,nj->ni", Rt, acc + np.array([0.0, 0.0, 9.81]))
            self._obs["accel_m_s2"][sel] = specific + g[:, 4:7] * S.ACCEL_SIGMA
            self._obs["gyro_rad_s"][sel] = omega[sel] + g[:, 7:10] * S.GYRO_SIGMA
            dxy = pos[sel, :2] - self.vio_prev[sel]
            sig = S.VIO_XY_BASE + S.VIO_XY_PROP * np.hypot(dxy[:, 0], dxy[:, 1])
            new_xy = self.vio_xy[sel] + dxy + g[:, 10:12] * sig[:, None]
            self.vio_xy[sel] = np.where(fs[:, None], self.vio_xy[sel], new_xy)
            self.vio_prev[sel] = np.where(fs[:, None], self.vio_prev[sel], pos[sel, :2])
            self._obs["vio_xyz_m"][sel] = np.c_[self.vio_xy[sel], pos[sel, 2] + g[:, 12] * S.VIO_ALT_SIGMA]
        obs = {k_: v.copy() for k_, v in self._obs.items()}
        obs["depth_pool"] = self.pool.numpy().reshape(n, POOL_H, POOL_W)
        if self.obs_mode == "full":
            obs["depth_m"] = self.depth.numpy().reshape(n, DH, DW)
        return obs

    def _host_noise(self, sel, k, visible, first):
        """Per drone: the miss uniform and 13 standard normals (4 box jitter, 3 accel, 3 gyro, 2 VIO xy,
        1 altitude), from the counter generator, or drawn in rescuesim's order from the flight's
        Generator (noise="rescuesim": a miss draws no jitter, the first reading no VIO xy)."""
        m = len(sel)
        if not self.noise:
            return np.ones(m), np.zeros((m, 13))
        if not self.exact_noise:
            return counter_uniforms(self.key[sel], k, 1)[:, 0], counter_normals(self.key[sel], k, 13)
        u, g = np.ones(m), np.zeros((m, 13))
        for j, i in enumerate(sel):
            gen = self.gens[i]
            if visible[j]:
                u[j] = gen.random()
                if u[j] >= S.MISS_PROB:
                    g[j, 0:4] = gen.standard_normal(4)
            g[j, 4:10] = gen.standard_normal(6)
            if not first[j]:
                g[j, 10:12] = gen.standard_normal(2)
            g[j, 12] = gen.standard_normal()
        return u, g

    #### Truth and helpers #########################################################################
    def truth(self) -> dict:
        """True state (rewards, checks; never a policy input)."""
        pos, vel, quat, omega = (x.copy() for x in self._host_state())
        return {"pos": pos, "vel": vel, "quat": quat, "omega": omega, "rpy": D.euler_from_matrix(D.quat_to_matrix(quat)),
                "distance_m": obb_distance(pos[:, None], self.h_obb)[:, 0], "t": self.steps.numpy() / float(D.CONTROL_HZ),
                "outcome": self.h_outcome.copy(), "terrain": [self.terrain_string(i) for i in range(self.n)],
                "run": self.h_run.copy(), "start": self.start.copy()}

    @staticmethod
    def outcome_name(code: int) -> str | None:
        """rescuesim's outcome string (None while flying)."""
        return None if code == FLYING else OUTCOMES[int(code)]

    @staticmethod
    def single(obs, i):
        """Observation of drone i as a rescuesim observation dict (``depth_m`` needs obs="full";
        in pooled mode the dict carries ``depth_pool`` instead)."""
        out = {"detected": int(obs["detected"][i]), "box": obs["box"][i]}
        if "depth_m" in obs:
            out["depth_m"] = obs["depth_m"][i]
        else:
            out["depth_pool"] = obs["depth_pool"][i]
        for k in ("gyro_rad_s", "accel_m_s2", "vio_xyz_m"):
            out[k] = obs[k][i]
        return out
