"""Search: where to fly and where to look until the box is first detected.

Bayesian search (Koopman; Stone, *Theory of Optimal Search*; Bourgault et al. 2003): a grid belief over where the box
is, updated every few frames with "not detected" from what the camera actually covered, and a planner that flies
toward the most probability mass per second of flight.

Belief. Cells of CELL m over |x|, |y| <= HALF (uniform prior: the box can be anywhere the props are). For each cell
and each of N_AZ azimuth sectors, q = the best detection probability any single look from that sector had. Looks
from one sector are strongly correlated (same line of sight), looks from sectors >= 30 deg apart are nearly
independent (measured: P(visible | visible 30 deg away) = 0.57 vs 0.42 unconditional, tools/search_sim.py data).
So P(box in cell and still unseen) is prior * prod_sectors (1 - q).

Sensor model, per look at a cell (its centre at box-centre height): inside the camera frustum and within 250 m,
- occluded within 40 m (depth reads closer than the cell along that pixel): 0;
- within depth range and the depth reaches it: P_NEAR;
- beyond depth range, clear for the first 40 m: P_FAR(distance) * pixel factor (a 4 x 1.55 m face is under one
  detector pixel beyond ~200 m). P_FAR is the visibility measured over the 178 public + hard boxes from that
  distance, toward the pessimistic (city) side, so the planner does not believe it covered what it did not.

Planner. Score(goal) = blurred mass around the goal / (time to fly there + turn time + T0). Fly at the search
altitude toward the best goal, facing forward (speed pitches the nose down, which tilts the 25 deg up camera onto
the ground ahead). A new goal only when the old one is reached, has lost most of its mass, or is clearly beaten.

The transit is the first leg: straight at the centre, high (most boxes are seen from far on the way in: measured
median first detection 5.5 s). Interface: update(pos, R_cam, eye, occluded_fn) after each frame without a detection;
step(pos, vel, yaw, t) -> (v_des world (3,), yaw_des). Pure geometry plus a callable that answers "is this world
point hidden within depth range?", so the same code runs in the agent (depth image) and in tools/search_sim.py
(ray casts).
"""
import numpy as np

from .core.camera import TAN_H, TAN_V

PIX_SR = (2 * TAN_H / 160) * (2 * TAN_V / 90)            # solid angle of one detector pixel
FACE_AREA = 4.0 * 1.55                                    # the box's long face, m^2
DEPTH_RANGE = 36.0                                        # depth readings beyond this are "nothing seen"
P_NEAR = 0.75
FAR_D = np.array([36.0, 50.0, 80.0, 120.0, 170.0, 250.0])
FAR_P = np.array([0.55, 0.45, 0.35, 0.27, 0.20, 0.17])


class SearchPlanner:
    def __init__(self, cell=10.0, half=170.0, n_az=12, alt=30.0, speed=18.0, transit_alt=30.0, transit_speed=18.0,
                 enter=40.0, blur=35.0, t0=4.0, turn_rate=1.2, reach=25.0, climb_rate=4.0, replan_s=1.0,
                 mode="bayes", update_every=3, p_near=P_NEAR, far_scale=1.0):
        self.cell, self.half, self.n_az = cell, half, n_az
        self.alt, self.speed, self.transit_alt, self.transit_speed = alt, speed, transit_alt, transit_speed
        self.enter, self.blur, self.t0, self.turn_rate, self.reach = enter, blur, t0, turn_rate, reach
        self.climb_rate, self.replan_s, self.mode, self.update_every = climb_rate, replan_s, mode, update_every
        self.p_near, self.far_scale = p_near, far_scale
        c = np.arange(-half + cell / 2, half, cell)
        self.cx, self.cy = np.meshgrid(c, c, indexing="xy")
        self.n = len(c)
        self.pts = np.stack([self.cx.ravel(), self.cy.ravel(), np.full(self.cx.size, 0.8)], axis=1)
        self.q = np.zeros((self.cx.size, n_az))           # best single-look detection probability per sector
        self.prior = np.ones(self.cx.size) / self.cx.size
        #### Gaussian blur kernel (separable) for the planner's mass map
        k = int(np.ceil(2 * blur / cell))
        x = np.arange(-k, k + 1) * cell
        self.kern = np.exp(-0.5 * (x / blur) ** 2)
        self.goal, self.goal_t, self.goal_mass0 = None, -1e9, 0.0
        self.transit = True
        self.alt_floor = 0.0                              # set by the agent: roofs seen ahead
        self.k = 0

    #### Belief ##################################################################
    def mass(self):
        """P(box in cell and not yet detected), unnormalised, flat (n*n,)."""
        return self.prior * np.prod(1.0 - self.q, axis=1)

    def p_unseen(self):
        """Probability the box would still be unseen if it is inside the grid (sanity / stopping)."""
        return float(self.mass().sum())

    def update(self, eye, R_cam, occluded_fn):
        """One frame without a detection: lower the belief where the camera looked."""
        self.k += 1
        if self.k % self.update_every:
            return
        rel = self.pts - eye
        dc = rel @ R_cam                                   # camera frame: forward, left, up
        fwd = dc[:, 0]
        dist = np.linalg.norm(rel, axis=1)
        inview = (fwd > 1.0) & (np.abs(dc[:, 1]) < TAN_H * fwd) & (np.abs(dc[:, 2]) < TAN_V * fwd) & (dist < 250.0)
        idx = np.nonzero(inview)[0]
        if not len(idx):
            return
        occ = occluded_fn(self.pts[idx], fwd[idx])
        d = dist[idx]
        p = np.where(fwd[idx] < DEPTH_RANGE, self.p_near, self.far_scale * np.interp(d, FAR_D, FAR_P))
        p = p * np.minimum(1.0, 2.0 * FACE_AREA / (d * d * PIX_SR))
        p = np.where(occ, 0.0, p)
        az = np.arctan2(-rel[idx, 1], -rel[idx, 0])        # bearing from the cell to the drone
        b = ((az + np.pi) / (2 * np.pi) * self.n_az).astype(int) % self.n_az
        self.q[idx, b] = np.maximum(self.q[idx, b], p)

    def reseed(self, center, sigma, weight=0.8):
        """The box was seen near `center` and then lost: put `weight` of the belief in a Gaussian there (sigma m),
        forget the "not seen" evidence inside it (we saw the box there, so those looks were misses or occlusions),
        and search again with the planner (no transit)."""
        d2 = np.sum((self.pts[:, :2] - np.asarray(center)[:2]) ** 2, axis=1)
        g = np.exp(-0.5 * d2 / sigma ** 2)
        m = self.mass()
        self.prior = (1 - weight) * m / max(m.sum(), 1e-12) + weight * g / g.sum()
        self.q[:] = 0.0
        self.transit, self.goal = False, None

    def _blurred(self):
        m = self.mass().reshape(self.n, self.n)
        m = np.apply_along_axis(lambda r: np.convolve(r, self.kern, mode="same"), 0, m)
        m = np.apply_along_axis(lambda r: np.convolve(r, self.kern, mode="same"), 1, m)
        return m

    #### Planner #################################################################
    def _choose(self, pos, vel, t):
        m = self._blurred()
        p = pos[:2]
        gx, gy = self.cx, self.cy
        dx, dy = gx - p[0], gy - p[1]
        dist = np.hypot(dx, dy)
        heading = np.arctan2(vel[1], vel[0]) if np.linalg.norm(vel[:2]) > 2.0 else np.arctan2(dy, dx)
        turn = np.abs((np.arctan2(dy, dx) - heading + np.pi) % (2 * np.pi) - np.pi)
        score = m / (dist / self.speed + turn / self.turn_rate + self.t0)
        score[dist < self.reach] = 0.0
        i = np.unravel_index(int(np.argmax(score)), score.shape)
        return np.array([gx[i], gy[i]]), float(score[i]), float(m[i])

    def step(self, pos, vel, yaw, t):
        pos, vel = np.asarray(pos, float), np.asarray(vel, float)
        p = pos[:2]
        v = np.zeros(3)
        if self.transit and np.linalg.norm(p) < self.enter:
            self.transit = False
        if self.transit:
            goal, speed, alt = np.zeros(2), self.transit_speed, self.transit_alt
        else:
            need = self.goal is None or np.linalg.norm(self.goal - p) < self.reach
            if not need and t - self.goal_t > self.replan_s:
                g, s, mg = self._choose(pos, vel, t)
                cur = self._blurred()
                i = np.unravel_index(np.argmin(np.hypot(self.cx - self.goal[0], self.cy - self.goal[1])), cur.shape)
                if cur[i] < 0.3 * self.goal_mass0:
                    need = True
                self.goal_t = t
            if need:
                self.goal, _, self.goal_mass0 = self._choose(pos, vel, t)
                self.goal_t = t
            goal, speed, alt = self.goal, self.speed, self.alt
        d = goal - p
        n = np.linalg.norm(d)
        v[:2] = d / max(n, 1e-6) * speed
        alt = max(alt, self.alt_floor)
        v[2] = np.clip(1.0 * (alt - pos[2]), -self.climb_rate, self.climb_rate)
        yaw_des = np.arctan2(v[1], v[0]) if np.linalg.norm(v[:2]) > 1.0 else yaw
        return v, yaw_des
