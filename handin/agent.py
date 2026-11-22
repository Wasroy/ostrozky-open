"""Search & Rescue agent, E10o0/optimal-agent: find the orange box and touch it, touching nothing else.

Built on merge/v2 (@sckoott's layered agent + @Wasroy's v3 predictive avoider + @E10o0's terminal watchdog +
@Anatole-JDM's tracker occlusion check), with the changes argued in docs/OPTIMAL_AGENT.md:
1. Search = Bayesian search (handin/search.py): a belief grid over where the box is, lowered every frame where the
   camera looked and saw nothing, and a planner that flies to the most probability per second. Flown at 30 m,
   above every roof seen so far (`ceiling`), at 22 m/s: the nose-down pitch tilts the up-tilted camera onto the
   ground ahead. Replaces the transit + 12 m spiral (measured offline on 828 searches: 99.2% detected within 75 s
   vs 96.1%; city 97% vs 88%; no time spent below building tops).
2. Approach: a 10 deg glide + 1 m, so the box stays inside the camera's vertical view (it sees at most ~10 deg plus
   the pitch below the horizon); then a "dash" at cruise speed into the box with the avoider told that the box is
   the goal (@sckoott's rl-patches 88f918f): contact is success, so a path that reaches the box first is safe, and
   only pixels at the box's own depth are ignored (a bush in front of it stays an obstacle).
3. Recovery: when the box is lost (terminal watchdog, then a failed re-acquire), the search belief is re-seeded as a
   Gaussian around the last estimate and the same planner searches it, instead of restarting from scratch.

Layers: core.Estimator -> TargetTracker -> mission (this file; SearchPlanner while nothing is seen) ->
PredictiveAvoidance -> core.VelocityController. Uses only the observation passed to act().
"""
import numpy as np
from scipy.spatial import cKDTree

from .avoidance_pred import PredictiveAvoidance
from .core import Estimator, HeadingInit, VelocityController
from .core import camera
from .search import SearchPlanner
from .tracker import TargetTracker

DT = 1 / 30


class Agent:
    #### Search (handin/search.py; values chosen with tools/search_sim.py on the public + hard boxes)
    SEARCH = dict(alt=30.0, transit_alt=30.0, speed=22.0, transit_speed=22.0, p_near=0.5, far_scale=0.7)
    CEILING_MARGIN = 8.0         # m above the highest obstacle point seen nearby while searching
    CEILING_RADIUS = 60.0        # m: obstacle points this close (horizontally) set the ceiling
    CEILING_NEIGHBOURS = 3       # a point is real only with more than this many neighbours within 1.5 m
    CEILING_MAX = 70.0           # m
    #### Approach and terminal
    APPROACH_SPEED = 10.0
    GLIDE_DEG = 10.0             # approach glide slope; the camera sees ~10 deg + pitch below the horizon
    GLIDE_OFFSET = 1.0           # m above the glide line
    TERMINAL_DIST = 20.0         # horizontal distance to the estimate that starts the terminal phase
    MIN_ALT = 0.6
    #### Terminal watchdog (merge/v2): no detection for WATCHDOG_S and no progress toward the estimate (or no
    #### detection for twice that) -> back to approach, higher, with sigma inflated; nothing seen for REACQ_S ->
    #### re-seeded search around the estimate. Not within WATCHDOG_NEAR of the estimate.
    WATCHDOG_S = 4.0
    WATCHDOG_PROGRESS = 0.5
    WATCHDOG_NEAR = 3.0
    REACQ_S = 8.0
    REACQ_ALT = 5.0
    REACQ_SIGMA = 10.0
    RESEED_SIGMA = 20.0          # m, minimum spread of the re-seeded belief

    def __init__(self):
        self.reset()

    def reset(self):
        self.est = Estimator()
        self.init = HeadingInit()
        self.ctrl = VelocityController()
        self.avoid = PredictiveAvoidance()
        self.target = TargetTracker()
        self.search = SearchPlanner(**self.SEARCH)
        self.t = 0.0
        self.phase = "init"
        self.progress = []
        self.reacq_until = None
        self.reacq_seen_t = None
        self.n_reacq = 0
        self.n_reseed = 0
        self.ceiling = 0.0
        self._ceil_k, self._ceil_top = 0, 0.0
        self.debug = {}

    #### Phases ####################################################################
    def _search(self, s, obs):
        self.search.alt_floor = self.ceiling
        v, yaw_des = self.search.step(s["pos"], s["vel"], s["rpy"][2], self.t)
        return v, yaw_des

    def _approach(self, s):
        T = self.target.pos
        d_xy = T[:2] - s["pos"][:2]
        dist = np.linalg.norm(d_xy)
        if self.reacq_until is not None:
            if self.target.last_seen_t != self.reacq_seen_t:
                self.reacq_until = None                      # seen again: a normal approach from here
            elif self.t > self.reacq_until:                  # nothing seen: search around the estimate
                self._lost()
                return None
        if dist < self.TERMINAL_DIST and self.target.sigma < 5.0 and self.reacq_until is None:
            self.phase = "terminal"
            self.progress = []
            return self._terminal(s)
        line = dist * np.tan(np.radians(self.GLIDE_DEG)) + self.GLIDE_OFFSET
        alt_des = float(np.clip(line, 3.0, max(s["pos"][2], 3.0)))
        if self.reacq_until is not None:
            alt_des = max(alt_des, self.REACQ_ALT)          # climb out of the cover to look again
        v = np.zeros(3)
        v[:2] = d_xy / max(dist, 1e-6) * self.APPROACH_SPEED
        v[2] = np.clip(1.0 * (alt_des - s["pos"][2]), -3, 2)
        return v

    def _terminal(self, s):
        T = self.target.pos
        dist = np.linalg.norm(T[:2] - s["pos"][:2])
        if self._watchdog(dist):
            return self._approach(s)
        #### Dash: a straight 3D line at the box centre at cruise speed until contact (contact = success). Flying
        #### fast pitches the nose down, which brings the up-tilted camera down onto the box.
        aim = np.array([T[0], T[1], camera.BOX_CENTER_Z])
        d = aim - s["pos"]
        v = d / max(np.linalg.norm(d), 1e-6) * self.APPROACH_SPEED
        v[2] = np.clip(v[2], -3.0, 2.0)
        if dist > self.TERMINAL_DIST * 1.8:
            self.phase = "approach"
        return v

    def _watchdog(self, dist):
        """Terminal stuck (frozen far from the box, or circling it unseen)? Then start a re-acquire."""
        self.progress.append((self.t, dist))
        while self.progress[0][0] < self.t - self.WATCHDOG_S:
            self.progress.pop(0)
        unseen = self.t - self.target.last_seen_t if self.target.last_seen_t is not None else np.inf
        full = self.progress[0][0] <= self.t - self.WATCHDOG_S + 2 * DT
        stalled = full and self.progress[0][1] - dist < self.WATCHDOG_PROGRESS
        if dist < self.WATCHDOG_NEAR or unseen < self.WATCHDOG_S or not (stalled or unseen > 2 * self.WATCHDOG_S):
            return False
        self.target.sigma = max(self.target.sigma, self.REACQ_SIGMA)
        self.reacq_until, self.reacq_seen_t = self.t + self.REACQ_S, self.target.last_seen_t
        self.n_reacq += 1
        self.phase = "approach"
        return True

    def _lost(self):
        """Re-acquire failed: search a Gaussian around the last estimate with the search planner."""
        self.reacq_until = None
        self.search.reseed(self.target.pos, max(self.RESEED_SIGMA, 2 * min(self.target.sigma, 50.0)))
        self.target.reset()
        self.n_reseed += 1
        self.phase = "search"

    def _update_ceiling(self):
        """Highest real obstacle point remembered near the drone (avoider memory) + margin; decays when clear.

        Real surfaces fill neighbouring voxels; noisy sky pixels (the 3-sigma tail of the clipped 40 m reading
        falls under DEPTH_VALID, ~1 pixel a frame) are isolated points that would otherwise ratchet the ceiling
        up without end (measured: city@v36 climbed to 495 m). So a point counts only with >= CEILING_NEIGHBOURS
        others within 1.5 m, and the ceiling never exceeds CEILING_MAX."""
        self._ceil_k += 1
        if self._ceil_k % 10 == 0:
            pts = self.avoid.mem_pts
            top = 0.0
            if len(pts):
                p = self.est.state()["pos"]
                near = np.einsum("ij,ij->i", pts[:, :2] - p[:2], pts[:, :2] - p[:2]) < self.CEILING_RADIUS ** 2
                cand = pts[near & (pts[:, 2] > 3.0)]
                if len(cand) > self.CEILING_NEIGHBOURS:
                    n = cKDTree(cand).query_ball_point(cand, 1.5, return_length=True)
                    real = cand[n > self.CEILING_NEIGHBOURS]
                    if len(real):
                        top = min(float(real[:, 2].max()) + self.CEILING_MARGIN, self.CEILING_MAX)
            self._ceil_top = top
        self.ceiling = max(self._ceil_top, self.ceiling - 0.5 * DT)

    #### Main entry ###############################################################
    def act(self, observation):
        s = self.est.update(observation)
        self.t += DT
        if self.phase == "init":
            sticks = self.init.step(observation, s["rpy"][2])
            if sticks is not None:
                return sticks.astype(np.float32)
            self.est.set_yaw(self.init.yaw)
            s = self.est.state()
            self.phase = "search"
        self.target.update(observation, s, self.t)
        if self.target.pos is not None and self.phase == "search":
            self.phase = "approach"
        yaw_des = None
        v_des = None
        if self.phase == "approach":
            v_des = self._approach(s)
        elif self.phase == "terminal":
            v_des = self._terminal(s)
        if self.phase == "search" or v_des is None:
            if not observation["detected"]:
                self.search.update(*camera.camera_pose(s),
                                   lambda pts, z: camera.occluded_points(pts, z, observation["depth_m"], s))
            self._update_ceiling()
            v_des, yaw_des = self._search(s, observation)
        #### Altitude floor
        if s["pos"][2] < self.MIN_ALT + 0.3 and self.phase != "terminal":
            v_des[2] = max(v_des[2], 1.0)
        elif s["pos"][2] < self.MIN_ALT:
            v_des[2] = max(v_des[2], 0.5)
        exclude = observation["box"] if observation["detected"] and self.phase in ("approach", "terminal") else None
        band, goal = None, None
        if exclude is not None:
            zb = camera.box_depth(exclude, observation["depth_m"])
            band = (zb - 1.0, zb + 2.5) if zb is not None else None     # box seen obliquely: up to ~4 m deep
        tgt = self.target.pos if self.phase in ("approach", "terminal") else None
        if tgt is not None and self.target.sigma < 5.0:
            goal = np.array([tgt[0], tgt[1], camera.BOX_CENTER_Z])
        v_safe = self.avoid.filter(v_des, observation["depth_m"], s, exclude, target=tgt, exclude_band=band,
                                   goal=goal)
        #### Face where we fly (the camera looks forward); hold heading when slow
        if yaw_des is None or np.linalg.norm(v_safe[:2] - v_des[:2]) > 1.0:
            yaw_des = np.arctan2(v_safe[1], v_safe[0]) if np.linalg.norm(v_safe[:2]) > 1.0 else s["rpy"][2]
        if self.phase in ("approach", "terminal") and self.target.pos is not None:
            to_t = self.target.pos[:2] - s["pos"][:2]
            if np.linalg.norm(to_t) < 30.0:
                yaw_des = np.arctan2(to_t[1], to_t[0])   # keep the target in view when close
        self.debug = {"phase": self.phase, "v_des": v_des, "v_safe": v_safe, "target": self.target.pos,
                      "sigma": self.target.sigma, "avoid": dict(self.avoid.last), "n_reacq": self.n_reacq,
                      "n_reseed": self.n_reseed, "ceiling": self.ceiling}
        return self.ctrl.sticks(s, v_safe, yaw_des).astype(np.float32)
