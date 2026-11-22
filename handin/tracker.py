"""Target tracker (sckoott attempt): fuses detections into one box estimate.

Each detection is localised by the shared `core.camera.localise` (a point on the box's visible
face, with a sigma and a kind). Fusion is a weighted mean with slow forgetting; a much better fix
that disagrees with the estimate replaces it. Bearing-only fixes (box above the horizon, far)
only seed the estimate.
"""
import numpy as np

from .core.camera import localise, BOX_CENTER_Z


class TargetTracker:
    def __init__(self, forget=0.995, gate_sigmas=4.0):
        self.forget, self.gate = forget, gate_sigmas
        self.reset()

    def reset(self):
        self.pos, self.sigma = None, np.inf
        self.last_seen_t, self.n = None, 0
        self.last_box, self.last_kind, self.last_range = None, None, None

    def update(self, obs, state, t):
        if self.pos is not None:
            self.sigma = self.sigma / np.sqrt(self.forget)          # slowly widen between fixes
        if not obs["detected"]:
            return
        point, sigma, kind = localise(obs["box"], obs["depth_m"], state)
        #### Occlusion check (Anatole-JDM): the depth inside the box rectangle can belong to something in
        #### front of the box (canopy, a car). If it is much shorter than where the ray meets the ground,
        #### use the ground estimate (measured: city 26, estimate 30 m short -> terminal phase 50 m out).
        if kind == "depth":
            g_point, g_sigma, g_kind = localise(obs["box"], np.full_like(obs["depth_m"], 40.0), state)
            if g_kind == "ground":
                eye = state["pos"]
                r_depth = np.linalg.norm(point[:2] - eye[:2])
                r_ground = np.linalg.norm(g_point[:2] - eye[:2])
                if r_depth < 0.7 * r_ground - 3.0:
                    point, sigma, kind = g_point, g_sigma, g_kind
        self.last_box, self.last_kind, self.last_seen_t = np.asarray(obs["box"]), kind, t
        self.last_range = np.linalg.norm(point - state["pos"]) if kind != "bearing" else None
        if kind == "bearing":
            if self.pos is None:                                     # far: aim along the ray for now
                self.pos, self.sigma = point, sigma
            return
        point[2] = BOX_CENTER_Z
        if self.pos is None or self.sigma > 500:
            self.pos, self.sigma = point, sigma
        else:
            d = np.linalg.norm(point[:2] - self.pos[:2])
            if d > self.gate * np.hypot(sigma, self.sigma) and sigma < self.sigma:
                self.pos, self.sigma = point, sigma                 # much better fix disagrees: jump
            else:
                w1, w2 = 1 / self.sigma ** 2, 1 / sigma ** 2
                self.pos = (w1 * self.pos + w2 * point) / (w1 + w2)
                self.sigma = 1 / np.sqrt(w1 + w2)
        self.n += 1
