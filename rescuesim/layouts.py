"""The five procedural terrain types. Each builder fills a :class:`Scene` with
primitives and with candidate hiding spots; :mod:`rescuesim.terrain` then
validates the candidates and keeps a well-spread subset as target poses."""
from __future__ import annotations

import math

import numpy as np

from .geometry import quat_from_axis_angle
from .scene import TARGET_HALF, Scene, rgba

TW = TARGET_HALF[1]          # target half width (0.825 m)


def _jitter_cuts(rng, lo, hi, step_lo, step_hi, min_last):
    xs = [lo]
    while hi - xs[-1] > step_hi:
        xs.append(xs[-1] + rng.uniform(step_lo, step_hi))
    if hi - xs[-1] < min_last and len(xs) > 1:
        xs[-1] = hi
    else:
        xs.append(hi)
    return xs


def _pieces(rng, a, b, piece=(10, 22), gap=(4, 9)):
    """Split segment a-b into solid pieces separated by gaps."""
    ax, ay = a
    bx, by = b
    L = math.hypot(bx - ax, by - ay)
    if L < 1:
        return []
    ux, uy = (bx - ax) / L, (by - ay) / L
    out, t = [], rng.uniform(0, gap[1])
    while t < L - 3:
        e = min(L, t + rng.uniform(*piece))
        out.append(((ax + ux * t, ay + uy * t), (ax + ux * e, ay + uy * e)))
        t = e + rng.uniform(*gap)
    return out


class _PointGrid:
    """Minimum-distance check for scattered points (Poisson-disk style dart throwing)."""

    def __init__(self, dmin):
        self.d = dmin
        self.cells = {}

    def ok(self, x, y):
        cx, cy = int(math.floor(x / self.d)), int(math.floor(y / self.d))
        for i in range(cx - 1, cx + 2):
            for j in range(cy - 1, cy + 2):
                for (px, py) in self.cells.get((i, j), ()):
                    if (px - x) ** 2 + (py - y) ** 2 < self.d * self.d:
                        return False
        return True

    def add(self, x, y):
        self.cells.setdefault((int(math.floor(x / self.d)), int(math.floor(y / self.d))), []).append((x, y))


# =============================================================================
# field: open farmland
# =============================================================================
CROPS = [rgba(0.78, 0.70, 0.38), rgba(0.46, 0.35, 0.24), rgba(0.36, 0.55, 0.26), rgba(0.55, 0.62, 0.30)]
HEDGE = rgba(0.17, 0.36, 0.14)
CANOPY = [rgba(0.20, 0.45, 0.16), rgba(0.26, 0.50, 0.18), rgba(0.16, 0.38, 0.13)]
TRUNK = rgba(0.36, 0.26, 0.17)
ROCK = rgba(0.52, 0.51, 0.48)
BUSH = rgba(0.22, 0.42, 0.18)
HAY = rgba(0.84, 0.74, 0.40)


def build_field(sc: Scene):
    rng = sc.rng
    xs = _jitter_cuts(rng, -150.0, 150.0, 42, 70, 30)
    ys = _jitter_cuts(rng, -150.0, 150.0, 42, 70, 30)
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            cx, cy = (xs[i] + xs[i + 1]) / 2, (ys[j] + ys[j + 1]) / 2
            sc.decal(cx, cy, (xs[i + 1] - xs[i]) / 2 - 1.5, (ys[j + 1] - ys[j]) / 2 - 1.5, 0.0,
                     CROPS[int(rng.integers(len(CROPS)))])

    # hedgerows and farm tracks along interior boundaries
    lines = []
    for i in range(1, len(xs) - 1):
        for j in range(len(ys) - 1):
            lines.append(((xs[i], ys[j] + 2.5), (xs[i], ys[j + 1] - 2.5)))
    for j in range(1, len(ys) - 1):
        for i in range(len(xs) - 1):
            lines.append(((xs[i] + 2.5, ys[j]), (xs[i + 1] - 2.5, ys[j])))
    for a, b in lines:
        r = rng.random()
        if r < 0.6:
            for pa, pb in _pieces(rng, a, b):
                h, w = rng.uniform(1.8, 3.0), rng.uniform(1.2, 1.8)
                sc.hedge(pa, pb, h, w, HEDGE)
                sc.spots_along(pa, pb, w / 2 + rng.uniform(0.35, 0.7) + TW, "hedge", n=1)
                L = math.hypot(pb[0] - pa[0], pb[1] - pa[1])
                t = rng.uniform(4, 25)
                while t < L - 2:
                    if rng.random() < 0.45:
                        f = t / L
                        tx, ty = pa[0] + (pb[0] - pa[0]) * f, pa[1] + (pb[1] - pa[1]) * f
                        sc.tree(tx, ty, rng.uniform(5.0, 7.0), rng.uniform(2.6, 3.8), 0.3,
                                CANOPY[int(rng.integers(3))], TRUNK)
                    t += rng.uniform(15, 30)
        elif r < 0.8:
            L = math.hypot(b[0] - a[0], b[1] - a[1])
            yaw = math.atan2(b[1] - a[1], b[0] - a[0])
            sc.decal((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, L / 2, 2.0, yaw, rgba(0.55, 0.47, 0.36), z=0.03)

    # farmsteads: barn, sheds, bale stacks, a tractor
    for _ in range(4):
        for _try in range(60):
            x, y = rng.uniform(-125, 125, 2)
            yaw = rng.choice([0.0, math.pi / 2]) + rng.normal(0, 0.05)
            w, d, h = rng.uniform(16, 22), rng.uniform(9, 11), rng.uniform(6.0, 7.0)
            if sc.free_rect(x, y, w / 2 + 12, d / 2 + 12, yaw):
                break
        else:
            continue
        sc.house(x, y, yaw, w, d, h, rgba(0.55, 0.20, 0.16), rgba(0.30, 0.30, 0.32))
        sc.spots_box_sides(x, y, yaw, w / 2, d / 2, rng.uniform(0.4, 0.8), "barn")
        c, s = math.cos(yaw), math.sin(yaw)
        for _k in range(3):
            for _t in range(20):
                lx, ly = rng.uniform(-w / 2 - 10, w / 2 + 10), rng.choice([-1, 1]) * rng.uniform(d / 2 + 5, d / 2 + 11)
                px, py = x + c * lx - s * ly, y + s * lx + c * ly
                kind = rng.random()
                if kind < 0.45:
                    sw, sd, shh = rng.uniform(5, 8), rng.uniform(3.5, 5), rng.uniform(2.6, 3.4)
                    if sc.free_rect(px, py, sw / 2 + 1.5, sd / 2 + 1.5, yaw):
                        sc.shed(px, py, yaw, sw, sd, shh)
                        sc.spots_box_sides(px, py, yaw, sw / 2, sd / 2, rng.uniform(0.4, 0.7), "shed")
                        break
                elif kind < 0.8:
                    bw, bd, bh = rng.uniform(5, 8), rng.uniform(2.4, 3.2), rng.uniform(1.8, 2.6)
                    if sc.free_rect(px, py, bw / 2 + 1.5, bd / 2 + 1.5, yaw):
                        sc.ground_box(px, py, bw / 2, bd / 2, bh, yaw, HAY, "hay")
                        sc.spots_box_sides(px, py, yaw, bw / 2, bd / 2, rng.uniform(0.4, 0.7), "hay")
                        break
                else:
                    if sc.free_rect(px, py, 3.0, 1.6, yaw):
                        sc.car(px, py, yaw, rgba(0.15, 0.45, 0.20), tag="tractor")
                        break

    # solitary trees in the fields
    for _ in range(30):
        x, y = rng.uniform(-145, 145, 2)
        if sc.free_circle(x, y, 4.5):
            tr = rng.uniform(0.28, 0.4)
            rc = rng.uniform(3.0, 4.2)
            sc.tree(x, y, rc + rng.uniform(2.2, 3.2), rc, tr, CANOPY[int(rng.integers(3))], TRUNK)
            sc.spots_around(x, y, tr + rng.uniform(0.4, 0.7) + TW, "tree", n=2)

    # round hay bales in rows and clusters
    for _ in range(16):
        x, y = rng.uniform(-140, 140, 2)
        yaw = rng.uniform(0, math.pi)
        n = int(rng.integers(3, 8))
        if not sc.free_rect(x, y, n * 0.9 + 1, 2.0, yaw):
            continue
        c, s = math.cos(yaw), math.sin(yaw)
        for k in range(n):
            o = (k - (n - 1) / 2) * 1.7
            sc.lying("cylinder", x + c * o, y + s * o, 0.75, 1.2, yaw + math.pi / 2, HAY, "hay")
        L = n * 1.7
        a = (x - c * L / 2, y - s * L / 2)
        b = (x + c * L / 2, y + s * L / 2)
        sc.spots_along(a, b, 0.75 + rng.uniform(0.4, 0.7) + TW, "bales", n=1)

    # rocks and bushes
    for _ in range(60):
        x, y = rng.uniform(-145, 145, 2)
        r = rng.uniform(0.5, 1.5)
        if sc.free_circle(x, y, r + 2.5):
            sc.rock(x, y, r, ROCK)
            sc.spots_around(x, y, r + rng.uniform(0.4, 0.7) + TW, "rock", n=1)
    for _ in range(80):
        x, y = rng.uniform(-145, 145, 2)
        r = rng.uniform(0.7, 1.4)
        if sc.free_circle(x, y, r + 0.5):
            sc.bush(x, y, r, BUSH)
            if rng.random() < 0.5:
                sc.spot(x + rng.normal(0, 0.4), y + rng.normal(0, 0.4), rng.uniform(0, math.pi), "bush")


# =============================================================================
# forest: dense trees, undergrowth, rocks, clearings, trails
# =============================================================================
F_CANOPY = [rgba(0.16, 0.38, 0.13), rgba(0.21, 0.44, 0.15), rgba(0.27, 0.47, 0.17)]
F_CONIFER = rgba(0.10, 0.28, 0.14)
F_TRUNK = rgba(0.32, 0.23, 0.15)
F_BUSH = rgba(0.20, 0.40, 0.16)
F_LOG = rgba(0.40, 0.30, 0.20)


def build_forest(sc: Scene):
    rng = sc.rng
    before = sc.occ.copy()
    clearings = []
    for _ in range(300):
        if len(clearings) >= 10:
            break
        x, y = rng.uniform(-125, 125, 2)
        R = rng.uniform(9, 20)
        if all(math.hypot(x - cx, y - cy) > R + cr + 14 for cx, cy, cr in clearings):
            clearings.append((x, y, R))
    for x, y, R in clearings:
        sc.mark_circle(x, y, R)
        sc.decal(x, y, R * 0.75, R * 0.75, rng.uniform(0, math.pi), rgba(0.36, 0.45, 0.22))

    trails = []
    for _ in range(3):
        side = int(rng.integers(4))
        t = rng.uniform(-120, 120)
        start = [(-150, t), (150, t), (t, -150), (t, 150)][side]
        heading = math.atan2(-start[1] + rng.uniform(-40, 40), -start[0] + rng.uniform(-40, 40))
        x, y = start
        pts = [(x, y)]
        while max(abs(x), abs(y)) <= 150.5 and len(pts) < 60:
            heading += rng.normal(0, 0.25)
            x, y = x + 12 * math.cos(heading), y + 12 * math.sin(heading)
            pts.append((x, y))
        trails.append(pts)
        for a, b in zip(pts[:-1], pts[1:]):
            L = math.hypot(b[0] - a[0], b[1] - a[1])
            yaw = math.atan2(b[1] - a[1], b[0] - a[0])
            mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
            if max(abs(mx), abs(my)) < 148:
                sc.decal(mx, my, L / 2 + 0.6, 1.8, yaw, rgba(0.42, 0.34, 0.24), z=0.03)
            sc.mark_rect(mx, my, L / 2 + 1.0, 2.6, yaw)
    zones = sc.occ & ~before

    grid = _PointGrid(5.2)
    deciduous = []
    for _ in range(9000):
        x, y = rng.uniform(-146.5, 146.5, 2)     # canopies (r <= 3.4) stay inside 150 m
        if not grid.ok(x, y) or not sc.free_circle(x, y, 0.6):
            continue
        grid.add(x, y)
        if rng.random() < 0.75:
            rc = rng.uniform(2.2, 3.3)
            hc = rc + rng.uniform(2.5, 4.0)
            tr = rng.uniform(0.18, 0.32)
            sc.tree(x, y, hc, rc, tr, F_CANOPY[int(rng.integers(3))], F_TRUNK)
            deciduous.append((x, y, tr))
        else:
            sc.conifer(x, y, rng.uniform(9, 14), rng.uniform(1.6, 2.3), F_CONIFER, F_TRUNK)
    sc.occ &= ~zones

    for _ in range(700):
        x, y = rng.uniform(-148, 148, 2)
        r = rng.uniform(0.5, 1.2)
        if sc.free_circle(x, y, r * 0.8):
            sc.bush(x, y, r, F_BUSH)
            if rng.random() < 0.15:
                sc.spot(x + rng.normal(0, 0.5), y + rng.normal(0, 0.5), rng.uniform(0, math.pi), "bush")
    for _ in range(140):
        x, y = rng.uniform(-145, 145, 2)
        r = rng.uniform(0.6, 1.6)
        if sc.free_circle(x, y, r + 0.3):
            sc.rock(x, y, r, ROCK)
            sc.spots_around(x, y, r + rng.uniform(0.4, 0.7) + TW, "rock", n=1)
    for _ in range(90):
        x, y = rng.uniform(-145, 145, 2)
        r, L, yaw = rng.uniform(0.22, 0.4), rng.uniform(3, 7), rng.uniform(0, math.pi)
        if sc.free_rect(x, y, L / 2 + r, r + 0.2, yaw):
            sc.lying("capsule", x, y, r, L, yaw, F_LOG, "log")
            c, s = math.cos(yaw), math.sin(yaw)
            sc.spots_along((x - c * L / 2, y - s * L / 2), (x + c * L / 2, y + s * L / 2),
                           r + rng.uniform(0.4, 0.7) + TW, "log", n=1, both=False)

    for x, y, R in clearings:
        for _ in range(8):
            a = rng.uniform(0, 2 * math.pi)
            rr = R - rng.uniform(-0.5, 2.5)
            sc.spot(x + rr * math.cos(a), y + rr * math.sin(a), a + math.pi / 2 + rng.normal(0, 0.2), "clearing-edge")
    for pts in trails:
        for a, b in zip(pts[:-1], pts[1:]):
            if rng.random() < 0.5:
                sc.spots_along(a, b, 2.0 + rng.uniform(0.3, 1.2) + TW, "trail", n=1)
    for k in rng.permutation(len(deciduous))[:250]:
        x, y, tr = deciduous[k]
        sc.spots_around(x, y, tr + rng.uniform(0.45, 0.8) + TW, "canopy", n=1)


# =============================================================================
# village: houses, sheds, fences, hedges, cars, gardens, streets
# =============================================================================
WALLS = [rgba(0.86, 0.83, 0.75), rgba(0.78, 0.70, 0.58), rgba(0.70, 0.63, 0.53)]
ROOFS = [rgba(0.55, 0.22, 0.16), rgba(0.33, 0.20, 0.15), rgba(0.30, 0.30, 0.33)]
CARS = [rgba(0.70, 0.10, 0.10), rgba(0.15, 0.25, 0.55), rgba(0.80, 0.80, 0.82), rgba(0.12, 0.12, 0.13), rgba(0.35, 0.45, 0.40)]
ASPHALT = rgba(0.26, 0.26, 0.28)


def _car_row(sc: Scene, a, b, prob, kind, cars=CARS, slot=6.2):
    """Parking lane from a to b: slots get a car with probability ``prob``; empty slots
    between two cars become 'among cars' candidates."""
    rng = sc.rng
    L = math.hypot(b[0] - a[0], b[1] - a[1])
    yaw = math.atan2(b[1] - a[1], b[0] - a[0])
    n = int(L // slot)
    occupied = [rng.random() < prob for _ in range(n)]
    for k in range(n):
        f = (k + 0.5) * slot / L
        x, y = a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f
        if occupied[k]:
            if sc.free_rect(x, y, 2.2, 0.95, yaw):
                sc.car(x, y, yaw + rng.normal(0, 0.03), cars[int(rng.integers(len(cars)))])
            else:
                occupied[k] = False
        elif 0 < k < n - 1 and occupied[k - 1] and rng.random() < 0.8:
            flip = math.pi if rng.random() < 0.5 else 0.0
            sc.spot(x, y, yaw + flip + rng.normal(0, 0.04), kind)


def build_village(sc: Scene):
    rng = sc.rng
    plots = Scene(rng)                                  # raster of claimed plots
    streets = [((-150.0, 0.0), (150.0, 0.0), 8.0)]
    for x in (-95, -25, 45, 115):
        x += rng.uniform(-6, 6)
        y0, y1 = (-150.0, 150.0) if rng.random() < 0.6 else ((-150.0, 0.0) if rng.random() < 0.5 else (0.0, 150.0))
        streets.append(((x, y0), (x, y1), 7.0))
    for y in (-72, 72):
        y += rng.uniform(-5, 5)
        streets.append(((-150.0, y), (150.0, y), 5.0))
    for a, b, w in streets:
        L = math.hypot(b[0] - a[0], b[1] - a[1])
        yaw = math.atan2(b[1] - a[1], b[0] - a[0])
        mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        sc.decal(mx, my, L / 2, w / 2, yaw, ASPHALT, z=0.03)
        plots.mark_rect(mx, my, L / 2, w / 2 + 0.3, yaw)

    for a, b, w in streets:
        L = math.hypot(b[0] - a[0], b[1] - a[1])
        yaw = math.atan2(b[1] - a[1], b[0] - a[0])
        ux, uy = math.cos(yaw), math.sin(yaw)
        nx, ny = -uy, ux
        # parked cars along both kerbs of the wider streets
        if w >= 7.0:
            for sgn in (1, -1):
                off = sgn * (w / 2 - 1.2)
                _car_row(sc, (a[0] + nx * off, a[1] + ny * off), (b[0] + nx * off, b[1] + ny * off), 0.35, "cars")
        for sgn in (1, -1):
            t = rng.uniform(0, 8)
            while t < L - 18:
                pw, pd = rng.uniform(18, 26), rng.uniform(22, 30)
                cx = a[0] + ux * (t + pw / 2) + sgn * nx * (w / 2 + 0.6 + pd / 2)
                cy = a[1] + uy * (t + pw / 2) + sgn * ny * (w / 2 + 0.6 + pd / 2)
                pyaw = yaw if sgn > 0 else yaw + math.pi        # local +y points away from street
                t += pw + rng.uniform(0, 3)
                if not plots.free_rect(cx, cy, pw / 2, pd / 2, pyaw):
                    continue
                plots.mark_rect(cx, cy, pw / 2, pd / 2, pyaw)
                if rng.random() < 0.85:
                    _plot(sc, cx, cy, pyaw, pw, pd)


def _plot(sc: Scene, cx, cy, yaw, pw, pd):
    rng = sc.rng
    c, s = math.cos(yaw), math.sin(yaw)

    def W(lx, ly):
        return cx + c * lx - s * ly, cy + s * lx + c * ly

    # local frame: x along the street, y away from the street; front edge at ly = -pd/2
    h = rng.uniform(4.2, 6.5)
    hw = rng.uniform(8, 12)
    hd = rng.uniform(6.5, min(10.0, 2 * (h - 0.3)))
    lx = rng.uniform(-pw / 2 + hw / 2 + 2.5, pw / 2 - hw / 2 - 2.5) if pw > hw + 5 else 0.0
    ly = -pd / 2 + rng.uniform(3.0, 5.5) + hd / 2
    hx, hy = W(lx, ly)
    if sc.free_rect(hx, hy, hw / 2 + 0.5, hd / 2 + 0.5, yaw):
        sc.house(hx, hy, yaw + rng.normal(0, 0.015), hw, hd, h, WALLS[int(rng.integers(3))], ROOFS[int(rng.integers(3))])
        sc.spots_box_sides(hx, hy, yaw, hw / 2, hd / 2, rng.uniform(0.45, 0.8), "house")
    # boundaries (sides and back), front fence with a gate gap
    corners = [W(-pw / 2, -pd / 2), W(pw / 2, -pd / 2), W(pw / 2, pd / 2), W(-pw / 2, pd / 2)]
    for (a, b, inward) in ((corners[1], corners[2], -1), (corners[2], corners[3], -1), (corners[3], corners[0], -1)):
        r = rng.random()
        if r < 0.5:
            fh = rng.uniform(1.1, 1.9)
            sc.fence(a, b, fh)
            sc.spots_along(a, b, 0.05 + rng.uniform(0.4, 0.7) + TW, "fence", n=1)
        elif r < 0.85:
            hh, hw2 = rng.uniform(1.6, 2.4), rng.uniform(0.9, 1.3)
            sc.hedge(a, b, hh, hw2)
            sc.spots_along(a, b, hw2 / 2 + rng.uniform(0.4, 0.7) + TW, "hedge", n=1)
    if rng.random() < 0.5:
        g = rng.uniform(-pw / 2 + 2, pw / 2 - 5)
        sc.fence(W(-pw / 2, -pd / 2), W(g, -pd / 2), 1.0)
        sc.fence(W(g + 3.2, -pd / 2), W(pw / 2, -pd / 2), 1.0)
    # back garden
    if rng.random() < 0.6:
        sw, sd, shh = rng.uniform(3, 4.5), rng.uniform(2.5, 3.5), rng.uniform(2.3, 2.8)
        sx, sy = W(rng.uniform(-pw / 2 + 3, pw / 2 - 3), pd / 2 - sd / 2 - rng.uniform(1.2, 2.5))
        if sc.free_rect(sx, sy, sw / 2 + 0.4, sd / 2 + 0.4, yaw):
            sc.shed(sx, sy, yaw, sw, sd, shh, rgba(0.45, 0.33, 0.22), rgba(0.25, 0.25, 0.27))
            sc.spots_box_sides(sx, sy, yaw, sw / 2, sd / 2, rng.uniform(0.4, 0.7), "shed")
    for _ in range(int(rng.integers(0, 4))):
        tx, ty = W(rng.uniform(-pw / 2 + 2, pw / 2 - 2), rng.uniform(0, pd / 2 - 2))
        if sc.free_circle(tx, ty, 1.2):
            tr = rng.uniform(0.2, 0.3)
            rc = rng.uniform(2.0, 2.8)
            sc.tree(tx, ty, rc + rng.uniform(2.0, 3.0), rc, tr, CANOPY[int(rng.integers(3))], TRUNK)
            sc.spots_around(tx, ty, tr + 0.5 + TW, "garden-tree", n=1)
    for _ in range(int(rng.integers(1, 6))):
        bx, by = W(rng.uniform(-pw / 2 + 1, pw / 2 - 1), rng.uniform(-pd / 2 + 1, pd / 2 - 1))
        r = rng.uniform(0.5, 1.0)
        if sc.free_circle(bx, by, r):
            sc.bush(bx, by, r, BUSH)
            if rng.random() < 0.3:
                sc.spot(bx, by, rng.uniform(0, math.pi), "bush")
    if rng.random() < 0.4:
        dx, dy = W(rng.choice([-1, 1]) * (pw / 2 - 2.0), -pd / 2 + 4.0)
        if sc.free_rect(dx, dy, 1.0, 2.3, yaw):
            sc.car(dx, dy, yaw + math.pi / 2, CARS[int(rng.integers(len(CARS)))])
            sc.spots_around(dx, dy, 2.2 + 0.5 + TW, "driveway-car", n=1)


# =============================================================================
# city: street grid, blocks, alleys, courtyards, parked cars, a park
# =============================================================================
BUILDINGS = [rgba(0.62, 0.60, 0.57), rgba(0.72, 0.66, 0.55), rgba(0.55, 0.36, 0.30), rgba(0.48, 0.50, 0.54)]
SIDEWALK = rgba(0.55, 0.55, 0.54)


def build_city(sc: Scene):
    rng = sc.rng
    centers = [-120.0, -60.0, 0.0, 60.0, 120.0]
    half = 23.0
    blocks = [(bx, by) for bx in centers for by in centers]
    park = blocks[int(rng.choice([6, 8, 16, 18]))]
    towers = set()
    while len(towers) < 3:
        b = blocks[int(rng.integers(len(blocks)))]
        if b != park:
            towers.add(b)
    for bx, by in blocks:
        sc.decal(bx, by, half + 1.5, half + 1.5, 0.0, SIDEWALK, z=0.03)
    # parked cars first along every kerb so the blocks do not claim the lanes
    for bx, by in blocks:
        off = half + 2.6
        for (a, b) in (((bx - half + 3, by - off), (bx + half - 3, by - off)),
                       ((bx - half + 3, by + off), (bx + half - 3, by + off)),
                       ((bx - off, by - half + 3), (bx - off, by + half - 3)),
                       ((bx + off, by - half + 3), (bx + off, by + half - 3))):
            _car_row(sc, a, b, 0.55, "parked-cars")
    for bx, by in blocks:
        if (bx, by) == park:
            _park(sc, bx, by, half)
        elif (bx, by) in towers:
            _tower_block(sc, bx, by, half)
        else:
            _perimeter_block(sc, bx, by, half)


def _bcolor(rng):
    return BUILDINGS[int(rng.integers(len(BUILDINGS)))]


def _perimeter_block(sc: Scene, bx, by, half):
    rng = sc.rng
    D = rng.uniform(10.5, 13.0)
    inner = half - D
    # each side: a row of buildings along the edge, with alleys into the courtyard
    sides = [  # (row centre offset, along axis is x?)
        ("S", (0.0, -half + D / 2), 0.0, half),
        ("N", (0.0, half - D / 2), 0.0, half),
        ("W", (-half + D / 2, 0.0), math.pi / 2, inner),
        ("E", (half - D / 2, 0.0), math.pi / 2, inner),
    ]
    alley_sides = set(rng.choice(["S", "N", "W", "E"], size=int(rng.integers(1, 3)), replace=False))
    for name, (ox, oy), yaw, hl in sides:
        c, s = math.cos(yaw), math.sin(yaw)
        cuts = [-hl]
        aw = rng.uniform(5.6, 7.0)
        if name in alley_sides and hl - 5 - aw > -hl + 5:
            m = rng.uniform(-hl + 5, hl - 5 - aw)
            cuts += [m, m + aw]
            # alley spot: box along the alley axis, against one wall
            ax_ = (m + aw / 2)
            side = rng.choice([-1, 1])
            lx = ax_ + side * (aw / 2 - 0.45 - TW)
            for ly in (rng.uniform(-D / 2 + 2.2, D / 2 - 2.2),):
                px, py = bx + ox + c * lx - s * ly, by + oy + s * lx + c * ly
                sc.spot(px, py, yaw + math.pi / 2 + (math.pi if rng.random() < 0.5 else 0), "alley")
        cuts.append(hl)
        segs = [(cuts[i], cuts[i + 1]) for i in range(0, len(cuts), 2)]
        for s0, s1 in segs:
            # split a solid run into 1-3 buildings of different heights (touching)
            n = int(rng.integers(1, 4))
            edges = np.linspace(s0, s1, n + 1)
            for e0, e1 in zip(edges[:-1], edges[1:]):
                if e1 - e0 < 1.0:
                    continue
                m = (e0 + e1) / 2
                px, py = bx + ox + c * m, by + oy + s * m
                hgt = rng.uniform(12, 42)
                sc.ground_box(px, py, (e1 - e0) / 2, D / 2, hgt, yaw, _bcolor(rng), "building")
                if rng.random() < 0.4:
                    sc.ground_box(px, py, 1.2, 1.0, 1.2, yaw, rgba(0.4, 0.4, 0.42), "rooftop", z0=hgt, mark=False)
    # courtyard
    for _ in range(int(rng.integers(0, 3))):
        tx, ty = bx + rng.uniform(-inner + 3, inner - 3), by + rng.uniform(-inner + 3, inner - 3)
        if sc.free_circle(tx, ty, 1.5):
            sc.tree(tx, ty, rng.uniform(4, 5.5), rng.uniform(2, 2.8), 0.25, CANOPY[int(rng.integers(3))], TRUNK)
    for _ in range(int(rng.integers(1, 4))):
        x, y = bx + rng.uniform(-inner + 1.5, inner - 1.5), by + rng.uniform(-inner + 1.5, inner - 1.5)
        if sc.free_rect(x, y, 1.0, 0.6, 0.0):
            sc.ground_box(x, y, 0.9, 0.5, 1.3, rng.choice([0.0, math.pi / 2]), rgba(0.25, 0.32, 0.28), "bins")
    if rng.random() < 0.5:
        x, y = bx + rng.uniform(-inner + 3, inner - 3), by + rng.uniform(-inner + 3, inner - 3)
        yaw = rng.choice([0.0, math.pi / 2])
        if sc.free_rect(x, y, 2.4, 1.1, yaw):
            sc.car(x, y, yaw, CARS[int(rng.integers(len(CARS)))])
    # courtyard spots against the inner facades
    for (lx, ly, yaw) in ((0.0, -inner, 0.0), (0.0, inner, 0.0), (-inner, 0.0, math.pi / 2), (inner, 0.0, math.pi / 2)):
        for _ in range(2):
            slide = rng.uniform(-inner + 2.5, inner - 2.5)
            nx, ny = (0.0, -np.sign(ly)) if yaw == 0.0 else (-np.sign(lx), 0.0)
            off = 0.45 + TW
            px = bx + (slide if yaw == 0.0 else lx) + nx * off
            py = by + (ly if yaw == 0.0 else slide) + ny * off
            sc.spot(px, py, yaw + (math.pi if rng.random() < 0.5 else 0.0), "courtyard")


def _tower_block(sc: Scene, bx, by, half):
    rng = sc.rng
    n = int(rng.integers(1, 3))
    for k in range(n):
        w, d = rng.uniform(14, 20), rng.uniform(14, 20)
        ox = 0.0 if n == 1 else (-1 if k == 0 else 1) * (half - w / 2 - 1.5)
        oy = rng.uniform(-half + d / 2 + 1.5, half - d / 2 - 1.5)
        x, y = bx + ox, by + oy
        if not sc.free_rect(x, y, w / 2, d / 2, 0.0):
            continue
        sc.ground_box(x, y, w / 2, d / 2, rng.uniform(40, 70), 0.0, _bcolor(rng), "tower")
        sc.spots_box_sides(x, y, 0.0, w / 2, d / 2, rng.uniform(0.45, 0.8), "tower")
    for _ in range(10):
        x, y = bx + rng.uniform(-half + 2, half - 2), by + rng.uniform(-half + 2, half - 2)
        yaw = rng.choice([0.0, math.pi / 2])
        if sc.free_rect(x, y, 2.5, 1.5, yaw, margin=0.6):
            sc.ground_box(x, y, 2.0, 0.7, 0.8, yaw, rgba(0.45, 0.44, 0.42), "planter")
            sc.bush(x + rng.uniform(-1, 1), y, 0.7, BUSH)
            sc.spots_box_sides(x, y, yaw, 2.0, 0.7, rng.uniform(0.4, 0.6), "planter")


def _park(sc: Scene, bx, by, half):
    rng = sc.rng
    sc.decal(bx, by, half, half, 0.0, rgba(0.34, 0.52, 0.26), z=0.04)
    sc.decal(bx, by, half, 1.6, rng.uniform(0, math.pi), rgba(0.62, 0.57, 0.47), z=0.05)
    px, py = bx + rng.uniform(-10, 10), by + rng.uniform(-10, 10)
    sc.decal(px, py, 5.0, 3.5, rng.uniform(0, math.pi), rgba(0.25, 0.40, 0.55), z=0.05)
    sc.mark_rect(px, py, 5.0, 4.0, 0.0)
    sc.ground_box(bx + rng.uniform(-15, 15), by + rng.uniform(-15, 15), 2.0, 1.6, 2.6, 0.0, rgba(0.55, 0.42, 0.30), "kiosk")
    grid = _PointGrid(6.0)
    for _ in range(200):
        x, y = bx + rng.uniform(-half + 2, half - 2), by + rng.uniform(-half + 2, half - 2)
        if grid.ok(x, y) and sc.free_circle(x, y, 1.5):
            grid.add(x, y)
            tr = rng.uniform(0.25, 0.35)
            rc = rng.uniform(2.6, 3.6)
            sc.tree(x, y, rc + rng.uniform(2.2, 3.2), rc, tr, CANOPY[int(rng.integers(3))], TRUNK)
            sc.spots_around(x, y, tr + 0.55 + TW, "park-tree", n=1)
    for _ in range(40):
        x, y = bx + rng.uniform(-half + 1, half - 1), by + rng.uniform(-half + 1, half - 1)
        r = rng.uniform(0.6, 1.2)
        if sc.free_circle(x, y, r):
            sc.bush(x, y, r, BUSH)
            if rng.random() < 0.4:
                sc.spot(x, y, rng.uniform(0, math.pi), "bush")


# =============================================================================
# rubble: collapsed walls, debris piles, wrecked cars, craters
# =============================================================================
CONCRETE = [rgba(0.58, 0.56, 0.53), rgba(0.50, 0.47, 0.44), rgba(0.62, 0.52, 0.44)]
DEBRIS = rgba(0.47, 0.44, 0.40)
BURNT = [rgba(0.22, 0.20, 0.19), rgba(0.35, 0.25, 0.20), rgba(0.30, 0.31, 0.33)]
CHAR = rgba(0.18, 0.15, 0.13)


def _tilt(rng, lo, hi):
    a = rng.uniform(0, 2 * math.pi)
    return quat_from_axis_angle((math.cos(a), math.sin(a), 0.0), rng.uniform(lo, hi))


def _debris_pile(sc: Scene, x, y, r):
    rng = sc.rng
    sc.sphere(x, y, -0.45 * r, r, DEBRIS, "debris")
    for _ in range(int(rng.integers(5, 11))):
        a, d = rng.uniform(0, 2 * math.pi), rng.uniform(0, 0.85 * r)
        px, py = x + d * math.cos(a), y + d * math.sin(a)
        size = rng.uniform(0.3, 1.1, 3) * [1.2, 1.0, 0.5]
        zc = math.sqrt(max(r * r - d * d, 0.0)) - 0.45 * r
        q = quat_from_axis_angle(rng.normal(size=3), rng.uniform(0, math.pi))
        sc.box((px, py, max(zc, size[2] * 0.5)), size, quat=q, color=CONCRETE[int(rng.integers(3))], tag="debris", mark=False)
    sc.mark_circle(x, y, r * 0.9)
    sc.spots_around(x, y, 0.9 * r + rng.uniform(0.5, 0.9) + TW, "debris", n=2)


def _shell(sc: Scene, x, y):
    rng = sc.rng
    w, d = rng.uniform(10, 16), rng.uniform(8, 12)
    yaw = rng.uniform(0, math.pi)
    if not sc.free_rect(x, y, w / 2 + 1, d / 2 + 1, yaw):
        return
    c, s = math.cos(yaw), math.sin(yaw)

    def W(lx, ly):
        return x + c * lx - s * ly, y + s * lx + c * ly

    col = CONCRETE[int(rng.integers(3))]
    for (a, b) in (((-w / 2, -d / 2), (w / 2, -d / 2)), ((w / 2, -d / 2), (w / 2, d / 2)),
                   ((w / 2, d / 2), (-w / 2, d / 2)), ((-w / 2, d / 2), (-w / 2, -d / 2))):
        n = int(rng.integers(2, 4))
        for k in range(n):
            if rng.random() < 0.25:
                continue
            f0, f1 = k / n, (k + 1) / n
            pa = W(a[0] + (b[0] - a[0]) * f0, a[1] + (b[1] - a[1]) * f0)
            pb = W(a[0] + (b[0] - a[0]) * f1, a[1] + (b[1] - a[1]) * f1)
            sc.segment(pa, pb, rng.uniform(2.2, 7.0), 0.4, col, "wall")
            sc.spots_along(pa, pb, 0.2 + rng.uniform(0.4, 0.7) + TW, "wall", n=1)
    if rng.random() < 0.5:
        slab = [("box", (w * 0.22, d * 0.3, 0.15), (0, 0, 0), col)]
        sc.group(slab, *W(rng.uniform(-w / 5, w / 5), rng.uniform(-d / 5, d / 5)), yaw, _tilt(rng, 0.25, 0.6), "slab", mark=False)
    if rng.random() < 0.6:
        _debris_pile(sc, *W(rng.uniform(-w / 4, w / 4), rng.uniform(-d / 4, d / 4)), rng.uniform(1.5, 2.5))


def _collapsed(sc: Scene, x, y):
    rng = sc.rng
    for _ in range(int(rng.integers(2, 5))):
        px, py = x + rng.uniform(-6, 6), y + rng.uniform(-6, 6)
        hx, hy = rng.uniform(1.5, 3.0), rng.uniform(1.2, 2.0)
        yaw = rng.uniform(0, math.pi)
        if sc.free_rect(px, py, hx, hy, yaw):
            sc.group([("box", (hx, hy, 0.13), (0, 0, 0), CONCRETE[int(rng.integers(3))])], px, py, yaw,
                     _tilt(rng, 0.17, 0.7), "slab", sink=0.02)
            sc.spots_box_sides(px, py, yaw, hx, hy, rng.uniform(0.6, 1.0), "slab")
    for _ in range(int(rng.integers(1, 3))):
        px, py = x + rng.uniform(-7, 7), y + rng.uniform(-7, 7)
        r = rng.uniform(1.5, 3.2)
        if sc.free_circle(px, py, r):
            _debris_pile(sc, px, py, r)
    for _ in range(2):
        px, py = x + rng.uniform(-8, 8), y + rng.uniform(-8, 8)
        yaw, L = rng.uniform(0, math.pi), rng.uniform(4, 8)
        a = (px - math.cos(yaw) * L / 2, py - math.sin(yaw) * L / 2)
        b = (px + math.cos(yaw) * L / 2, py + math.sin(yaw) * L / 2)
        if sc.free_rect(px, py, L / 2, 0.4, yaw):
            sc.segment(a, b, rng.uniform(1.0, 2.6), 0.4, CONCRETE[int(rng.integers(3))], "wall")
            sc.spots_along(a, b, 0.2 + rng.uniform(0.4, 0.7) + TW, "wall-stub", n=1)


def _crater(sc: Scene, x, y):
    rng = sc.rng
    R = rng.uniform(4.5, 7.0)
    if not sc.free_circle(x, y, R + 2):
        return
    for k in range(2):
        sc.decal(x, y, R * 0.8, R * 0.8, k * math.pi / 4, CHAR, z=0.03 + 0.005 * k)
    n = int(rng.integers(10, 15))
    for k in range(n):
        a = 2 * math.pi * k / n + rng.normal(0, 0.15)
        rr = R + rng.uniform(0, 1.5)
        px, py = x + rr * math.cos(a), y + rr * math.sin(a)
        if rng.random() < 0.5:
            sc.rock(px, py, rng.uniform(0.6, 1.3), DEBRIS)
        else:
            size = rng.uniform(0.5, 1.4, 3) * [1.4, 1.0, 0.6]
            sc.group([("box", tuple(size), (0, 0, 0), CONCRETE[int(rng.integers(3))])], px, py, a,
                     _tilt(rng, 0.1, 0.6), "debris", sink=0.1)
    sc.spot(x, y, rng.uniform(0, math.pi), "crater")
    sc.spot(x + rng.uniform(-1.5, 1.5), y + rng.uniform(-1.5, 1.5), rng.uniform(0, math.pi), "crater")


def _wrecks(sc: Scene, x, y):
    rng = sc.rng
    for _ in range(int(rng.integers(3, 7))):
        px, py = x + rng.uniform(-8, 8), y + rng.uniform(-8, 8)
        yaw = rng.uniform(0, 2 * math.pi)
        if not sc.free_rect(px, py, 2.4, 1.3, yaw):
            continue
        r = rng.random()
        if r < 0.5:
            tilt = _tilt(rng, 0.0, 0.25)
        elif r < 0.75:
            tilt = quat_from_axis_angle((1, 0, 0), math.pi / 2 + rng.normal(0, 0.1))
        else:
            tilt = quat_from_axis_angle((1, 0, 0), math.pi + rng.normal(0, 0.1))
        sc.car(px, py, yaw, BURNT[int(rng.integers(3))], tilt=tilt, tag="wreck")
        sc.spots_around(px, py, 2.3 + rng.uniform(0.5, 0.9) + TW, "wrecks", n=1)


def _scorched(sc: Scene, x, y):
    rng = sc.rng
    for _ in range(int(rng.integers(2, 6))):
        px, py = x + rng.uniform(-9, 9), y + rng.uniform(-9, 9)
        if not sc.free_circle(px, py, 1.0):
            continue
        if rng.random() < 0.7:
            sc.cylinder(px, py, 0.0, rng.uniform(0.2, 0.35), rng.uniform(3, 7), CHAR, "burnt-tree", caps=True)
            sc.spots_around(px, py, 0.35 + 0.6 + TW, "burnt-tree", n=1)
        else:
            yaw = rng.uniform(0, math.pi)
            sc.lying("capsule", px, py, rng.uniform(0.2, 0.35), rng.uniform(3, 6), yaw, CHAR, "burnt-tree")
    for _ in range(int(rng.integers(3, 8))):
        px, py = x + rng.uniform(-10, 10), y + rng.uniform(-10, 10)
        size = rng.uniform(0.3, 0.9, 3)
        if sc.free_circle(px, py, 1.0):
            sc.group([("box", tuple(size), (0, 0, 0), CONCRETE[int(rng.integers(3))])], px, py,
                     rng.uniform(0, math.pi), _tilt(rng, 0, 0.7), "debris", sink=0.05)


def build_rubble(sc: Scene):
    rng = sc.rng
    for k in range(2):
        yaw = rng.uniform(0, math.pi)
        sc.decal(rng.uniform(-60, 60), rng.uniform(-60, 60), 150, 4.5, yaw, rgba(0.38, 0.36, 0.33), z=0.03 + 0.005 * k)
    kinds = ["shell", "collapsed", "crater", "wrecks", "debris", "scorched"]
    probs = [0.30, 0.22, 0.13, 0.14, 0.11, 0.10]
    for gx in np.arange(-130, 131, 26.0):
        for gy in np.arange(-130, 131, 26.0):
            x, y = gx + rng.uniform(-4, 4), gy + rng.uniform(-4, 4)
            kind = kinds[int(rng.choice(len(kinds), p=probs))]
            if kind == "shell":
                _shell(sc, x, y)
            elif kind == "collapsed":
                _collapsed(sc, x, y)
            elif kind == "crater":
                _crater(sc, x, y)
            elif kind == "wrecks":
                _wrecks(sc, x, y)
            elif kind == "debris":
                for _ in range(int(rng.integers(1, 4))):
                    px, py = x + rng.uniform(-6, 6), y + rng.uniform(-6, 6)
                    r = rng.uniform(1.6, 3.5)
                    if sc.free_circle(px, py, r):
                        _debris_pile(sc, px, py, r)
            else:
                _scorched(sc, x, y)


# =============================================================================
# empty: flat ground, no props (testing and calibration only)
# =============================================================================
def build_empty(sc: Scene):
    rng = sc.rng
    for i in range(7):
        for j in range(7):
            sc.spot(-90 + 30 * i, -90 + 30 * j, rng.uniform(0, math.pi), "open")


BUILDERS = {
    "field": (build_field, rgba(0.42, 0.55, 0.28)),
    "forest": (build_forest, rgba(0.30, 0.36, 0.20)),
    "village": (build_village, rgba(0.40, 0.55, 0.30)),
    "city": (build_city, rgba(0.36, 0.36, 0.37)),
    "rubble": (build_rubble, rgba(0.46, 0.43, 0.39)),
    "empty": (build_empty, rgba(0.42, 0.50, 0.32)),
}
