"""Film one flight as an MP4: chase camera, onboard camera with the detector box, depth, minimap.

    python tools/video_flight.py --agent handin --terrain rubble@v8 --seed 0 --run 0 --out flight.mp4

Replays the flight exactly as `rescuesim.evaluate` flies it (same seed and run), then draws each
control step (30 fps, real time). Needs ffmpeg on PATH and Pillow. Dev tool, not part of the sim.
"""
from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys

import numpy as np
import pybullet as p
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rescuesim import RescueEnv  # noqa: E402
from rescuesim import sensors as S  # noqa: E402
from rescuesim.agent_loader import load_agent_class  # noqa: E402

W, H = 1280, 720
MAIN_W = 960
SIDE_W = 320
CAM_H = 180
MAP_PX = 320
MAP_HALF = 200.0           # metres shown on each side of the minimap centre
VFOV_DEG = math.degrees(2 * math.atan(math.tan(math.radians(102.0) / 2) * 9 / 16))
PHASE_FR = {"init": "INITIALISATION", "search": "RECHERCHE", "approach": "APPROCHE",
            "terminal": "CONTACT FINAL"}


def _font(size):
    for name in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def depth_rgb(depth):
    """Depth (126 x 224, m) -> near warm / far dark colour image."""
    d = np.clip(np.asarray(depth, float), 0, 40) / 40.0
    t = 1.0 - d
    r = np.clip(1.6 * t, 0, 1)
    g = np.clip(1.6 * t - 0.4, 0, 1)
    b = np.clip(0.5 - 0.8 * np.abs(t - 0.35), 0, 1)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


def add_sky(rgb):
    """TinyRenderer leaves empty pixels pure white: paint them with a sky gradient."""
    rgb = rgb.copy()
    mask = np.all(rgb == 255, axis=-1)
    h = rgb.shape[0]
    t = np.linspace(0, 1, h)[:, None]
    top, horizon = np.array([92, 140, 200]), np.array([200, 220, 235])
    grad = (top * (1 - t) + horizon * t).astype(np.uint8)
    sky = np.broadcast_to(grad[:, None, :], rgb.shape)
    rgb[mask] = sky[mask]
    return rgb


def draw_map(env, cx, cy, to_map):
    """Top-down minimap from the terrain's primitives (footprints in their own colours)."""
    g = env.terrain.layout.ground_rgba
    img = Image.new("RGB", (MAP_PX, MAP_PX), tuple(int(255 * c * 0.85) for c in g[:3]))
    d = ImageDraw.Draw(img)
    prims = sorted(env.terrain.prims, key=lambda pr: pr.aabb()[1][2])
    for pr in prims:
        lo, hi = pr.aabb()
        x0, y1 = to_map(lo[0], lo[1], cx, cy)
        x1, y0 = to_map(hi[0], hi[1], cx, cy)
        if x1 < 0 or y1 < 0 or x0 > MAP_PX or y0 > MAP_PX:
            continue
        col = tuple(int(255 * c) for c in pr.color[:3])
        d.rectangle([x0, y0, max(x1, x0 + 1), max(y1, y0 + 1)], fill=col)
    return img


class DroneMarker:
    """A visual-only quadcopter (body + 4 rotors) in the render world, parked underground while sensors render."""

    def __init__(self, cid):
        self.cid = cid
        arm = 0.32
        shapes = [p.GEOM_BOX] + [p.GEOM_CYLINDER] * 4
        half = [[0.22, 0.14, 0.07]] + [[0, 0, 0]] * 4
        radii = [0] + [0.2] * 4
        lengths = [0] + [0.02] * 4
        pos = [[0, 0, 0]] + [[sx * arm, sy * arm, 0.04] for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1))]
        colors = [[0.08, 0.08, 0.1, 1]] + [[0.9, 0.1, 0.1, 1]] * 4
        vs = p.createVisualShapeArray(shapeTypes=shapes, halfExtents=half, radii=radii, lengths=lengths,
                                      visualFramePositions=pos, rgbaColors=colors, physicsClientId=cid) \
            if hasattr(p, "createVisualShapeArray") else -1
        if vs < 0:
            vs = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.05], rgbaColor=[0.9, 0.1, 0.1, 1],
                                     physicsClientId=cid)
        self.body = p.createMultiBody(0, -1, vs, basePosition=[0, 0, -80], physicsClientId=cid)

    def place(self, pos, quat):
        p.resetBasePositionAndOrientation(self.body, list(pos), list(quat), physicsClientId=self.cid)

    def hide(self):
        p.resetBasePositionAndOrientation(self.body, [0, 0, -80], [0, 0, 0, 1], physicsClientId=self.cid)


def fly(agent_dir, terrain, seed, run, timeout):
    Agent = load_agent_class(agent_dir)
    env = RescueEnv(terrain, max_time_s=timeout)
    agent = Agent()
    obs, info = env.reset(seed=seed, options={"run": run})
    agent.reset()
    yield env, agent, obs, info
    while info["outcome"] is None:
        try:
            action = agent.act(obs)
        except Exception:
            env.fail("error")
            break
        obs, _, _, _, info = env.step(action)
        yield env, agent, obs, info


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--agent", default="handin")
    ap.add_argument("--terrain", default="rubble@v8")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--out", default="flight.mp4")
    ap.add_argument("--hold", type=float, default=3.0, help="seconds to hold the last frame")
    args = ap.parse_args()

    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{W}x{H}", "-r", "30", "-i", "-", "-c:v", "libx264", "-preset", "medium",
                           "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", args.out],
                          stdin=subprocess.PIPE)
    f_big, f_mid, f_small = _font(30), _font(20), _font(15)
    path_xy, cam_eye, cam_look = [], None, None
    marker = minimap = tgt_px = None
    frame = None
    first_det = None

    def to_map(x, y, cx, cy):
        return ((x - cx) / MAP_HALF * 0.5 + 0.5) * MAP_PX, (0.5 - (y - cy) / MAP_HALF * 0.5) * MAP_PX

    for env, agent, obs, info in fly(args.agent, args.terrain, args.seed, args.run, args.timeout):
        rw = env.rworld
        st = env.quad.state
        pos, vel = np.asarray(st.pos, float), np.asarray(st.vel, float)
        if marker is None:
            marker = DroneMarker(rw.cid)
            tgt = env.target
            mcx, mcy = (tgt.x + pos[0]) / 2, (tgt.y + pos[1]) / 2
            mcx = float(np.clip(mcx, -330 + MAP_HALF, 330 - MAP_HALF))
            mcy = float(np.clip(mcy, -330 + MAP_HALF, 330 - MAP_HALF))
            minimap = draw_map(env, mcx, mcy, to_map)
            tgt_px = to_map(tgt.x, tgt.y, mcx, mcy)
            map_c = (mcx, mcy)
        path_xy.append(to_map(pos[0], pos[1], *map_c))
        if obs["detected"] and first_det is None:
            first_det = info["t"]

        # Chase camera: behind the flight direction, smoothed.
        sp = float(np.linalg.norm(vel[:2]))
        R = np.array(p.getMatrixFromQuaternion(list(st.quat))).reshape(3, 3)
        heading = vel[:2] / sp if sp > 1.0 else R[:2, 0] / max(1e-6, np.linalg.norm(R[:2, 0]))
        back = 7.0 + 0.25 * sp
        want_eye = pos + np.array([-heading[0] * back, -heading[1] * back, 2.5 + 0.08 * sp])
        want_look = pos + np.array([heading[0] * 6.0, heading[1] * 6.0, -0.5])
        a = 0.15 if cam_eye is not None else 1.0
        cam_eye = want_eye if cam_eye is None else (1 - a) * cam_eye + a * want_eye
        cam_look = want_look if cam_look is None else (1 - a) * cam_look + a * want_look
        cam_eye[2] = max(cam_eye[2], 1.0)
        marker.place(pos, st.quat)
        chase = add_sky(rw.render_rgb(cam_eye, cam_look, [0, 0, 1], MAIN_W, H, fov_deg=62.0, far=400.0))
        # Onboard camera (same pose as the sensors).
        eye, fwd, up, _right = env.camera()
        marker.hide()
        onboard = add_sky(rw.render_rgb(eye, eye + fwd, up, SIDE_W, CAM_H, fov_deg=VFOV_DEG, far=400.0))

        img = Image.new("RGB", (W, H), (12, 14, 18))
        img.paste(Image.fromarray(chase), (0, 0))
        ob = Image.fromarray(onboard)
        d = ImageDraw.Draw(ob)
        if obs["detected"]:
            x1, y1, x2, y2 = [float(v) for v in obs["box"]]
            bx = [x1 * SIDE_W, y1 * CAM_H, x2 * SIDE_W, y2 * CAM_H]
            pad = 4
            d.rectangle([bx[0] - pad, bx[1] - pad, bx[2] + pad, bx[3] + pad], outline=(0, 255, 120), width=2)
            d.text((bx[0] - pad, max(0, bx[1] - pad - 16)), "CIBLE", fill=(0, 255, 120), font=f_small)
        img.paste(ob, (MAIN_W, 0))
        img.paste(Image.fromarray(depth_rgb(obs["depth_m"])).resize((SIDE_W, CAM_H), Image.NEAREST),
                  (MAIN_W, CAM_H))
        mm = minimap.copy()
        dm = ImageDraw.Draw(mm)
        if len(path_xy) > 1:
            dm.line(path_xy, fill=(255, 60, 60), width=2)
        tx, ty = tgt_px
        dm.ellipse([tx - 6, ty - 6, tx + 6, ty + 6], outline=(255, 150, 0), width=3)
        dx, dy = path_xy[-1]
        dm.ellipse([dx - 4, dy - 4, dx + 4, dy + 4], fill=(255, 255, 255))
        img.paste(mm, (MAIN_W, 2 * CAM_H))

        d = ImageDraw.Draw(img)
        for y, label in ((0, "CAMÉRA EMBARQUÉE"), (CAM_H, "PROFONDEUR (0–40 m)"), (2 * CAM_H, "CARTE")):
            d.rectangle([MAIN_W, y, MAIN_W + SIDE_W - 1, y + 20], fill=(0, 0, 0))
            d.text((MAIN_W + 6, y + 2), label, fill=(220, 220, 220), font=f_small)
        d.line([MAIN_W, 0, MAIN_W, H], fill=(40, 44, 52), width=2)
        phase = PHASE_FR.get(getattr(agent, "phase", ""), str(getattr(agent, "phase", "")).upper())
        d.rectangle([0, 0, MAIN_W, 46], fill=(0, 0, 0))
        d.text((14, 8), f"OSTROZKY · {args.terrain}", fill=(255, 255, 255), font=f_mid)
        ph_col = (0, 230, 120) if phase in ("APPROCHE", "CONTACT FINAL") else (255, 200, 60)
        d.text((MAIN_W - 260, 8), phase, fill=ph_col, font=f_mid)
        d.rectangle([0, H - 40, MAIN_W, H], fill=(0, 0, 0))
        hud = (f"t {info['t']:5.1f} s    vitesse {np.linalg.norm(vel) * 3.6:5.1f} km/h    "
               f"altitude {pos[2]:5.1f} m    distance cible {info['distance_m']:6.1f} m    "
               f"cible {'VUE' if obs['detected'] else '—'}")
        d.text((14, H - 32), hud, fill=(230, 230, 230), font=f_small)
        d.text((MAIN_W + 6, H - 40 + 22), "Aucun pilote · aucune liaison radio", fill=(150, 150, 150), font=f_small)
        frame = img
        ff.stdin.write(img.tobytes())

    if frame is not None:
        out = info["outcome"]
        end = frame.copy()
        d = ImageDraw.Draw(end)
        ok = out == "success"
        msg = "CIBLE ATTEINTE" if ok else out.upper()
        col = (0, 230, 120) if ok else (255, 80, 80)
        d.rectangle([MAIN_W // 2 - 300, H // 2 - 70, MAIN_W // 2 + 300, H // 2 + 70], fill=(0, 0, 0))
        d.text((MAIN_W // 2 - 280, H // 2 - 55), msg, fill=col, font=_font(56))
        sub = f"{info['t']:.1f} s de vol" + (f" · cible repérée à {first_det:.1f} s" if first_det is not None else "")
        d.text((MAIN_W // 2 - 280, H // 2 + 20), sub, fill=(220, 220, 220), font=f_mid)
        for _ in range(int(args.hold * 30)):
            ff.stdin.write(end.tobytes())
        print(f"{args.terrain}: {out} after {info['t']:.1f} s")
    ff.stdin.close()
    ff.wait()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
