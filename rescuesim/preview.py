"""Top-down PNG previews of terrain layouts.

    python -m rescuesim.preview [--terrains field forest ...] [--layout-seed S] [--out previews]

Uses matplotlib when installed (footprints drawn from the primitives, poses
labelled); otherwise falls back to an orthographic TinyRenderer image written
with a minimal PNG encoder.
"""
from __future__ import annotations

import argparse
import os
import struct
import zlib

import numpy as np

from .geometry import box_corners
from .scene import TARGET_HALF
from .terrain import TERRAIN_TYPES, get_layout


def _hull(pts: np.ndarray) -> np.ndarray:
    pts = np.unique(np.round(pts, 4), axis=0)
    if len(pts) < 3:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for q in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], q) <= 0:
            lower.pop()
        lower.append(tuple(q))
    for q in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], q) <= 0:
            upper.pop()
        upper.append(tuple(q))
    return np.array(lower[:-1] + upper[:-1])


def write_png(path: str, rgb: np.ndarray):
    """Minimal 8-bit RGB PNG writer (no dependencies)."""
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[i].astype(np.uint8).tobytes() for i in range(h))

    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)))
        f.write(chunk(b"IDAT", zlib.compress(raw, 6)))
        f.write(chunk(b"IEND", b""))


def preview_matplotlib(kind: str, layout_seed, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    from .world import prim_mesh

    lay = get_layout(kind, layout_seed)
    fig, ax = plt.subplots(figsize=(11, 11), dpi=150)
    ax.set_facecolor(lay.ground_rgba[:3])
    order = sorted(lay.prims, key=lambda pr: (pr.shape != "quad", pr.aabb()[1][2]))
    for pr in order:
        v, _ = prim_mesh(pr)
        top = pr.aabb()[1][2]
        shade = 1.0 if pr.shape == "quad" else min(1.0, 0.75 + 0.25 * min(top, 30.0) / 30.0)
        col = np.clip(np.asarray(pr.color[:3]) * shade, 0, 1)
        ax.add_patch(Polygon(_hull(v[:, :2]), closed=True, facecolor=col, edgecolor="none"))
    for k, pose in enumerate(lay.poses, start=1):
        c = box_corners(pose.center, TARGET_HALF, pose.yaw)[[0, 2, 6, 4], :2]
        ax.add_patch(Polygon(c, closed=True, facecolor=(1, 0.45, 0), edgecolor="k", linewidth=0.5))
        ax.text(pose.x + 2.2, pose.y + 2.2, f"v{k}", fontsize=6, color="k",
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7))
    ax.set_xlim(-155, 155)
    ax.set_ylim(-155, 155)
    ax.set_aspect("equal")
    ax.set_title(f"{kind} (layout {lay.layout_seed}): {len(lay.prims)} primitives, {len(lay.poses)} target poses")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def preview_render(kind: str, layout_seed, path: str, size: int = 1024):
    from .world import RenderWorld

    lay = get_layout(kind, layout_seed)
    rw = RenderWorld(list(lay.prims), lay.ground_rgba)
    try:
        rgb = rw.render_rgb((0, 0, 300), (0, 0, 0), (0, 1, 0), size, size, far=400.0, ortho_half=155.0)
    finally:
        rw.close()
    write_png(path, rgb)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--terrains", nargs="+", default=list(TERRAIN_TYPES))
    ap.add_argument("--layout-seed", type=int, default=None)
    ap.add_argument("--out", default="previews")
    ap.add_argument("--renderer", choices=["auto", "matplotlib", "pybullet"], default="auto")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    use_mpl = args.renderer == "matplotlib"
    if args.renderer == "auto":
        try:
            import matplotlib  # noqa: F401
            use_mpl = True
        except ImportError:
            use_mpl = False
    for kind in args.terrains:
        path = os.path.join(args.out, f"{kind}.png")
        (preview_matplotlib if use_mpl else preview_render)(kind, args.layout_seed, path)
        print("wrote", path)


if __name__ == "__main__":
    main()
