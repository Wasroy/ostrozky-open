"""Batch evaluation of an agent over terrains, poses and runs (parallel).

    python -m rescuesim.evaluate --agent path/to/agent_dir [--terrains field forest ...]
        [--poses N] [--runs R] [--seed S] [--workers W] [--timeout 120] [--out DIR]
        [--layout-seed L] [--log]

Writes DIR/flights.csv (one row per flight) and DIR/metrics.json (per terrain
type and overall) and prints a summary table.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import sys
import time
import traceback

import numpy as np

from .terrain import TERRAIN_TYPES, get_layout

LATENCY_WARN_MS = 33.0
CSV_FIELDS = [
    "terrain", "pose", "run", "seed", "layout_seed",
    "start_x", "start_y", "start_z", "start_yaw_deg", "start_speed", "start_distance_m",
    "outcome", "time_s", "steps", "path_length_m", "first_detection_s", "final_distance_m",
    "mean_latency_ms",
]
METRIC_KEYS = [
    "success_rate", "collision_rate", "timeout_rate", "out_of_arena_rate", "error_rate",
    "reach_time_s", "reach_mean_speed_m_s", "path_efficiency", "detection_rate",
    "first_detection_s", "failed_final_distance_m", "latency_mean_ms", "agent_fps", "flights",
]


def _fly_job(job: dict) -> list[dict]:
    """Worker: one env and one Agent instance, all runs of one (terrain, pose)."""
    from .agent_loader import load_agent_class
    from .env import RescueEnv

    kind, pose, runs = job["terrain"], job["pose"], job["runs"]
    rows = []
    try:
        Agent = load_agent_class(job["agent"])
        agent = Agent()
        agent_err = None
    except Exception:
        agent, agent_err = None, traceback.format_exc()
    env = RescueEnv(f"{kind}@v{pose}", layout_seed=job["layout_seed"], max_time_s=job["timeout"])
    try:
        for run in runs:
            obs, info = env.reset(seed=job["seed"], options={"run": run})
            start = env.start
            start_dist = info["distance_m"]
            path, first_det, lat = 0.0, None, []
            prev = info["pos"].copy()
            log = {k: [] for k in ("pos", "vel", "rpy", "action", "detected", "box", "vio", "latency")} if job["log_dir"] else None
            depth_log = []
            if obs["detected"]:
                first_det = 0.0
            if agent is None:
                obs, _r, term, trunc, info = env.fail("error")
                err = agent_err
            else:
                err = None
                try:
                    agent.reset()
                except Exception:
                    err = traceback.format_exc()
                    obs, _r, term, trunc, info = env.fail("error")
            while err is None:
                t0 = time.perf_counter()
                try:
                    action = agent.act(obs)
                    dt = time.perf_counter() - t0
                except Exception:
                    dt = time.perf_counter() - t0
                    lat.append(dt)
                    err = traceback.format_exc()
                    obs, _r, term, trunc, info = env.fail("error")
                    break
                lat.append(dt)
                step_idx = env.steps
                if log is not None and step_idx % 30 == 0:
                    depth_log.append(obs["depth_m"].astype(np.float16))
                obs, _r, term, trunc, info = env.step(action)
                path += float(np.linalg.norm(info["pos"] - prev))
                prev = info["pos"].copy()
                if first_det is None and obs["detected"]:
                    first_det = info["t"]
                if log is not None:
                    try:
                        act_arr = np.asarray(action, float).reshape(4)
                    except Exception:
                        act_arr = np.full(4, np.nan)
                    for k, v in (("pos", info["pos"]), ("vel", info["vel"]), ("rpy", info["rpy"]), ("action", act_arr),
                                 ("detected", obs["detected"]), ("box", obs["box"]), ("vio", obs["vio_xyz_m"]),
                                 ("latency", dt)):
                        log[k].append(v)
                if term or trunc:
                    break
            row = {
                "terrain": kind, "pose": pose, "run": run, "seed": job["seed"],
                "layout_seed": env.terrain.layout_seed,
                "start_x": round(start["x"], 3), "start_y": round(start["y"], 3), "start_z": round(start["z"], 3),
                "start_yaw_deg": round(math.degrees(start["yaw"]), 2), "start_speed": round(start["speed"], 3),
                "start_distance_m": round(start_dist, 3),
                "outcome": info["outcome"], "time_s": round(info["t"], 4), "steps": env.steps,
                "path_length_m": round(path, 3),
                "first_detection_s": "" if first_det is None else round(first_det, 4),
                "final_distance_m": round(info["distance_m"], 3),
                "mean_latency_ms": round(1000 * float(np.mean(lat)), 4) if lat else "",
                "_latency_sum": float(np.sum(lat)), "_latency_n": len(lat), "_error": err,
            }
            rows.append(row)
            if log is not None:
                name = os.path.join(job["log_dir"], f"{kind}_v{pose}_r{run}.npz")
                arrays = {k: np.asarray(v) for k, v in log.items()}
                arrays["depth_every_30"] = np.asarray(depth_log) if depth_log else np.zeros((0, 126, 224), np.float16)
                np.savez_compressed(name, **arrays)
    finally:
        env.close()
    return rows


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    out = {k: None for k in METRIC_KEYS}
    out["flights"] = n
    if n == 0:
        return out

    def rate(o):
        return sum(r["outcome"] == o for r in rows) / n

    out["success_rate"] = rate("success")
    out["collision_rate"] = rate("collision")
    out["timeout_rate"] = rate("timeout")
    out["out_of_arena_rate"] = rate("out_of_arena")
    out["error_rate"] = rate("error")
    succ = [r for r in rows if r["outcome"] == "success"]
    if succ:
        out["reach_time_s"] = float(np.mean([r["time_s"] for r in succ]))
        out["reach_mean_speed_m_s"] = float(np.mean([r["path_length_m"] / max(r["time_s"], 1e-9) for r in succ]))
        out["path_efficiency"] = float(np.mean([r["start_distance_m"] / max(r["path_length_m"], 1e-9) for r in succ]))
    det = [r for r in rows if r["first_detection_s"] != ""]
    out["detection_rate"] = len(det) / n
    if det:
        out["first_detection_s"] = float(np.mean([r["first_detection_s"] for r in det]))
    failed = [r for r in rows if r["outcome"] != "success"]
    if failed:
        out["failed_final_distance_m"] = float(np.mean([r["final_distance_m"] for r in failed]))
    lat_n = sum(r["_latency_n"] for r in rows)
    if lat_n:
        out["latency_mean_ms"] = 1000.0 * sum(r["_latency_sum"] for r in rows) / lat_n
        out["agent_fps"] = 1000.0 / out["latency_mean_ms"] if out["latency_mean_ms"] > 0 else None
    return out


def _fmt(v, pct=False, nd=1):
    if v is None:
        return "-"
    return f"{100 * v:.0f}%" if pct else f"{v:.{nd}f}"


def print_table(per_terrain: dict, overall: dict, stream=sys.stdout):
    head = f"{'terrain':10s} {'flights':>7s} {'success':>8s} {'crash':>6s} {'timeout':>8s} {'out':>5s} {'error':>6s} {'reach_s':>8s} {'detect':>7s} {'lat_ms':>7s}"
    print(head, file=stream)
    print("-" * len(head), file=stream)
    for name, m in list(per_terrain.items()) + [("overall", overall)]:
        print(f"{name:10s} {m['flights']:7d} {_fmt(m['success_rate'], True):>8s} {_fmt(m['collision_rate'], True):>6s} "
              f"{_fmt(m['timeout_rate'], True):>8s} {_fmt(m['out_of_arena_rate'], True):>5s} {_fmt(m['error_rate'], True):>6s} "
              f"{_fmt(m['reach_time_s']):>8s} {_fmt(m['detection_rate'], True):>7s} {_fmt(m['latency_mean_ms'], nd=2):>7s}",
              file=stream)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent", required=True, help="folder with agent.py (importable as a package)")
    ap.add_argument("--terrains", nargs="+", default=list(TERRAIN_TYPES))
    ap.add_argument("--poses", type=int, default=40, help="fly poses v1..vN of each terrain")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--timeout", type=float, default=120.0, help="flight time limit (s)")
    ap.add_argument("--out", default="eval_out")
    ap.add_argument("--layout-seed", type=int, default=None, help="non-default layout (generalisation tests)")
    ap.add_argument("--log", action="store_true", help="write one .npz per flight to DIR/logs")
    args = ap.parse_args(argv)

    agent_dir = os.path.abspath(args.agent)
    os.makedirs(args.out, exist_ok=True)
    log_dir = os.path.join(os.path.abspath(args.out), "logs") if args.log else None
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    jobs = []
    for kind in args.terrains:
        n_avail = len(get_layout(kind, args.layout_seed).poses)
        if args.poses > n_avail:
            print(f"warning: {kind} has only {n_avail} poses; flying v1..v{n_avail}")
        for pose in range(1, min(args.poses, n_avail) + 1):
            jobs.append({"agent": agent_dir, "terrain": kind, "pose": pose, "runs": list(range(args.runs)),
                         "seed": args.seed, "timeout": args.timeout, "layout_seed": args.layout_seed,
                         "log_dir": log_dir})
    n_flights = len(jobs) * args.runs
    workers = max(1, min(args.workers, len(jobs)))
    print(f"evaluating {agent_dir}: {len(args.terrains)} terrain(s), {n_flights} flights, {workers} worker(s)")
    t0 = time.perf_counter()
    rows = []
    if workers == 1:
        results = map(_fly_job, jobs)
    else:
        pool = mp.get_context("spawn").Pool(workers)
        results = pool.imap_unordered(_fly_job, jobs)
    try:
        for res in results:
            rows.extend(res)
            done = len(rows)
            last = res[-1] if res else None
            if last:
                print(f"  [{done}/{n_flights}] {last['terrain']}@v{last['pose']} run {last['run']}: "
                      f"{last['outcome']} after {last['time_s']:.1f} s", flush=True)
            for r in res:
                if r["_error"]:
                    print(f"    agent error in {r['terrain']}@v{r['pose']} run {r['run']}:\n{r['_error']}", flush=True)
    finally:
        if workers > 1:
            pool.close()
            pool.join()
    rows.sort(key=lambda r: (args.terrains.index(r["terrain"]), r["pose"], r["run"]))
    with open(os.path.join(args.out, "flights.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    per = {k: summarize([r for r in rows if r["terrain"] == k]) for k in args.terrains}
    overall = summarize(rows)
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump({"overall": overall, "terrains": per,
                   "config": {"agent": agent_dir, "terrains": args.terrains, "poses": args.poses, "runs": args.runs,
                              "seed": args.seed, "timeout_s": args.timeout, "layout_seed": args.layout_seed}},
                  f, indent=2)
    print()
    print_table(per, overall)
    print(f"\nwall time {time.perf_counter() - t0:.1f} s; wrote {os.path.join(args.out, 'flights.csv')} and metrics.json")
    if overall["latency_mean_ms"] is not None and overall["latency_mean_ms"] > LATENCY_WARN_MS:
        print(f"WARNING: mean act() latency {overall['latency_mean_ms']:.1f} ms exceeds {LATENCY_WARN_MS:.0f} ms "
              f"(the 30 Hz control budget)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
