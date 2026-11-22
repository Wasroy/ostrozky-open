"""Check gpusim against rescuesim (the reference) and print a table.

    python -m gpusim.validate [--quick] [--only dynamics contacts render e2e speed] [--workers W]

* dynamics: the same stick sequences (8 manoeuvres, 4 s) flown open loop in rescuesim and in gpusim
  (Warp kernel, float32) and in gpusim.dynamics (NumPy, float64) from the same start;
* contacts: straight drone paths into props and boxes, first touching sub-step in rescuesim's
  PyBullet world vs gpusim's contact kernel;
* render: random drone poses near the targets of all 5 terrains, noise off: depth error, sky/hit
  mismatch, detector box agreement (rescuesim's segmentation box vs gpusim's ray-cast box);
* e2e: the team agent (``handin``, loaded like rescuesim.evaluate) flown from the same starts in
  both simulators (seed 0, run 0); plus rescuesim against itself with other noise samples (same
  start), which is the agreement to expect from noise alone, and gpusim fed rescuesim's own noise
  samples (noise="rescuesim"), which leaves only the simulator differences;
* speed: drone-steps per second of FastBatchEnv.
"""
from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import sys
import time
import traceback

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

KINDS = ("field", "forest", "village", "city", "rubble")
LINE = "-" * 96


def _hdr(title):
    print(f"\n{title}\n{LINE}", flush=True)


#### Dynamics ######################################################################################
def manoeuvres(n_steps=120):
    t = np.arange(n_steps) / 30.0
    sq = np.where((t // 0.5) % 2 == 0, 1.0, -1.0)
    rng = np.random.default_rng(7)
    smooth = np.cumsum(rng.normal(0, 0.15, (n_steps, 4)), 0)
    smooth = np.clip(smooth - smooth.mean(0), -0.9, 0.9)
    const = lambda *a: np.tile(a, (n_steps, 1)).astype(float)
    return {
        "hover": const(0, 0, 0, 0),
        "forward (pitch .5)": const(0, 0.5, 0, 0.3),
        "right (roll .6)": const(0.6, 0, 0, 0.2),
        "yaw spin + climb": const(0, 0, 1, 0.4),
        "full pitch + thrust": const(0, 1, 0, 1),
        "bang-bang roll/pitch": np.c_[sq, -sq, 0.5 * sq, np.full(n_steps, 0.3)],
        "descent (thr -.6)": const(0.2, -0.2, -0.3, -0.6),
        "random smooth sticks": smooth,
    }


def check_dynamics(device):
    from rescuesim.env import RescueEnv
    from gpusim.dynamics import BatchQuad, quat_to_matrix
    from gpusim.fast import FastBatchEnv
    man = manoeuvres()
    names = list(man)
    start = {"pos": (180.0, 190.0, 100.0), "yaw": 0.4, "speed": 5.0}
    env = RescueEnv("empty@v1", render_depth=False, render_detector=False)
    ref_p, ref_R = [], []
    for name in names:
        env.reset(seed=0, options={"start_pos": start["pos"], "start_yaw": start["yaw"], "start_speed": start["speed"]})
        P, Rs = [], []
        for a in man[name]:
            env.step(a)
            P.append(env.quad.state.pos.copy())
            Rs.append(quat_to_matrix(np.asarray(env.quad.state.quat)[None])[0])
        ref_p.append(P)
        ref_R.append(Rs)
    env.close()
    ref_p, ref_R = np.array(ref_p), np.array(ref_R)
    m = len(names)
    gpu = FastBatchEnv(m, terrains=("empty",), device=device, noise=False)
    gpu.reset(terrains=["empty@v1"] * m, runs=[0] * m, starts=[start] * m)
    cpu = BatchQuad(m)
    cpu.reset(np.arange(m), np.tile(start["pos"], (m, 1)), start["yaw"], start["speed"])
    gp, gR, cp, cR = [], [], [], []
    for k in range(man[names[0]].shape[0]):
        acts = np.stack([man[nm][k] for nm in names])
        gpu.step(acts)
        tr = gpu.truth()
        gp.append(tr["pos"])
        gR.append(quat_to_matrix(tr["quat"]))
        cpu.step(acts)
        cp.append(cpu.pos.copy())
        cR.append(quat_to_matrix(cpu.quat))
    gp, gR, cp, cR = (np.transpose(np.array(x), (1, 0) + tuple(range(2, np.array(x).ndim))) for x in (gp, gR, cp, cR))

    def ang(Ra, Rb):
        c = (np.einsum("...ij,...ij->...", Ra, Rb) - 1.0) / 2.0
        return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))

    rows = []
    for j, nm in enumerate(names):
        dist = np.linalg.norm(ref_p[j, -1] - ref_p[j, 0])
        rows.append((nm, dist, np.linalg.norm(gp[j, -1] - ref_p[j, -1]), np.linalg.norm(gp[j] - ref_p[j], axis=1).max(),
                     ang(gR[j, -1], ref_R[j, -1]), np.linalg.norm(cp[j, -1] - ref_p[j, -1]), ang(cR[j, -1], ref_R[j, -1])))
    _hdr("Open-loop dynamics, 4 s (120 steps) from the same start: error vs rescuesim")
    print(f"{'manoeuvre':24s} {'flown m':>8s} {'warp pos@4s':>12s} {'warp max':>9s} {'warp att deg':>13s} {'numpy pos':>10s} {'numpy att':>10s}")
    for r in rows:
        print(f"{r[0]:24s} {r[1]:8.1f} {100 * r[2]:10.2f}cm {100 * r[3]:7.2f}cm {r[4]:13.4f} {r[5]:10.1e} {r[6]:10.1e}")
    return {"dynamics_max_pos_err_cm": float(max(100 * r[3] for r in rows)),
            "dynamics_max_att_err_deg": float(max(r[4] for r in rows)),
            "dynamics_numpy_max_pos_err_m": float(max(r[5] for r in rows))}


#### Contacts ######################################################################################
def check_contacts(device, n_paths=200):
    from rescuesim.dynamics import QuadParams
    from rescuesim.geometry import quat_to_matrix
    from rescuesim.world import CollisionWorld
    from gpusim.fast import FastBatchEnv
    env = FastBatchEnv(1, terrains=KINDS, device=device, noise=False)
    _hdr(f"Contacts: {n_paths} straight paths (3-25 m/s) into props / the box per terrain, first touching sub-step")
    print(f"{'terrain':10s} {'touching':>9s} {'same sub-step+body':>19s} {'1 sub-step early':>17s} {'differ':>7s}")
    tot = np.zeros(4)
    for ti, kind in enumerate(KINDS):
        sc = env.scenes[ti]
        lay = sc.layout
        cw = CollisionWorld(list(lay.prims), QuadParams().collision_half_extents)
        rng = np.random.default_rng(100 + ti)
        coll = [p for p in lay.prims if p.collides]
        res = np.zeros(4)
        S_ = 240
        for k in range(n_paths):
            v_i = int(rng.integers(sc.n_poses))
            pose = lay.poses[v_i]
            cw.set_target(pose)
            if k % 4 == 0:
                c = np.array(pose.center)
                lo, hi = c - [2.0, 0.8, 0.77], c + [2.0, 0.8, 0.77]
            else:
                lo, hi = map(np.asarray, coll[rng.integers(len(coll))].aabb())
            tgt = rng.uniform(lo, hi)
            d = rng.normal(size=3)
            d[2] = abs(d[2]) * 0.5
            d /= np.linalg.norm(d)
            start = tgt + d * (np.linalg.norm(hi - lo) / 2 + 2.0)
            vel = (tgt - start) / np.linalg.norm(tgt - start) * rng.uniform(3, 25)
            q = rng.normal(size=4)
            q /= np.linalg.norm(q)
            P = start + vel[None] * (np.arange(S_)[:, None] / 240.0)
            if cw.drone_contacts(P[0], tuple(q)):
                continue
            ri, rk = -1, 0
            for i in range(S_):
                h = cw.drone_contacts(P[i], tuple(q))
                if h:
                    ri, rk = i, (1 if cw.target in h else 0) | (2 if h - {cw.target} else 0)
                    break
            g = env._contacts(np.full(S_, ti), P, np.tile(quat_to_matrix(q), (S_, 1, 1)), np.tile(sc.boxes[v_i], (S_, 1)), 1e-3)
            gi = int(np.flatnonzero(g)[0]) if g.any() else -1
            gk = int(g[gi]) if gi >= 0 else 0
            res[0] += ri >= 0 or gi >= 0
            if ri == gi and rk == gk:
                res[1] += 1
            elif ri >= 0 and gi == ri - 1 and rk == gk:
                res[2] += 1
            else:
                res[3] += 1
        cw.close()
        n = res[1] + res[2] + res[3]
        tot += res
        print(f"{kind:10s} {int(res[0]):9d} {100 * res[1] / n:18.1f}% {100 * res[2] / n:16.1f}% {100 * res[3] / n:6.1f}%")
    n = tot[1:].sum()
    print(f"{'all':10s} {int(tot[0]):9d} {100 * tot[1] / n:18.1f}% {100 * tot[2] / n:16.1f}% {100 * tot[3] / n:6.1f}%")
    return {"contact_same": float(tot[1] / n), "contact_one_early": float(tot[2] / n), "contact_differ": float(tot[3] / n)}


#### Rendering #####################################################################################
def check_render(device, per_terrain=30):
    from rescuesim import sensors as S
    from rescuesim.dynamics import QuadParams
    from rescuesim.geometry import quat_from_euler, quat_to_matrix
    from rescuesim.world import CollisionWorld, RenderWorld
    from gpusim.render import Renderer
    from gpusim.scene import GpuScenes, load_terrain_scene
    scenes = [load_terrain_scene(k) for k in KINDS]
    ren = Renderer(GpuScenes(scenes, device), device)
    rng = np.random.default_rng(11)
    _hdr(f"Rendering, noise off: {per_terrain} random poses per terrain (drone 5-80 m from a target, facing it +-0.4 rad)")
    print(f"{'terrain':10s} {'frames':>6s} {'depth med':>10s} {'depth p99':>10s} {'sky/hit mism.':>14s} {'box seen':>9s} "
          f"{'seen agree':>11s} {'box identical':>14s} {'edge |d| px':>12s}")
    allerr, allmis, totals, both_total = [], [], np.zeros(5), 0
    for ti, kind in enumerate(KINDS):
        lay = scenes[ti].layout
        rw = RenderWorld(list(lay.prims), lay.ground_rgba)
        cw = CollisionWorld(list(lay.prims), QuadParams().collision_half_extents)
        P, Q, O, ref_d, ref_b = [], [], [], [], []
        while len(P) < per_terrain:
            v = int(rng.integers(len(lay.poses)))
            pose = lay.poses[v]
            a = rng.uniform(-math.pi, math.pi)
            dist = rng.uniform(5, 80)
            pos = np.array([pose.x + dist * math.cos(a), pose.y + dist * math.sin(a), rng.uniform(1.0, 30.0)])
            q = quat_from_euler(rng.normal(0, 0.15), rng.normal(0, 0.25), a + math.pi + rng.normal(0, 0.4))
            cw.set_target(pose)
            if cw.drone_contacts(pos, q):
                continue
            rw.set_target(pose)
            cam = S.camera_frame(pos, quat_to_matrix(q))
            mask, off, _dist = rw.render_target_mask(cam)
            if mask is not None and mask.any():
                cols, rows = np.flatnonzero(mask.any(0)), np.flatnonzero(mask.any(1))
                ref_b.append([off[0] + cols[0], off[1] + rows[0], off[0] + cols[-1] + 1, off[1] + rows[-1] + 1])
            elif sum(b[0] < 0 for b in ref_b) >= per_terrain // 2:
                continue                       # keep about half of the frames with the box in view
            else:
                ref_b.append([-1] * 4)
            ref_d.append(np.clip(rw.render_depth(cam), 0.0, S.DEPTH_MAX_M))
            P.append(pos)
            Q.append(q)
            O.append(scenes[ti].boxes[v])
        rw.close()
        cw.close()
        gd, gb = ren.render(np.full(len(P), ti), np.array(P), np.array(Q), np.array(O))
        ref_d, ref_b = np.array(ref_d), np.array(ref_b)
        err = np.abs(gd - ref_d)
        mis = (gd >= S.DEPTH_MAX_M - 1e-3) != (ref_d >= S.DEPTH_MAX_M - 1e-3)
        seen_r, seen_g = ref_b[:, 0] >= 0, gb[:, 0] >= 0
        either = seen_r | seen_g
        same = np.all(gb == ref_b, 1)
        both = seen_r & seen_g
        edge = np.abs(gb[both] - ref_b[both]).mean() if both.any() else 0.0
        pct = lambda x: 100 * x.mean() if x.size else float("nan")
        allerr.append(err.ravel())
        allmis.append(mis.ravel())
        totals += [either.sum(), (seen_r == seen_g)[either].sum(), same[either].sum(), len(P), np.abs(gb[both] - ref_b[both]).sum()]
        both_total += both.sum()
        print(f"{kind:10s} {len(P):6d} {100 * np.median(err):8.3f}cm {100 * np.percentile(err, 99):8.2f}cm "
              f"{100 * mis.mean():13.3f}% {int(either.sum()):9d} {pct((seen_r == seen_g)[either]):10.1f}% "
              f"{pct(same[either]):13.1f}% {edge:12.3f}")
    err, mis = np.concatenate(allerr), np.concatenate(allmis)
    print(f"{'all':10s} {int(totals[3]):6d} {100 * np.median(err):8.3f}cm {100 * np.percentile(err, 99):8.2f}cm "
          f"{100 * mis.mean():13.3f}% {int(totals[0]):9d} {100 * totals[1] / totals[0]:10.1f}% "
          f"{100 * totals[2] / totals[0]:13.1f}% {totals[4] / max(1.0, 4 * both_total):12.3f}")
    return {"depth_median_cm": float(100 * np.median(err)), "depth_p99_cm": float(100 * np.percentile(err, 99)),
            "depth_mean_abs_cm": float(100 * err.mean()), "sky_hit_mismatch": float(mis.mean()),
            "box_frames": int(totals[0]), "box_identical": float(totals[2] / totals[0]),
            "box_seen_agree": float(totals[1] / totals[0]), "box_edge_err_px": float(totals[4] / max(1.0, 4 * both_total))}


#### End to end ####################################################################################
def _rescuesim_flight(job):
    """One flight in rescuesim, flown like rescuesim.evaluate does (one env, one Agent)."""
    from rescuesim.agent_loader import load_agent_class
    from rescuesim.env import RescueEnv
    env = RescueEnv(job["terrain"], max_time_s=job["timeout"])
    try:
        agent = load_agent_class(job["agent"])()
        opts = {"run": job["run"]}
        if job.get("start"):
            s = job["start"]
            opts.update(start_pos=(s["x"], s["y"], s["z"]), start_yaw=s["yaw"], start_speed=s["speed"])
        obs, info = env.reset(seed=job["seed"], options=opts)
        start = dict(env.start)
        agent.reset()
        while True:
            try:
                action = agent.act(obs)
            except Exception:
                obs, _r, term, trunc, info = env.fail("error")
                break
            obs, _r, term, trunc, info = env.step(action)
            if term or trunc:
                break
        return {"terrain": job["terrain"], "outcome": info["outcome"], "t": info["t"], "start": start,
                "final_distance_m": info["distance_m"]}
    finally:
        env.close()


def _agent_worker(conn, agent_dir):
    from rescuesim.agent_loader import load_agent_class
    Agent = load_agent_class(agent_dir)
    agents = {}
    while True:
        msg = conn.recv()
        if msg[0] == "stop":
            break
        if msg[0] == "reset":
            for i in msg[1]:
                agents[i] = Agent()
                agents[i].reset()
            conn.send(True)
        else:
            out = {}
            for i, ob in msg[1].items():
                try:
                    a = np.asarray(agents[i].act(ob), float)
                except Exception:
                    a = None
                out[i] = a
            conn.send(out)


def fly_gpusim(agent_dir, terrains, runs, seed, timeout, device, workers, noise=True):
    from gpusim.fast import FLYING, FastBatchEnv
    n = len(terrains)
    env = FastBatchEnv(n, terrains=KINDS, seed=seed, device=device, obs="full", noise=noise, timeout_s=timeout)
    ctx = mp.get_context("spawn")
    W = max(1, min(workers, n))
    parts = [list(range(w, n, W)) for w in range(W)]
    pipes, procs = [], []
    for w in range(W):
        a, b = ctx.Pipe()
        pr = ctx.Process(target=_agent_worker, args=(b, agent_dir), daemon=True)
        pr.start()
        pipes.append(a)
        procs.append(pr)
    try:
        for w in range(W):
            pipes[w].send(("reset", parts[w]))
        for w in range(W):
            pipes[w].recv()
        obs = env.reset(terrains=terrains, runs=runs)
        outcome = np.zeros(n, np.int32)
        t_steps = 0
        while (outcome == FLYING).any():
            act = np.zeros((n, 4))
            for w in range(W):
                pipes[w].send(("act", {i: env.single(obs, i) for i in parts[w] if outcome[i] == FLYING}))
            for w in range(W):
                for i, a in pipes[w].recv().items():
                    act[i] = a if a is not None and a.shape == (4,) else np.nan
            obs, done, outcome, info = env.step(act)
            t_steps += 1
        tr = env.truth()
    finally:
        for w in range(W):
            try:
                pipes[w].send(("stop",))
            except Exception:
                pass
        for pr in procs:
            pr.join(timeout=10)
    return [{"terrain": terrains[i], "outcome": env.outcome_name(outcome[i]), "t": float(tr["t"][i]),
             "start": tr["start"][i], "final_distance_m": float(tr["distance_m"][i])} for i in range(n)], t_steps


def check_e2e(device, agent_dir, poses=10, seed=0, timeout=120.0, workers=8, kinds=KINDS, runs=1, baseline=True):
    """Columns: A = rescuesim (the reference); B = rescuesim again from the same start with other noise
    samples (seed + 1000): the agreement noise alone allows; C = gpusim with its own noise (the
    training configuration); D = gpusim with rescuesim's noise samples (noise="rescuesim"): only the
    simulators differ."""
    kinds = tuple(kinds)
    jobs = [{"agent": agent_dir, "terrain": f"{k}@v{v}", "run": r, "seed": seed, "timeout": timeout}
            for k in kinds for v in range(1, poses + 1) for r in range(runs)]
    n = len(jobs)
    _hdr(f"End to end: agent {os.path.basename(agent_dir)}, {n} flights ({len(kinds)} terrains x v1..v{poses} x "
         f"{runs} run(s), seed {seed}), timeout {timeout:.0f} s")
    t0 = time.perf_counter()
    pool = mp.get_context("spawn").Pool(workers)
    try:
        ref = pool.map(_rescuesim_flight, jobs, chunksize=1)
        t_ref = time.perf_counter() - t0
        base = None
        if baseline:
            base_jobs = [dict(j, seed=seed + 1000, start=r["start"]) for j, r in zip(jobs, ref)]
            base = pool.map(_rescuesim_flight, base_jobs, chunksize=1)
    finally:
        pool.close()
        pool.join()
    terr, rns = [j["terrain"] for j in jobs], [j["run"] for j in jobs]
    t0 = time.perf_counter()
    gpu, n_steps = fly_gpusim(agent_dir, terr, rns, seed, timeout, device, workers, noise=True)
    t_gpu = time.perf_counter() - t0
    same, _ = fly_gpusim(agent_dir, terr, rns, seed, timeout, device, workers, noise="rescuesim")
    starts_equal = all(np.allclose([r["start"][k] for k in ("x", "y", "z", "yaw", "speed")], g["start"], atol=1e-6)
                       for r, g in zip(ref, gpu))
    print(f"starts identical in both simulators: {starts_equal}   (wall time: rescuesim {t_ref:.0f} s on {workers} "
          f"processes; gpusim {t_gpu:.0f} s for {n_steps} batch steps, agents on {workers} processes)")
    cols = [("A rescuesim", ref)] + ([("B resc. other noise", base)] if base else []) +         [("C gpusim", gpu), ("D gpusim resc. noise", same)]
    print(f"{'':10s} {'':7s} " + " ".join(f"{c[0]:>21s}" for c in cols))
    print(f"{'terrain':10s} {'flights':>7s} " + " ".join(f"{'success  agree w/ A':>21s}" for _ in cols))
    out = {}
    for k in kinds + ("all",):
        sel = [i for i, j in enumerate(jobs) if k == "all" or j["terrain"].startswith(k + "@")]
        cells = []
        for name, res in cols:
            sr = np.mean([res[i]["outcome"] == "success" for i in sel])
            ag = np.mean([res[i]["outcome"] == ref[i]["outcome"] for i in sel])
            cells.append(f"{100 * sr:8.0f}% {100 * ag:10.0f}%")
            if k == "all":
                out[name[0]] = (float(sr), float(ag))
        print(f"{k:10s} {len(sel):7d} " + " ".join(f"{c:>21s}" for c in cells))
    print("disagreements with rescuesim (A):  terrain run: A | B | C | D  (outcome, time)")
    for i, j in enumerate(jobs):
        if any(res[i]["outcome"] != ref[i]["outcome"] for _, res in cols):
            print(f"   {j['terrain']:9s} r{j['run']}: " + " | ".join(f"{res[i]['outcome']:>9s} {res[i]['t']:5.1f}s" for _, res in cols))
    res = {"e2e_flights": n, "e2e_starts_identical": bool(starts_equal),
           "e2e_rescuesim_success": out["A"][0], "e2e_gpusim_success": out["C"][0], "e2e_gpusim_agreement": out["C"][1],
           "e2e_gpusim_same_noise_success": out["D"][0], "e2e_gpusim_same_noise_agreement": out["D"][1]}
    if base:
        res.update(e2e_rescuesim_other_noise_success=out["B"][0], e2e_noise_only_agreement=out["B"][1])
    return res


#### Throughput ####################################################################################
def check_speed(device, sizes=(256, 1024, 4096), full_sizes=(256, 1024), steps=30):
    from gpusim.fast import FastBatchEnv
    _hdr(f"Throughput ({device}), 5 terrains, noise on, hover-ish random sticks, {steps} timed steps")
    print(f"{'obs':8s} {'n':>6s} {'drone-steps/s':>14s} {'ms/step':>9s}")
    out = {}
    for mode, ns in (("pooled", sizes), ("full", full_sizes)):
        for n in ns:
            try:
                env = FastBatchEnv(n, terrains=KINDS, device=device, obs=mode)
                env.reset()
                rng = np.random.default_rng(0)
                acts = lambda: np.clip(rng.normal(0, 0.2, (n, 4)), -1, 1)
                for _ in range(3):
                    env.step(acts())
                t0 = time.perf_counter()
                for _ in range(steps):
                    env.step(acts())
                dt = time.perf_counter() - t0
                rate = n * steps / dt
                out[f"{mode}_{n}"] = rate
                print(f"{mode:8s} {n:6d} {rate:14.0f} {1000 * dt / steps:9.1f}")
                del env
            except Exception as e:  # out of memory on small GPUs
                print(f"{mode:8s} {n:6d}   failed: {type(e).__name__}: {str(e)[:60]}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", default=["dynamics", "contacts", "render", "e2e", "speed"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--agent", default=os.path.join(ROOT, "handin"))
    ap.add_argument("--poses", type=int, default=10, help="e2e: poses v1..vN of each terrain")
    ap.add_argument("--terrains", nargs="+", default=list(KINDS), help="e2e terrains")
    ap.add_argument("--runs", type=int, default=1, help="e2e: runs per pose")
    ap.add_argument("--no-baseline", action="store_true", help="e2e: skip the rescuesim noise-only rerun")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--quick", action="store_true", help="smaller samples (smoke test)")
    args = ap.parse_args(argv)
    import warp as wp
    wp.config.quiet = True
    wp.init()
    res = {}
    t0 = time.perf_counter()
    if "dynamics" in args.only:
        res.update(check_dynamics(args.device))
    if "contacts" in args.only:
        res.update(check_contacts(args.device, 40 if args.quick else 200))
    if "render" in args.only:
        res.update(check_render(args.device, 6 if args.quick else 30))
    if "e2e" in args.only:
        res.update(check_e2e(args.device, os.path.abspath(args.agent), 2 if args.quick else args.poses,
                             workers=args.workers, kinds=args.terrains, runs=args.runs,
                             baseline=not args.no_baseline))
    if "speed" in args.only:
        res.update(check_speed(args.device, (256,) if args.quick else (256, 1024, 4096), (256,) if args.quick else (256, 1024)))
    _hdr(f"Summary ({time.perf_counter() - t0:.0f} s)")
    for k, v in res.items():
        print(f"  {k:32s} {v:.4g}" if isinstance(v, float) else f"  {k:32s} {v}")
    return res


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
