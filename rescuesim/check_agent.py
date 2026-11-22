"""Quick API check for an agent folder.

    python -m rescuesim.check_agent --agent path/to/agent_dir

Runs 10 steps on village@v1 and checks that ``act`` returns 4 finite floats in
[-1, 1]. Prints "Agent API OK" and exits 0 on success.
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback

import numpy as np

from .agent_loader import load_agent_class
from .env import RescueEnv


def check_agent(agent_dir: str, terrain: str = "village@v1", steps: int = 10, seed: int = 0, verbose: bool = True) -> bool:
    def say(msg):
        if verbose:
            print(msg)

    try:
        Agent = load_agent_class(agent_dir)
        agent = Agent()
    except Exception:
        say("could not import/instantiate Agent:\n" + traceback.format_exc())
        return False
    env = RescueEnv(terrain)
    try:
        obs, _info = env.reset(seed=seed)
        agent.reset()
        lat = []
        for k in range(steps):
            t0 = time.perf_counter()
            try:
                action = agent.act(obs)
            except Exception:
                say(f"step {k}: act() raised:\n" + traceback.format_exc())
                return False
            lat.append(time.perf_counter() - t0)
            try:
                a = np.asarray(action, dtype=float)
            except (TypeError, ValueError):
                say(f"step {k}: action {action!r} is not numeric")
                return False
            if a.shape != (4,):
                say(f"step {k}: action shape {a.shape}, expected (4,)")
                return False
            if not np.all(np.isfinite(a)):
                say(f"step {k}: action {a} is not finite")
                return False
            if np.any(np.abs(a) > 1.0):
                say(f"step {k}: action {a} outside [-1, 1]")
                return False
            obs, _r, term, trunc, info = env.step(a)
            if term or trunc:
                say(f"(flight ended early at step {k}: {info['outcome']})")
                break
        say(f"mean act latency {1000 * float(np.mean(lat)):.2f} ms over {len(lat)} steps")
        say("Agent API OK")
        return True
    finally:
        env.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent", required=True, help="folder with agent.py (importable as a package)")
    ap.add_argument("--terrain", default="village@v1")
    ap.add_argument("--steps", type=int, default=10)
    args = ap.parse_args(argv)
    return 0 if check_agent(args.agent, args.terrain, args.steps) else 1


if __name__ == "__main__":
    sys.exit(main())
