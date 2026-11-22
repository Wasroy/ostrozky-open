"""gpusim: rescuesim batched on the GPU (NVIDIA Warp), for training.

GPU engine originally written by @sckoott (OSTROZKY team) for the EDTH Paris 2026 hackathon; ported to rescuesim.

rescuesim is the reference: gpusim reproduces its dynamics, cameras, sensor models, starts and
outcome rules for thousands of flights at once (``python -m gpusim.validate`` measures how closely).

    from gpusim import FastBatchEnv
    env = FastBatchEnv(n=1024, terrains=("field", "forest", "village", "city", "rubble"), obs="pooled")
    obs = env.reset()
    obs, done, outcome, info = env.step(actions)          # actions (N, 4) in [-1, 1]

Needs ``warp-lang`` (``pip install -e .[gpu]``) and, for speed, a CUDA GPU. ``gpusim.dynamics``
(the batched NumPy dynamics) and ``gpusim.scene`` (terrain export) work without Warp.
"""
__all__ = ["FastBatchEnv", "OUTCOMES"]


def __getattr__(name):
    if name in __all__:
        from . import fast
        return getattr(fast, name)
    raise AttributeError(name)
