"""Import ``<dir>/agent.py`` as ``<dirname>.agent`` with ``<dir>``'s parent on sys.path."""
from __future__ import annotations

import importlib
import os
import sys


def load_agent_class(agent_dir: str):
    path = os.path.abspath(agent_dir).rstrip("\\/")
    if not os.path.isfile(os.path.join(path, "agent.py")):
        raise FileNotFoundError(f"{path} has no agent.py")
    parent, name = os.path.split(path)
    if not name.isidentifier():
        raise ValueError(f"agent folder name {name!r} is not a valid Python package name")
    if parent not in sys.path:
        sys.path.insert(0, parent)
    module = importlib.import_module(f"{name}.agent")
    if not hasattr(module, "Agent"):
        raise AttributeError(f"{name}.agent has no class Agent")
    return module.Agent
