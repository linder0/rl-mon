"""Small helpers shared across the scripts (train / enjoy / export / tune).

Kept deliberately tiny and dependency-light so every entry point can import it
without pulling in anything heavy beyond what it already uses.
"""

import json
import os

DEFAULT_ENV_ID = "Hopper-v5"

# Policy activation functions, keyed by the names train.py's --activation-fn
# and the RL-Zoo Optuna search use. Single source of truth for the mapping.
# torch is optional so the MJX/Brax venv (no torch) can import this module.
try:
    from torch import nn

    ACTIVATION_FNS = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "elu": nn.ELU,
        "leaky_relu": nn.LeakyReLU,
    }
except ImportError:  # torch-free environment (e.g. requirements-mjx.txt)
    ACTIVATION_FNS = {}


def env_label(env_id):
    """Short folder name for an env id, e.g. 'Hopper-v5' -> 'hopper'."""
    return env_id.split("-v")[0].lower()


def env_from_config(run_dir):
    """Read the env id recorded in a run's config.json, if present."""
    cfg = os.path.join(run_dir, "config.json")
    if os.path.exists(cfg):
        try:
            return json.load(open(cfg)).get("env_id")
        except Exception:
            return None
    return None


def resolve_latest(label):
    """Resolve runs/<label>/latest to a run dir, falling back to latest.txt.

    Returns the (possibly non-existent) symlink path if neither is found, so
    callers can produce a clear "train one first" error.
    """
    run = os.path.join("runs", label, "latest")
    if os.path.exists(run):
        return run
    txt = os.path.join("runs", label, "latest.txt")
    if os.path.exists(txt):
        return open(txt).read().strip()
    return run
