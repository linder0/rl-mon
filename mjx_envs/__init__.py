"""MJX ports of the gymnasium -v5 locomotion envs (see locomotion.py) and the
parametric monster generator (see monsters.py).

locomotion.py is imported lazily (PEP 562): it needs jax/brax, but
mjx_envs.monsters must stay importable from the torch venv (export_onnx.py),
which has neither.
"""

__all__ = ["ENVS", "make_env"]


def __getattr__(name):
    if name in __all__:
        from mjx_envs import locomotion
        return getattr(locomotion, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
