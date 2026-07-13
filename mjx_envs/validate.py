"""Construction-parity gate: the MJX envs must build the SAME observation,
reward, and termination signal as gymnasium, given the same physics state.

For random rollouts in the real gymnasium env, every C-MuJoCo transition
(state_before, action, state_after) is replayed through the MJX env's pure
functions (obs_from_data / is_healthy / reward_terms) and compared against
what gymnasium actually returned.

This validates *construction*, not dynamics: MJX and C MuJoCo integrate
slightly differently, so trajectories diverge — but given the same state, the
observation/reward math must agree. For Ant, cfrc_ext is additionally
recomputed with mjx.rne_postconstraint from the C-MuJoCo state and compared
against C's values, since that is the code path training uses.

Usage:
    python -m mjx_envs.validate                    # all supported envs
    python -m mjx_envs.validate --env Walker2d-v5
"""

import argparse

import gymnasium as gym
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

from mjx_envs.locomotion import ENVS, AntMjx, make_env

# f32 vs f64 rounding on values up to ~10 (clipped qvel) is ~1e-6; leave
# headroom. cfrc_ext goes through a full f32 RNE recomputation, so it gets a
# looser tolerance.
OBS_ATOL = 1e-4
REWARD_ATOL = 1e-4
CFRC_ATOL = 5e-3


def put_data(env, mj_model, mj_data):
    """C-MuJoCo state -> mjx.Data on device (carries qpos/qvel/xpos/contact)."""
    return mjx.put_data(mj_model, mj_data)


def validate_env(env_id: str, steps: int = 200, seed: int = 0) -> dict:
    print(f"[{env_id}] rolling {steps} random steps in gymnasium ...")
    genv = gym.make(env_id)
    u = genv.unwrapped
    menv = make_env(env_id)
    is_ant = isinstance(menv, AntMjx)

    obs_fn = jax.jit(menv.obs_from_data)
    healthy_fn = jax.jit(menv.is_healthy)
    rew_fn = jax.jit(
        lambda xv, healthy, action, data: menv.reward_terms(
            xv, healthy, action, data)[0])
    cfrc_fn = jax.jit(lambda data: mjx.rne_postconstraint(menv.sys, data).cfrc_ext)

    rng = np.random.default_rng(seed)
    obs, _ = genv.reset(seed=seed)
    worst = {"obs": 0.0, "reward": 0.0, "healthy_mismatches": 0, "cfrc": 0.0}

    data_before = put_data(menv, u.model, u.data)
    x_before = float(menv.forward_pos(data_before))

    for t in range(steps):
        action = genv.action_space.sample().astype(np.float64)
        obs, reward, terminated, truncated, info = genv.step(action)
        data_after = put_data(menv, u.model, u.data)

        # Observation construction (uses C-computed cfrc_ext for Ant).
        rec = np.asarray(obs_fn(data_after))
        worst["obs"] = max(worst["obs"], float(np.max(np.abs(rec - obs))))

        # Ant: recompute cfrc_ext in JAX from the same state — the training
        # code path — and compare against C-MuJoCo's values.
        if is_ant:
            jax_cfrc = np.asarray(cfrc_fn(data_after))
            c_cfrc = np.asarray(u.data.cfrc_ext)
            lo, hi = menv.contact_force_range
            err = np.max(np.abs(np.clip(jax_cfrc, lo, hi) - np.clip(c_cfrc, lo, hi)))
            worst["cfrc"] = max(worst["cfrc"], float(err))

        # Termination flag.
        healthy_mjx = bool(healthy_fn(data_after))
        if healthy_mjx != bool(u.is_healthy):
            worst["healthy_mismatches"] += 1

        # Reward, recomputed from the same before/after positions + action.
        x_after = float(menv.forward_pos(data_after))
        xv = (x_after - x_before) / menv.dt
        rec_reward = float(rew_fn(jnp.float32(xv),
                                  jnp.float32(1.0 if u.is_healthy else 0.0),
                                  jnp.asarray(action, dtype=jnp.float32),
                                  data_after))
        worst["reward"] = max(worst["reward"], abs(rec_reward - float(reward)))

        x_before = x_after
        data_before = data_after
        if terminated or truncated:
            obs, _ = genv.reset()
            data_before = put_data(menv, u.model, u.data)
            x_before = float(menv.forward_pos(data_before))

    genv.close()
    return worst


def main():
    p = argparse.ArgumentParser(description="MJX env construction-parity gate")
    p.add_argument("--env", default=None, choices=sorted(ENVS),
                   help="validate one env (default: all)")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    env_ids = [args.env] if args.env else sorted(ENVS)
    failures = []
    for env_id in env_ids:
        w = validate_env(env_id, args.steps, args.seed)
        ok = (w["obs"] <= OBS_ATOL and w["reward"] <= REWARD_ATOL
              and w["healthy_mismatches"] == 0 and w["cfrc"] <= CFRC_ATOL)
        status = "PASS" if ok else "FAIL"
        print(f"[{env_id}] {status}  max|obs err|={w['obs']:.3g}  "
              f"max|reward err|={w['reward']:.3g}  "
              f"healthy mismatches={w['healthy_mismatches']}  "
              f"max|cfrc err|={w['cfrc']:.3g}")
        if not ok:
            failures.append(env_id)

    if failures:
        raise SystemExit(f"FAILED construction parity: {failures}")
    print("All envs match gymnasium construction.")


if __name__ == "__main__":
    main()
