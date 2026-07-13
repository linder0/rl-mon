"""Watch a trained PPO agent play a MuJoCo task in a live viewer.

Runs are grouped by environment under runs/<env>/. By default this loads the
most recent run for the chosen env (runs/<env>/latest), auto-detecting the
exact env id from that run's config.json.

Usage:
    python enjoy.py                                  # latest Hopper run
    python enjoy.py --env Walker2d-v5                # latest Walker run
    python enjoy.py --run runs/hopper/ppo_20260708_132700
    python enjoy.py --best                            # use best_model.zip
    python enjoy.py --model runs/.../checkpoints/ppo_hopper_100000_steps.zip
    python enjoy.py --episodes 10
    python enjoy.py --stochastic                      # sample instead of argmax

The VecNormalize stats are loaded so the policy sees the same normalized
observations it trained on; without them the agent behaves randomly.
"""

import argparse
import json
import os
import pickle

import numpy as np

from rl_common import DEFAULT_ENV_ID, env_from_config, env_label, resolve_latest


def parse_args():
    p = argparse.ArgumentParser(description="Render a trained PPO agent")
    p.add_argument("--env", type=str, default=DEFAULT_ENV_ID,
                   help="Gymnasium env id (used to find runs/<env>/latest)")
    p.add_argument("--run", type=str, default=None,
                   help="run folder to load (default: runs/<env>/latest)")
    p.add_argument("--model", type=str, default=None,
                   help="explicit path to a .zip model (overrides --run)")
    p.add_argument("--stats", type=str, default=None,
                   help="explicit path to the VecNormalize .pkl")
    p.add_argument("--best", action="store_true",
                   help="load best_model.zip instead of final_model.zip")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--stochastic", action="store_true",
                   help="sample actions instead of using the deterministic mean")
    return p.parse_args()


def resolve_run_dir(args):
    if args.run:
        return args.run
    return resolve_latest(env_label(args.env))


def trainer_of(run_dir):
    """'brax' for train_mjx.py runs, else 'sb3' (same logic as export_onnx)."""
    cfg = os.path.join(run_dir, "config.json")
    if os.path.exists(cfg):
        try:
            if json.load(open(cfg)).get("trainer") == "brax":
                return "brax"
        except Exception:
            pass
    if (os.path.exists(os.path.join(run_dir, "final_brax.pkl"))
            or os.path.exists(os.path.join(run_dir, "best_brax.pkl"))):
        return "brax"
    return "sb3"


def resolve_paths(args):
    """Return (model_path, stats_path, env_id, trainer) with fallbacks."""
    if args.model:
        run_dir = os.path.dirname(args.model) or "."
        env_id = env_from_config(run_dir) or args.env
        stats = args.stats or os.path.join(run_dir, "vecnormalize.pkl")
        trainer = "brax" if args.model.endswith(".pkl") else "sb3"
        return args.model, stats, env_id, trainer

    run_dir = resolve_run_dir(args)
    env_id = env_from_config(run_dir) or args.env
    trainer = trainer_of(run_dir)
    if trainer == "brax":
        model_name = "best_brax.pkl" if args.best else "final_brax.pkl"
    else:
        model_name = "best_model.zip" if args.best else "final_model.zip"
    model_path = os.path.join(run_dir, model_name)
    stats_path = args.stats or os.path.join(run_dir, "vecnormalize.pkl")
    return model_path, stats_path, env_id, trainer


class BraxNumpyPolicy:
    """Deterministic tanh_normal actor from a train_mjx.py pickle, as plain
    numpy (no torch/jax needed to watch a policy)."""

    ACTS = {
        "swish": lambda x: x / (1.0 + np.exp(-x)),
        "silu": lambda x: x / (1.0 + np.exp(-x)),
        "tanh": np.tanh,
        "relu": lambda x: np.maximum(x, 0.0),
        "elu": lambda x: np.where(x >= 0, x, np.exp(x) - 1.0),
    }

    def __init__(self, payload):
        norm = payload["normalizer"]
        self.mean = np.asarray(norm["mean"], dtype=np.float64)
        self.std = np.asarray(norm["std"], dtype=np.float64)
        self.act_dim = payload["act_dim"]
        self.act_fn = self.ACTS[payload["activation"]]
        tree = payload["policy_params"].get("params", payload["policy_params"])
        self.layers = [
            (np.asarray(tree[n]["kernel"], dtype=np.float64),
             np.asarray(tree[n]["bias"], dtype=np.float64))
            for n in sorted(tree, key=lambda n: int(n.split("_")[-1]))
        ]

    def act(self, obs):
        x = (np.asarray(obs, dtype=np.float64) - self.mean) / self.std
        for i, (kernel, bias) in enumerate(self.layers):
            x = x @ kernel + bias
            if i < len(self.layers) - 1:
                x = self.act_fn(x)
        return np.tanh(x[:self.act_dim])


def run_brax(model_path, env_id, episodes):
    import gymnasium as gym

    with open(model_path, "rb") as f:
        payload = pickle.load(f)
    policy = BraxNumpyPolicy(payload)
    env = gym.make(env_id, render_mode="human")
    for ep in range(episodes):
        obs, _ = env.reset()
        done, ep_reward, ep_len = False, 0.0, 0
        while not done:
            obs, reward, terminated, truncated, _ = env.step(policy.act(obs))
            done = terminated or truncated
            ep_reward += float(reward)
            ep_len += 1
        print(f"episode {ep + 1}/{episodes}: length={ep_len}, reward={ep_reward:.1f}")
    env.close()


def main():
    args = parse_args()
    model_path, stats_path, env_id, trainer = resolve_paths(args)
    if not os.path.exists(model_path):
        raise SystemExit(
            f"No model found at {model_path}. Train one first: "
            f"python train.py --env {args.env}")
    print(f"Env: {env_id}")
    print(f"Loading model: {model_path}")

    if trainer == "brax":
        # Brax policies are deterministic here (tanh of the Gaussian mean).
        run_brax(model_path, env_id, args.episodes)
        return

    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import VecNormalize

    # Single env with an on-screen viewer window.
    env = make_vec_env(env_id, n_envs=1, env_kwargs={"render_mode": "human"})

    # Restore the training normalization stats and freeze them for inference.
    if os.path.exists(stats_path):
        env = VecNormalize.load(stats_path, env)
        env.training = False
        env.norm_reward = False
    else:
        print(f"[warn] no VecNormalize stats at {stats_path}; "
              "the agent will likely behave poorly.")

    model = PPO.load(model_path, env=env)

    for ep in range(args.episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        ep_len = 0
        while not done:
            action, _ = model.predict(obs, deterministic=not args.stochastic)
            obs, reward, dones, infos = env.step(action)
            env.render()
            done = bool(dones[0])
            # norm_reward=False, so reward[0] is the raw environment reward.
            ep_reward += float(reward[0])
            ep_len += 1
        print(f"episode {ep + 1}/{args.episodes}: length={ep_len}, reward={ep_reward:.1f}")

    env.close()


if __name__ == "__main__":
    main()
