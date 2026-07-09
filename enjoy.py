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
import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

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


def resolve_paths(args):
    """Return (model_path, stats_path, env_id) with sensible fallbacks."""
    if args.model:
        run_dir = os.path.dirname(args.model) or "."
        env_id = env_from_config(run_dir) or args.env
        stats = args.stats or os.path.join(run_dir, "vecnormalize.pkl")
        return args.model, stats, env_id

    run_dir = resolve_run_dir(args)
    env_id = env_from_config(run_dir) or args.env
    model_name = "best_model.zip" if args.best else "final_model.zip"
    model_path = os.path.join(run_dir, model_name)
    stats_path = args.stats or os.path.join(run_dir, "vecnormalize.pkl")
    return model_path, stats_path, env_id


def main():
    args = parse_args()
    model_path, stats_path, env_id = resolve_paths(args)
    if not os.path.exists(model_path):
        raise SystemExit(
            f"No model found at {model_path}. Train one first: "
            f"python train.py --env {args.env}")
    print(f"Env: {env_id}")
    print(f"Loading model: {model_path}")

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
