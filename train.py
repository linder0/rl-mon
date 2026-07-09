"""Train a PPO agent on MuJoCo Hopper-v5 using Stable-Baselines3.

Runs are grouped by environment, and each run gets its own timestamped folder:

    runs/<env>/<run-name>/          e.g. runs/hopper/ppo_20260708_132700/
      config.json          # hyperparameters, seed, versions, git commit, command
      log.txt              # human-readable training tables (the stdout dump)
      progress.csv         # every logged scalar, per iteration (easy to plot)
      events.out.tfevents  # TensorBoard scalars + rollout videos
      checkpoints/         # policy snapshots every --save-freq steps
      eval/evaluations.npz # eval returns over training
      best_model.zip       # best policy by eval reward
      final_model.zip      # policy at end of training
      vecnormalize.pkl     # obs/reward normalization stats (needed to reload)

Usage:
    python train.py                       # 1M steps on Hopper, tuned defaults
    python train.py --env Walker2d-v5     # a different MuJoCo task
    python train.py --timesteps 200000    # shorter run
    python train.py --run-name my_run     # fixed name instead of timestamp

Watch everything live with:
    tensorboard --logdir runs
"""

import argparse
import datetime as dt
import json
import os
import socket
import subprocess
import sys

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.logger import Video, configure
import torch

from rl_common import ACTIVATION_FNS, DEFAULT_ENV_ID, env_label


# Base hyperparameters: the RL-Baselines3-Zoo tuned config for the Hopper/Walker
# family. Defined once so we can both pass them to PPO and record them in
# config.json.
HYPERPARAMS = dict(
    learning_rate=3e-4,
    n_steps=512,
    batch_size=256,
    n_epochs=10,
    gamma=0.999,
    gae_lambda=0.99,
    clip_range=0.2,
    ent_coef=0.0,
    max_grad_norm=0.5,
    policy_kwargs=dict(net_arch=[256, 256]),
)

# Per-environment tuned overrides, applied on top of HYPERPARAMS and keyed by the
# short env label (e.g. "ant"). Ant is a higher-DOF 3D body: it does better with
# longer rollouts and the standard MuJoCo PPO discount/GAE (0.99 / 0.95) than
# Hopper's high 0.999 / 0.99. CLI flags still override these.
TUNED_BY_ENV = {
    "ant": dict(
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
    ),
}


class VideoRecorderCallback(BaseCallback):
    """Periodically roll out the current policy and log a video to TensorBoard.

    Renders in an offscreen ``rgb_array`` env (no window needed, works headless)
    and writes the clip under the "trajectory/video" tag so it shows up in the
    TensorBoard "Images" tab.
    """

    def __init__(self, eval_env, render_freq, n_eval_episodes=1, deterministic=True):
        super().__init__()
        self._eval_env = eval_env
        self._render_freq = render_freq
        self._n_eval_episodes = n_eval_episodes
        self._deterministic = deterministic
        self._last_log = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_log < self._render_freq:
            return True
        self._last_log = self.num_timesteps

        frames = []

        def grab_frame(_locals, _globals):
            frame = self._eval_env.render()
            if frame is not None:
                frames.append(frame)

        from stable_baselines3.common.evaluation import evaluate_policy

        # Sync the latest normalization stats into the eval env before rendering.
        if isinstance(self._eval_env, VecNormalize) and isinstance(self.model.get_vec_normalize_env(), VecNormalize):
            self._eval_env.obs_rms = self.model.get_vec_normalize_env().obs_rms

        evaluate_policy(
            self.model,
            self._eval_env,
            n_eval_episodes=self._n_eval_episodes,
            deterministic=self._deterministic,
            render=False,
            callback=grab_frame,
        )

        if frames:
            # (T, H, W, C) -> (1, T, C, H, W) tensor of uint8 that TB expects.
            video = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)[None]
            self.logger.record(
                "trajectory/video",
                Video(video, fps=30),
                exclude=("stdout", "log", "json", "csv"),
            )
        return True


def parse_args():
    p = argparse.ArgumentParser(description="PPO on MuJoCo control tasks")
    p.add_argument("--env", type=str, default=DEFAULT_ENV_ID,
                   help="Gymnasium env id (e.g. Hopper-v5, Walker2d-v5)")
    p.add_argument("--timesteps", type=int, default=1_000_000,
                   help="total environment steps to train for")
    p.add_argument("--n-envs", type=int, default=8,
                   help="number of parallel environments")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--logdir", type=str, default="runs",
                   help="root directory; runs are grouped as <logdir>/<env>/<run>")
    p.add_argument("--run-name", type=str, default=None,
                   help="run folder name (default: ppo_<timestamp>)")
    p.add_argument("--save-freq", type=int, default=50_000,
                   help="save a policy checkpoint every N steps")
    p.add_argument("--eval-freq", type=int, default=25_000,
                   help="evaluate the policy every N steps (0 disables eval)")
    p.add_argument("--eval-episodes", type=int, default=5,
                   help="episodes per evaluation")
    p.add_argument("--video-freq", type=int, default=50_000,
                   help="log a rollout video to TensorBoard every N steps "
                        "(0 disables video recording)")
    p.add_argument("--no-update-latest", action="store_true",
                   help="do not repoint runs/<env>/latest at this run "
                        "(use when running parallel jobs so they don't clobber it)")

    # Hyperparameter overrides. Default None means "use the tuned value in
    # HYPERPARAMS"; anything passed here wins. This is how best_params.py
    # validates a tuned config, and how you try one-off configs by hand,
    # without editing this file.
    hp = p.add_argument_group("hyperparameter overrides (default: tuned values)")
    hp.add_argument("--learning-rate", type=float, default=None)
    hp.add_argument("--n-steps", type=int, default=None)
    hp.add_argument("--batch-size", type=int, default=None)
    hp.add_argument("--n-epochs", type=int, default=None)
    hp.add_argument("--gamma", type=float, default=None)
    hp.add_argument("--gae-lambda", type=float, default=None)
    hp.add_argument("--clip-range", type=float, default=None)
    hp.add_argument("--ent-coef", type=float, default=None)
    hp.add_argument("--max-grad-norm", type=float, default=None)
    hp.add_argument("--net-arch", type=str, default=None,
                    help="comma-separated hidden sizes, e.g. '256,256' or '400,300'")
    hp.add_argument("--activation-fn", type=str, default=None,
                    choices=list(ACTIVATION_FNS),
                    help="policy activation (default: SB3's Tanh). Set this to "
                         "match a tuned config from best_params.py.")
    return p.parse_args()


def build_hyperparams(args):
    """Start from the base tuned HYPERPARAMS, layer any per-env tuned overrides,
    then apply CLI overrides (which win)."""
    hp = dict(HYPERPARAMS)
    hp.update(TUNED_BY_ENV.get(env_label(args.env), {}))
    overrides = {
        "learning_rate": args.learning_rate,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "n_epochs": args.n_epochs,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "clip_range": args.clip_range,
        "ent_coef": args.ent_coef,
        "max_grad_norm": args.max_grad_norm,
    }
    for key, val in overrides.items():
        if val is not None:
            hp[key] = val
    if args.net_arch is not None:
        sizes = [int(x) for x in args.net_arch.split(",") if x.strip()]
        hp["policy_kwargs"] = dict(hp.get("policy_kwargs", {}), net_arch=sizes)
    if args.activation_fn is not None:
        hp["policy_kwargs"] = dict(hp.get("policy_kwargs", {}),
                                   activation_fn=ACTIVATION_FNS[args.activation_fn])
    return hp


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def pkg_version(name):
    try:
        import importlib.metadata as md
        return md.version(name)
    except Exception:
        return None


def write_config(run_dir, args, run_name, hyperparams):
    """Record everything needed to understand/reproduce this run."""
    config = {
        "run_name": run_name,
        "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "command": " ".join(sys.argv),
        "env_id": args.env,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "git_commit": git_commit(),
        "args": vars(args),
        "hyperparameters": hyperparams,
        "versions": {
            "python": sys.version.split()[0],
            "stable_baselines3": pkg_version("stable_baselines3"),
            "gymnasium": pkg_version("gymnasium"),
            "mujoco": pkg_version("mujoco"),
            "torch": pkg_version("torch"),
            "numpy": pkg_version("numpy"),
        },
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2, default=str)
    return config


def make_norm_eval_env(env_id, seed):
    """Single offscreen env wrapped like training but with frozen, raw reward."""
    env = make_vec_env(env_id, n_envs=1, seed=seed,
                       env_kwargs={"render_mode": "rgb_array"})
    return VecNormalize(env, norm_obs=True, norm_reward=False,
                        clip_obs=10.0, training=False)


def update_latest_pointer(env_root, run_dir):
    """Refresh <env_root>/latest -> the current run dir for convenient reloading."""
    link = os.path.join(env_root, "latest")
    try:
        if os.path.islink(link) or os.path.exists(link):
            os.remove(link)
        os.symlink(os.path.abspath(run_dir), link)
    except OSError:
        # Symlinks may be unavailable (e.g. some filesystems); fall back to a file.
        with open(os.path.join(env_root, "latest.txt"), "w") as f:
            f.write(os.path.abspath(run_dir))


def main():
    args = parse_args()
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"ppo_{timestamp}"
    # Group runs by environment: <logdir>/<env>/<run-name>/
    env_root = os.path.join(args.logdir, env_label(args.env))
    run_dir = os.path.join(env_root, run_name)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)

    hyperparams = build_hyperparams(args)
    config = write_config(run_dir, args, run_name, hyperparams)
    print(f"Env: {args.env}")
    print(f"Run directory: {run_dir}")
    print(f"Device: {config['device']}")

    # Vectorized envs let PPO collect experience in parallel.
    env = make_vec_env(args.env, n_envs=args.n_envs, seed=args.seed)
    # Normalizing observations and rewards is a big lever on MuJoCo locomotion.
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO(policy="MlpPolicy", env=env, seed=args.seed, verbose=1,
                **hyperparams)

    # Log to stdout AND a persistent log.txt, progress.csv, and TensorBoard,
    # all inside the run folder. This is what makes every run self-documenting.
    logger = configure(run_dir, ["stdout", "log", "csv", "tensorboard"])
    model.set_logger(logger)

    callbacks = []

    # Policy snapshots on disk (weights + optimizer + normalization stats).
    callbacks.append(CheckpointCallback(
        save_freq=max(args.save_freq // args.n_envs, 1),
        save_path=os.path.join(run_dir, "checkpoints"),
        name_prefix=f"ppo_{env_label(args.env)}",
        save_vecnormalize=True,
    ))

    # Clean eval curve (eval/mean_reward) + best-model saving + evaluations.npz.
    if args.eval_freq > 0:
        eval_cb = EvalCallback(
            make_norm_eval_env(args.env, args.seed + 100),
            best_model_save_path=run_dir,
            log_path=os.path.join(run_dir, "eval"),
            eval_freq=max(args.eval_freq // args.n_envs, 1),
            n_eval_episodes=args.eval_episodes,
            deterministic=True,
            render=False,
        )
        callbacks.append(eval_cb)

    # Rollout videos to TensorBoard.
    if args.video_freq > 0:
        callbacks.append(VideoRecorderCallback(
            make_norm_eval_env(args.env, args.seed + 200),
            render_freq=args.video_freq,
        ))

    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    # Final policy + normalization stats live alongside the rest of the run.
    model.save(os.path.join(run_dir, "final_model.zip"))
    env.save(os.path.join(run_dir, "vecnormalize.pkl"))
    if not args.no_update_latest:
        update_latest_pointer(env_root, run_dir)

    print(f"\nAll artifacts recorded in: {run_dir}")
    print(f"  config.json / log.txt / progress.csv / events.* (TensorBoard)")
    print(f"  checkpoints/  best_model.zip  final_model.zip  vecnormalize.pkl")
    print(f"Reload the latest run with: python enjoy.py --env {args.env}")


if __name__ == "__main__":
    main()
