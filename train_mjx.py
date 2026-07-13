"""Train PPO on GPU with Brax + MJX, writing the same run-folder contract as
train.py so export_onnx.py and the web viewer work unchanged.

    runs/<env>/<run-name>/
      config.json          # + "trainer": "brax", jax/brax versions, device
      log.txt              # human-readable eval progress lines
      progress.csv         # time/total_timesteps + reward/diagnostic columns
      eval/evaluations.npz # eval curve (mean/std encoded as two columns)
      checkpoints/         # params snapshot at every eval
      best_brax.pkl        # best params by eval reward (pure-numpy pickle)
      final_brax.pkl       # params at end of training

The envs are the faithful gymnasium-v5 ports in mjx_envs/ (validated by
mjx_envs/validate.py), so exported policies keep the exact observation
contract the browser rebuilds.

Learning rate is ADAPTIVE_KL by default: Brax adjusts the LR each update to
hold a target KL (desired_kl), so there is no decay schedule to hand-tune.

Usage (on a GPU box; refuses to run on CPU unless --allow-cpu):
    python train_mjx.py --env Walker2d-v5
    python train_mjx.py --env Ant-v5 --num-envs 8192 --timesteps 100000000
    python train_mjx.py --env Hopper-v5 --timesteps 2000000 --allow-cpu  # smoke
"""

import argparse
import csv
import datetime as dt
import functools
import json
import os
import pickle
import socket
import subprocess
import sys

import numpy as np

from rl_common import DEFAULT_ENV_ID, env_label

# Columns written to progress.csv. Superset-compatible with what
# export_onnx.py's read_stats() looks for; missing values stay empty.
CSV_FIELDS = [
    "time/total_timesteps",
    "rollout/ep_rew_mean",
    "rollout/ep_len_mean",
    "eval/mean_reward",
    "eval/mean_ep_length",
    "train/learning_rate",
    "train/approx_kl",
    "train/value_loss",
    "train/entropy_loss",
    "train/policy_gradient_loss",
]

# Activation functions by name; the name is recorded in config.json and the
# .pkl so export_onnx.py can rebuild the network in torch (SiLU/Tanh/...).
def _activations():
    import jax
    import jax.numpy as jnp
    return {
        "swish": jax.nn.silu,
        "silu": jax.nn.silu,
        "tanh": jnp.tanh,
        "relu": jax.nn.relu,
        "elu": jax.nn.elu,
    }


# Brax-native tuned defaults per env label. These are NOT translations of the
# SB3 hyperparameters (different rollout structure, thousands of envs); they
# start from the Brax/MJX locomotion recipes. reward_scaling replaces SB3's
# VecNormalize reward normalization and is a per-env constant.
BRAX_HYPERPARAMS = dict(
    num_timesteps=50_000_000,
    num_envs=2048,
    unroll_length=10,
    batch_size=1024,
    num_minibatches=32,
    num_updates_per_batch=8,
    learning_rate=3e-4,
    lr_schedule="adaptive_kl",
    desired_kl=0.01,
    lr_min=1e-5,
    lr_max=1e-2,
    entropy_cost=1e-2,
    discounting=0.99,
    gae_lambda=0.95,
    clipping_epsilon=0.3,
    max_grad_norm=1.0,
    vf_coef=0.5,
    reward_scaling=5.0,
    episode_length=1000,
    net_arch=[128, 128, 128, 128],
    value_net_arch=[256, 256, 256, 256, 256],
    activation="swish",
)

TUNED_BY_ENV = {
    "hopper": dict(reward_scaling=10.0),
    "walker2d": dict(reward_scaling=5.0),
    # Ant: the lever is envs-at-scale. Contact-heavy, needs the larger batch.
    "ant": dict(num_timesteps=100_000_000, num_envs=4096, unroll_length=5,
                batch_size=2048, num_updates_per_batch=4,
                discounting=0.97, reward_scaling=10.0),
    # AntFood: same body/contact profile as Ant, so reuse its recipe. The
    # foraging task adds a sparse pickup bonus on top of a dense approach
    # reward, so keep reward_scaling modest to avoid value-target spikes.
    "antfood": dict(num_timesteps=100_000_000, num_envs=4096, unroll_length=5,
                    batch_size=2048, num_updates_per_batch=4,
                    discounting=0.97, reward_scaling=5.0),
    # AntFood2Leg: same recipe as AntFood. Meant to be warm-started from a
    # trained forager (--init-from), so its default budget is shorter.
    "antfood2leg": dict(num_timesteps=50_000_000, num_envs=4096, unroll_length=5,
                        batch_size=2048, num_updates_per_batch=4,
                        discounting=0.97, reward_scaling=5.0),
    # AntGetUp: self-righting from a fallen start. Dense per-step reward (~0..2.5)
    # so keep reward_scaling modest; harder exploration, so give it more steps.
    "antgetup": dict(num_timesteps=100_000_000, num_envs=4096, unroll_length=5,
                     batch_size=2048, num_updates_per_batch=4,
                     discounting=0.97, reward_scaling=1.0),
}


def parse_args():
    p = argparse.ArgumentParser(description="Brax PPO on MJX ports of the gym envs")
    p.add_argument("--env", type=str, default=DEFAULT_ENV_ID)
    p.add_argument("--timesteps", type=int, default=None,
                   help="total env steps (default: per-env tuned budget)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--logdir", type=str, default="runs")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--num-evals", type=int, default=20,
                   help="number of eval points across training")
    p.add_argument("--num-eval-envs", type=int, default=128)
    p.add_argument("--no-update-latest", action="store_true")
    p.add_argument("--init-from", type=str, default=None,
                   help="warm-start policy/value/normalizer from a *_brax.pkl "
                        "(same observation size); e.g. fine-tune a new task "
                        "from an existing forager")
    p.add_argument("--allow-cpu", action="store_true",
                   help="skip the GPU pre-flight check (VERY slow; smoke tests only)")

    hp = p.add_argument_group("hyperparameter overrides (default: tuned values)")
    hp.add_argument("--num-envs", type=int, default=None)
    hp.add_argument("--unroll-length", type=int, default=None)
    hp.add_argument("--batch-size", type=int, default=None)
    hp.add_argument("--num-minibatches", type=int, default=None)
    hp.add_argument("--num-updates-per-batch", type=int, default=None)
    hp.add_argument("--learning-rate", type=float, default=None)
    hp.add_argument("--lr-schedule", choices=["adaptive_kl", "none"], default=None)
    hp.add_argument("--desired-kl", type=float, default=None)
    hp.add_argument("--entropy-cost", type=float, default=None)
    hp.add_argument("--discounting", type=float, default=None)
    hp.add_argument("--gae-lambda", type=float, default=None)
    hp.add_argument("--clipping-epsilon", type=float, default=None)
    hp.add_argument("--max-grad-norm", type=float, default=None)
    hp.add_argument("--reward-scaling", type=float, default=None)
    hp.add_argument("--episode-length", type=int, default=None)
    hp.add_argument("--net-arch", type=str, default=None,
                    help="comma-separated policy hidden sizes, e.g. '128,128,128,128'")
    hp.add_argument("--value-net-arch", type=str, default=None)
    hp.add_argument("--activation", choices=["swish", "silu", "tanh", "relu", "elu"],
                    default=None)
    return p.parse_args()


def build_hyperparams(args):
    hp = dict(BRAX_HYPERPARAMS)
    hp.update(TUNED_BY_ENV.get(env_label(args.env), {}))
    for key in ("num_envs", "unroll_length", "batch_size", "num_minibatches",
                "num_updates_per_batch", "learning_rate", "lr_schedule",
                "desired_kl", "entropy_cost", "discounting", "gae_lambda",
                "clipping_epsilon", "max_grad_norm", "reward_scaling",
                "episode_length", "activation"):
        val = getattr(args, key)
        if val is not None:
            hp[key] = val
    if args.timesteps is not None:
        hp["num_timesteps"] = args.timesteps
    for flag, key in (("net_arch", "net_arch"), ("value_net_arch", "value_net_arch")):
        raw = getattr(args, flag)
        if raw is not None:
            hp[key] = [int(x) for x in raw.split(",") if x.strip()]

    if (hp["batch_size"] * hp["num_minibatches"]) % hp["num_envs"] != 0:
        raise SystemExit(
            f"batch_size*num_minibatches ({hp['batch_size']}*{hp['num_minibatches']}) "
            f"must be divisible by num_envs ({hp['num_envs']}) — Brax requirement.")
    return hp


def preflight_device(allow_cpu):
    """Refuse to run on CPU: a silently-degraded driver/CUDA mismatch would
    train ~1000x slower and waste the whole GPU rental."""
    import jax
    dev = jax.devices()[0]
    platform = dev.platform.lower()
    if platform in ("gpu", "cuda", "rocm", "tpu"):
        return f"{platform}:{getattr(dev, 'device_kind', dev)}"
    msg = (f"JAX sees only '{platform}' devices ({dev}). On a GPU box this "
           "means the CUDA install/driver is broken — fix it (pip install "
           "'jax[cuda12]', check nvidia-smi) instead of training on CPU.")
    if not allow_cpu:
        raise SystemExit("PRE-FLIGHT FAILED: " + msg + "\n(Use --allow-cpu for smoke tests only.)")
    print(f"[warn] {msg}\n[warn] --allow-cpu set: continuing anyway.")
    return f"{platform}:{getattr(dev, 'device_kind', dev)}"


def resolve_lr_schedule(hp):
    """Map our lr_schedule name to Brax kwargs, with a capability check and a
    constant-LR fallback for Brax versions predating ADAPTIVE_KL."""
    if hp["lr_schedule"] == "none":
        return {}
    try:
        from brax.training.agents.ppo import optimizer as ppo_optimizer
        schedule = ppo_optimizer.LRSchedule.ADAPTIVE_KL
    except (ImportError, AttributeError):
        print("[warn] this Brax version has no ADAPTIVE_KL learning-rate "
              "schedule; falling back to a constant learning rate. "
              "Install brax>=0.14.2 for adaptive LR.")
        hp["lr_schedule"] = "none (fallback: brax too old)"
        return {}
    return dict(
        learning_rate_schedule=schedule,
        desired_kl=hp["desired_kl"],
        learning_rate_schedule_min_lr=hp["lr_min"],
        learning_rate_schedule_max_lr=hp["lr_max"],
    )


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


def write_config(run_dir, args, run_name, hyperparams, device, task_spec=None):
    config = {
        "run_name": run_name,
        "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "command": " ".join(sys.argv),
        "env_id": args.env,
        "trainer": "brax",
        "device": device,
        "git_commit": git_commit(),
        "args": vars(args),
        # The env's distinctive task parameters (reward/reach/food/upright/...),
        # so each run records the exact task it trained on — not just the PPO
        # hyperparameters. Keeps variants and tweaks distinguishable.
        "task_spec": task_spec,
        "hyperparameters": hyperparams,
        "versions": {
            "python": sys.version.split()[0],
            "jax": pkg_version("jax"),
            "brax": pkg_version("brax"),
            "mujoco": pkg_version("mujoco"),
            "mujoco_mjx": pkg_version("mujoco-mjx"),
            "gymnasium": pkg_version("gymnasium"),
            "numpy": pkg_version("numpy"),
        },
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2, default=str)
    return config


def to_numpy_tree(tree):
    """flax/jax pytree -> plain nested dicts of numpy arrays (unpicklable
    anywhere, no jax/flax needed on the loading side)."""
    import jax
    tree = jax.device_get(tree)

    def convert(x):
        if hasattr(x, "items"):  # dict / FrozenDict
            return {k: convert(v) for k, v in x.items()}
        return np.asarray(x)

    return convert(tree)


def params_payload(params, env_id, hp, obs_dim, act_dim):
    """Serialize Brax PPO params (normalizer, policy, value) as pure numpy."""
    normalizer, policy_params, value_params = params
    count = normalizer.count
    if hasattr(count, "to_numpy"):  # brax UInt64 (hi/lo 32-bit pair)
        count = count.to_numpy()
    return {
        "trainer": "brax_ppo",
        "env_id": env_id,
        "obs_dim": int(obs_dim),
        "act_dim": int(act_dim),
        "distribution": "tanh_normal",
        "activation": hp["activation"],
        "policy_hidden": list(hp["net_arch"]),
        "value_hidden": list(hp["value_net_arch"]),
        "normalizer": {
            "mean": np.asarray(normalizer.mean, dtype=np.float64),
            "std": np.asarray(normalizer.std, dtype=np.float64),
            "count": float(np.asarray(count)),
        },
        "policy_params": to_numpy_tree(policy_params),
        "value_params": to_numpy_tree(value_params),
    }


def load_restore_params(path, network_factory, obs_dim, act_dim):
    """Rebuild Brax's ``(normalizer, policy, value)`` restore tuple from one of
    our numpy pickles (see params_payload). Brax feeds this to ppo.train's
    ``restore_params`` to warm-start the TrainingState.

    The policy/value trees are reconstructed with the *exact* container types a
    fresh Brax init produces — by unflattening our saved leaves into a fresh
    init's treedef — so the optimizer state (built from the fresh init) still
    lines up. The normalizer is rebuilt from the saved mean/std/count."""
    import jax
    import jax.numpy as jnp
    from brax.training import types as btypes
    from brax.training.acme import running_statistics

    with open(path, "rb") as f:
        payload = pickle.load(f)
    saved_obs = int(payload.get("obs_dim", obs_dim))
    if saved_obs != obs_dim:
        raise SystemExit(
            f"--init-from {path}: source obs_dim {saved_obs} != env obs_dim "
            f"{obs_dim}. Warm-start needs a source trained on the same "
            "observation.")

    ppo_network = network_factory(obs_dim, act_dim)
    key = jax.random.PRNGKey(0)

    def match(fresh, saved):
        # Both trees have identical (sorted-key) structure, so their flattened
        # leaves line up; unflatten into the fresh treedef to get Brax's type.
        leaves, _ = jax.tree_util.tree_flatten(saved)
        _, treedef = jax.tree_util.tree_flatten(fresh)
        return jax.tree_util.tree_unflatten(
            treedef, [jnp.asarray(x, jnp.float32) for x in leaves])

    policy = match(ppo_network.policy_network.init(key), payload["policy_params"])
    value = match(ppo_network.value_network.init(key), payload["value_params"])

    n = payload["normalizer"]
    mean = jnp.asarray(n["mean"], jnp.float32)
    std = jnp.asarray(n["std"], jnp.float32)
    cnt = int(n["count"])
    base = running_statistics.init_state(jnp.zeros((obs_dim,), jnp.float32))
    # Welford std = sqrt(summed_variance / count); invert to stay consistent
    # so the first post-restore update recomputes the same std.
    normalizer = base.replace(
        count=btypes.UInt64(hi=cnt >> 32, lo=cnt & 0xFFFFFFFF),
        mean=mean, std=std,
        summed_variance=std * std * jnp.float32(max(cnt, 1)))
    return (normalizer, policy, value)


def update_latest_pointer(env_root, run_dir):
    link = os.path.join(env_root, "latest")
    try:
        if os.path.islink(link) or os.path.exists(link):
            os.remove(link)
        os.symlink(os.path.abspath(run_dir), link)
    except OSError:
        with open(os.path.join(env_root, "latest.txt"), "w") as f:
            f.write(os.path.abspath(run_dir))


def main():
    args = parse_args()
    device = preflight_device(args.allow_cpu)

    import jax  # after pre-flight so the failure message is ours
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as ppo_train

    from mjx_envs import make_env

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"ppo_mjx_{timestamp}"
    env_root = os.path.join(args.logdir, env_label(args.env))
    run_dir = os.path.join(env_root, run_name)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "eval"), exist_ok=True)

    hp = build_hyperparams(args)
    lr_kwargs = resolve_lr_schedule(hp)

    env = make_env(args.env)
    eval_env = make_env(args.env)
    obs_dim, act_dim = env.observation_size, env.action_size

    config = write_config(run_dir, args, run_name, hp, device,
                          task_spec=env.task_spec())
    log_path = os.path.join(run_dir, "log.txt")

    def log(msg):
        print(msg)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    log(f"Env: {args.env} (MJX port)")
    log(f"Run directory: {run_dir}")
    log(f"Device: {device}")
    log(f"LR schedule: {hp['lr_schedule']}")

    activation = _activations()[hp["activation"]]
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=tuple(hp["net_arch"]),
        value_hidden_layer_sizes=tuple(hp["value_net_arch"]),
        activation=activation,
    )

    restore_params = None
    if args.init_from:
        restore_params = load_restore_params(
            args.init_from, network_factory, obs_dim, act_dim)
        log(f"Warm-starting policy/value/normalizer from: {args.init_from}")

    # -- callbacks wired into Brax's train loop -------------------------------
    csv_path = os.path.join(run_dir, "progress.csv")
    with open(csv_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()

    eval_t, eval_mean, eval_std, eval_len = [], [], [], []
    state = {"latest_params": None, "best_reward": -np.inf}

    def save_pkl(path, params):
        with open(path, "wb") as f:
            pickle.dump(params_payload(params, args.env, hp, obs_dim, act_dim), f)

    def policy_params_fn(current_step, make_policy, params):
        state["latest_params"] = params
        ckpt = os.path.join(run_dir, "checkpoints",
                            f"ppo_{env_label(args.env)}_{current_step}_steps.pkl")
        save_pkl(ckpt, params)

    def progress_fn(num_steps, metrics):
        get = lambda k: (float(metrics[k]) if k in metrics else None)
        reward = get("eval/episode_reward")
        reward_std = get("eval/episode_reward_std") or 0.0
        ep_len = get("eval/avg_episode_length")

        row = {
            "time/total_timesteps": num_steps,
            "rollout/ep_rew_mean": reward,
            "rollout/ep_len_mean": ep_len,
            "eval/mean_reward": reward,
            "eval/mean_ep_length": ep_len,
            "train/learning_rate": get("training/learning_rate"),
            "train/approx_kl": get("training/kl_mean"),
            "train/value_loss": get("training/v_loss"),
            "train/entropy_loss": get("training/entropy_loss"),
            "train/policy_gradient_loss": get("training/policy_loss"),
        }
        with open(csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(
                {k: ("" if v is None else v) for k, v in row.items()})

        if reward is None:
            return
        # Encode mean±std as two synthetic columns so read_stats()'s
        # results.mean(axis=1)/std(axis=1) reproduce the true mean and band.
        eval_t.append(num_steps)
        eval_mean.append([reward - reward_std, reward + reward_std])
        eval_len.append([ep_len or 0.0, ep_len or 0.0])
        np.savez(os.path.join(run_dir, "eval", "evaluations.npz"),
                 timesteps=np.asarray(eval_t, dtype=np.int64),
                 results=np.asarray(eval_mean, dtype=np.float64),
                 ep_lengths=np.asarray(eval_len, dtype=np.float64))

        log(f"steps {num_steps:>12,}  eval_reward {reward:9.1f} ± {reward_std:6.1f}  "
            f"ep_len {ep_len or 0:6.1f}"
            + (f"  lr {row['train/learning_rate']:.2e}"
               if row["train/learning_rate"] else ""))

        if state["latest_params"] is not None and reward > state["best_reward"]:
            state["best_reward"] = reward
            save_pkl(os.path.join(run_dir, "best_brax.pkl"), state["latest_params"])

    # -- train -----------------------------------------------------------------
    log(f"Training {hp['num_timesteps']:,} steps, num_envs={hp['num_envs']} ...")
    make_policy, params, _ = ppo_train.train(
        environment=env,
        eval_env=eval_env,
        num_timesteps=hp["num_timesteps"],
        num_envs=hp["num_envs"],
        episode_length=hp["episode_length"],
        unroll_length=hp["unroll_length"],
        batch_size=hp["batch_size"],
        num_minibatches=hp["num_minibatches"],
        num_updates_per_batch=hp["num_updates_per_batch"],
        learning_rate=hp["learning_rate"],
        entropy_cost=hp["entropy_cost"],
        discounting=hp["discounting"],
        gae_lambda=hp["gae_lambda"],
        clipping_epsilon=hp["clipping_epsilon"],
        max_grad_norm=hp["max_grad_norm"],
        vf_loss_coefficient=hp["vf_coef"],
        reward_scaling=hp["reward_scaling"],
        normalize_observations=True,
        network_factory=network_factory,
        num_evals=max(args.num_evals, 2),
        num_eval_envs=args.num_eval_envs,
        deterministic_eval=True,
        seed=args.seed,
        progress_fn=progress_fn,
        policy_params_fn=policy_params_fn,
        restore_params=restore_params,
        **lr_kwargs,
    )

    save_pkl(os.path.join(run_dir, "final_brax.pkl"), params)
    if state["latest_params"] is None:  # tiny runs may never eval mid-training
        save_pkl(os.path.join(run_dir, "best_brax.pkl"), params)
    if not args.no_update_latest:
        update_latest_pointer(env_root, run_dir)

    log(f"\nAll artifacts recorded in: {run_dir}")
    log("  config.json / log.txt / progress.csv / eval/evaluations.npz")
    log("  checkpoints/  best_brax.pkl  final_brax.pkl")
    log(f"Best eval reward: {state['best_reward']:.1f}")
    log(f"Export for the web viewer with: python export_onnx.py --env {args.env}")


if __name__ == "__main__":
    main()
