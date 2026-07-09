"""Read the best trial from an RL-Zoo/Optuna PPO study and turn it into
something you can actually run.

The Zoo's Optuna search samples params in a transformed space
(``batch_size_pow``, ``one_minus_gamma``, ``net_arch='medium'``, ...). This
script converts them back to real hyperparameters using the Zoo's own
``convert_onpolicy_params`` (so it stays correct across Zoo versions) and prints:

  * a summary of the study (trial counts + best value),
  * the winning hyperparameters,
  * ready-to-paste train.py CLI flags, and
  * (optionally) full multi-seed validation commands, and/or a ppo.yml block.

Typical use:
    # After a search on hyperparams/ppo_tune.yml:
    python best_params.py --study walker2d_ppo_tune

    # Emit the commands to validate the winner at full budget over 3 seeds:
    python best_params.py --study walker2d_ppo_tune --print-cmd --seeds 1 2 3

    # Emit a hyperparams/ppo.yml block to paste for Zoo-side validation:
    python best_params.py --study walker2d_ppo_tune --yaml
"""

import argparse

import optuna
from rl_zoo3.hyperparams_opt import convert_onpolicy_params

from rl_common import ACTIVATION_FNS

# torch.nn activation class name (e.g. "ReLU") -> the name train.py's
# --activation-fn expects (e.g. "relu"), derived from the shared mapping.
_ACTIVATION_NAMES = {cls.__name__: name for name, cls in ACTIVATION_FNS.items()}


def parse_args():
    p = argparse.ArgumentParser(description="Extract best PPO hyperparameters "
                                            "from an Optuna study")
    p.add_argument("--study", default="walker2d_ppo_tune",
                   help="Optuna study name")
    p.add_argument("--storage",
                   default="sqlite:///logs/ppo_walker2d_study.db",
                   help="Optuna storage URL")
    p.add_argument("--env", default="Walker2d-v5",
                   help="env id used in the emitted validation commands")
    p.add_argument("--print-cmd", action="store_true",
                   help="print full train.py validation commands")
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3],
                   help="seeds for the emitted validation commands")
    p.add_argument("--timesteps", type=int, default=3_000_000,
                   help="budget for the emitted validation commands")
    p.add_argument("--yaml", action="store_true",
                   help="also print a hyperparams/ppo.yml block for the winner")
    return p.parse_args()


def real_hyperparams(trial):
    """Best trial's sampled params -> real PPO hyperparameters."""
    hp = convert_onpolicy_params(dict(trial.params))
    # net_arch comes back as dict(pi=[...], vf=[...]); the Zoo's PPO space keeps
    # pi == vf, so a single comma-separated list round-trips cleanly.
    net_arch = hp["policy_kwargs"]["net_arch"]
    sizes = net_arch["pi"] if isinstance(net_arch, dict) else net_arch
    activation_cls = hp["policy_kwargs"]["activation_fn"].__name__
    return {
        "learning_rate": hp["learning_rate"],
        "n_steps": hp["n_steps"],
        "batch_size": hp["batch_size"],
        "n_epochs": hp["n_epochs"],
        "gamma": hp["gamma"],
        "gae_lambda": hp["gae_lambda"],
        "clip_range": hp["clip_range"],
        "ent_coef": hp["ent_coef"],
        "max_grad_norm": hp["max_grad_norm"],
        "net_arch": [int(x) for x in sizes],
        "activation_fn": _ACTIVATION_NAMES.get(activation_cls, activation_cls),
    }


def fmt(v):
    return f"{v:.6g}" if isinstance(v, float) else str(v)


def train_flags(hp):
    arch = ",".join(str(x) for x in hp["net_arch"])
    return (
        f"--learning-rate {fmt(hp['learning_rate'])} "
        f"--n-steps {hp['n_steps']} "
        f"--batch-size {hp['batch_size']} "
        f"--n-epochs {hp['n_epochs']} "
        f"--gamma {fmt(hp['gamma'])} "
        f"--gae-lambda {fmt(hp['gae_lambda'])} "
        f"--clip-range {fmt(hp['clip_range'])} "
        f"--ent-coef {fmt(hp['ent_coef'])} "
        f"--max-grad-norm {fmt(hp['max_grad_norm'])} "
        f"--net-arch {arch} "
        f"--activation-fn {hp['activation_fn']}"
    )


def yaml_block(env, hp):
    arch = ", ".join(str(x) for x in hp["net_arch"])
    act = f"nn.{ACTIVATION_FNS[hp['activation_fn']].__name__}"
    pk = f"dict(net_arch=[{arch}], activation_fn={act})"
    return "\n".join([
        f"{env}:",
        "  normalize: true",
        "  n_envs: 8",
        "  policy: 'MlpPolicy'",
        "  n_timesteps: !!float 3e6",
        f"  batch_size: {hp['batch_size']}",
        f"  n_steps: {hp['n_steps']}",
        f"  gamma: {fmt(hp['gamma'])}",
        f"  learning_rate: !!float {hp['learning_rate']:.6g}",
        f"  ent_coef: !!float {hp['ent_coef']:.6g}",
        f"  clip_range: {fmt(hp['clip_range'])}",
        f"  n_epochs: {hp['n_epochs']}",
        f"  gae_lambda: {fmt(hp['gae_lambda'])}",
        f"  max_grad_norm: {fmt(hp['max_grad_norm'])}",
        f"  policy_kwargs: \"{pk}\"",
    ])


def main():
    args = parse_args()
    try:
        study = optuna.load_study(study_name=args.study, storage=args.storage)
    except Exception as e:
        raise SystemExit(f"Could not load study '{args.study}' from "
                         f"{args.storage}: {e}")

    from collections import Counter
    states = Counter(t.state.name for t in study.trials)
    completed = [t for t in study.trials if t.value is not None]
    print(f"Study: {args.study}  ({args.storage})")
    print(f"Trials: {len(study.trials)}  {dict(states)}")
    if not completed:
        raise SystemExit("No completed trials yet — run the search first.")

    best = study.best_trial
    hp = real_hyperparams(best)
    print(f"Best trial: #{best.number}  eval_mean_reward = {best.value:.1f}\n")
    print("Winning hyperparameters:")
    for k, v in hp.items():
        print(f"  {k:>14}: {fmt(v) if not isinstance(v, list) else v}")

    print("\ntrain.py flags:")
    print(f"  {train_flags(hp)}")

    if args.print_cmd:
        print("\nValidation runs (full budget, fresh seeds):")
        for s in args.seeds:
            name = f"ppo_{args.env.split('-v')[0].lower()}_tuned_s{s}"
            print(f"  python train.py --env {args.env} "
                  f"--timesteps {args.timesteps} --seed {s} "
                  f"--run-name {name} --video-freq 0 \\\n"
                  f"    {train_flags(hp)}")

    if args.yaml:
        print("\nhyperparams/ppo.yml block:\n")
        print(yaml_block(args.env, hp))


if __name__ == "__main__":
    main()
