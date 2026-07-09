# RL MuJoCo — PPO

Train MuJoCo control agents (Hopper by default) with
[PPO](https://arxiv.org/abs/1707.06347) via
[Stable-Baselines3](https://stable-baselines3.readthedocs.io/). Every run is
fully self-documenting.

## Setup

Modern MuJoCo ships as a pip wheel — no license, no separate install.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
python train.py                     # 1M steps on Hopper, tuned defaults
python train.py --env Walker2d-v5   # a different MuJoCo task
python train.py --env Ant-v5 --timesteps 5000000   # Ant uses its own tuned config
python train.py --timesteps 200000  # quicker run
```

Hyperparameters default to the tuned Hopper/Walker config, with per-environment
overrides in `TUNED_BY_ENV` (e.g. Ant uses longer rollouts and 0.99/0.95
discount/GAE). Any `--flag` still overrides the tuned value.

Live curves + rollout videos:

```bash
tensorboard --logdir runs
```

Override any tuned hyperparameter from the CLI (used by the sweep runner):

```bash
python train.py --env Walker2d-v5 --learning-rate 1e-3 --n-steps 2048 \
    --net-arch 400,300
```

## Watch it

```bash
python enjoy.py                     # latest Hopper run in a live viewer
python enjoy.py --env Walker2d-v5   # latest run of another env
python enjoy.py --best              # load the best checkpoint by eval reward
```

## Watch it in the browser

Run a trained policy live in the browser (MuJoCo compiled to WebAssembly, the
policy exported to ONNX, rendered with three.js) — much nicer-looking than the
MuJoCo viewer, and shareable as a static site. Export a policy, then start the
web app:

```bash
python export_onnx.py --all-envs           # export every run (final + best)
cd web && npm install && npm run dev       # http://localhost:5173
```

It's an all-in-one testing dashboard: a grouped Run picker (every run/seed and
its final/best variant), the live 3D rollout, and a per-run stats panel with
summary cards, eval/training reward curves, and the run's hyperparameters. See
[`web/README.md`](web/README.md) for details.

## Tune hyperparameters (RL Baselines3 Zoo + Optuna)

Hyperparameter search is handled by [RL Baselines3
Zoo](https://github.com/DLR-RM/rl-baselines3-zoo), the official SB3 companion.
It runs an **Optuna** study (Bayesian/TPE search) with **pruning** (kill bad
trials early), so it needs far less compute than a brute-force grid. The search
space is built into the Zoo — you don't hand-write it.

There are two config files:

- `hyperparams/ppo.yml` — the **validation/final** config (full `3e6` budget),
  mirrors `train.py`'s tuned values so Zoo runs are comparable to `runs/`.
- `hyperparams/ppo_tune.yml` — the **search** config (short `1e6` budget). Each
  Optuna trial trains a whole model, so the per-trial budget is the main cost
  lever; keep it short for the search, use the full budget only to validate.

The three-step loop is: **search → read off the winner → validate across seeds.**

```bash
# 1. Search: 100 trials, 4 in parallel, TPE sampler + median pruning.
#    Short per-trial budget (ppo_tune.yml); results persist in a SQLite DB.
python -m rl_zoo3.train --algo ppo --env Walker2d-v5 \
    -f logs --conf-file hyperparams/ppo_tune.yml \
    --optimize --n-trials 100 --n-jobs 4 \
    --sampler tpe --pruner median --n-evaluations 20 \
    --storage sqlite:///logs/ppo_walker2d_study.db --study-name walker2d_ppo_tune

# 2. Read the winner (converts Optuna's transformed params back to real ones):
python best_params.py --study walker2d_ppo_tune

# 3. Validate: emit full-budget train.py commands for fresh seeds, then run them.
#    These land in runs/ and work with enjoy.py; a single seed's peak is luck.
python best_params.py --study walker2d_ppo_tune --print-cmd --seeds 1 2 3
```

`best_params.py` can also emit a ready-to-paste `hyperparams/ppo.yml` block
(`--yaml`) if you'd rather validate through the Zoo instead of `train.py`.

The search's per-trial output and a report CSV are written under `logs/ppo/`.
Watch all trials/seeds in TensorBoard with `tensorboard --logdir logs`.

`train.py` also accepts direct hyperparameter overrides
(`--learning-rate`, `--n-steps`, `--net-arch 400,300`, …) for quick one-off
manual experiments without the Zoo.

## How runs are organized

Runs are grouped by environment; each run gets its own timestamped folder:

```
runs/
  <env>/                         # e.g. hopper/, walker2d/
    latest -> <most recent run>  # symlink used by enjoy.py
    ppo_<timestamp>/
      config.json                # hyperparameters, seed, versions, git commit, command
      log.txt                    # human-readable training tables
      progress.csv               # every scalar per iteration (incl. eval/mean_reward)
      events.out.tfevents        # TensorBoard scalars + rollout videos
      checkpoints/               # policy snapshots every --save-freq steps
      eval/evaluations.npz       # eval returns over training
      best_model.zip             # best policy by eval reward
      final_model.zip            # policy at end of training
      vecnormalize.pkl           # obs/reward normalization stats (needed to reload)
```

## Files

- `train.py` — training entry point (vectorized envs, `VecNormalize`, tuned PPO,
  checkpoints, eval curve, TensorBoard videos, per-run documentation). Accepts
  hyperparameter overrides via CLI flags.
- `hyperparams/ppo.yml` — RL Baselines3 Zoo config for validation/final runs
  (full budget), mirroring `train.py`.
- `hyperparams/ppo_tune.yml` — Zoo config for the Optuna search phase (short
  per-trial budget).
- `best_params.py` — read the best trial from an Optuna study and emit train.py
  flags / validation commands / a `ppo.yml` block.
- `rl_common.py` — shared helpers (env-name utilities, run resolution, the
  activation-function map) used by the scripts above.
- `enjoy.py` — load a trained policy and render it (auto-detects the env from the
  run's `config.json`).
- `requirements.txt` — pinned dependencies.

## What to expect

- Hopper is considered "solved" around a return of **~3000+**.
- Watch `rollout/ep_rew_mean` and `eval/mean_reward` climb in TensorBoard.
