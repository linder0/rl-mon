# Models & environments

This project trains PPO policies (Brax/MJX, `train_mjx.py`) on a small family of
MuJoCo control tasks and serves them in the browser viewer (`web-next/`). This
doc catalogs the **environments** (the tasks) and explains how each **run**
records exactly what it trained on, so different training experiments stay
distinguishable.

All envs live in `mjx_envs/locomotion.py`. Two groups:

- **Faithful gym ports** — byte-for-byte replicas of the gymnasium `-v5`
  observation/reward/termination, gated by `mjx_envs/validate.py`. Policies
  transfer to C-MuJoCo (`enjoy.py`, the web viewer) unchanged.
- **Custom tasks** — new tasks with no gymnasium equivalent. Deliberately kept
  out of the parity gate. The "food" target is a code-side concept (in the env
  and mirrored in the viewer), **not** a body in the MJCF.

## Environment catalog

| env id | class | obs dim | task / reward | termination |
|---|---|---|---|---|
| `Hopper-v5` | `HopperMjx` | 11 | forward x-velocity + survive − ctrl | z, angle, state out of range |
| `Walker2d-v5` | `Walker2dMjx` | 17 | forward x-velocity + survive − ctrl | z, angle out of range |
| `Ant-v5` | `AntMjx` | 105 | forward x-velocity + survive − ctrl − contact | torso z ∉ [0.2, 1.0] |
| `AntFood-v5` | `AntFoodMjx` | 107 | approach food + pickup bonus + survive − ctrl − contact | torso z ∉ [0.2, 1.0] **or flipped** |
| `AntFood2Leg-v5` | `AntFood2LegMjx` | 107 | same, but pickup needs ≥2 feet on the food | torso z ∉ [0.2, 1.0] **or flipped** |
| `Monster-<name>-v0` | `MonsterMjx` | varies | Ant's task on a generated morphology | geometry-derived z range **or flipped** |

### Faithful ports (Hopper / Walker2d / Ant)

Standard gymnasium `-v5` locomotion. Reward = `forward_reward_weight *
x_velocity + healthy_reward − ctrl_cost` (Ant adds a `contact_cost`). Obs is a
qpos slice + qvel (Ant also appends clipped `cfrc_ext` contact forces). See the
class docstrings for the exact health checks. Do **not** change these envs'
`obs_from_data` / `is_healthy` / `reward_terms` without re-running the parity
gate — that is the whole point of them.

### AntFood — forage by touching food

`AntFoodMjx` (subclass of `AntMjx`). A 2-D "food" target is spawned on an
annulus around the torso (`spawn_radius_range`, default 3–8 m) and carried in
`state.info["food"]`. Reward replaces Ant's pure forward term with:

- **approach**: `food_reward_weight * (closed distance / dt)` — dense, guides
  the ant to the food,
- **pickup bonus**: `food_bonus` when collected, then the food respawns nearby,
- plus Ant's survive / ctrl / contact terms.

Collection (`_reached`) is **torso within `reach_radius`** (default 1.0 m). The
observation is the standard Ant obs with the food's position **relative to the
torso** appended (+2 dims → 107).

### AntFood2Leg — forage by planting two feet

`AntFood2LegMjx` (subclass of `AntFoodMjx`). Identical to AntFood except
collection requires **≥ `min_feet` (default 2) of the four ankle geoms within
`foot_radius` (default 0.5 m)** of the food, instead of the torso being close.
Same 107-dim observation, so it can be **warm-started** from an AntFood policy
(`--init-from`). Meaningfully harder: the ant must straddle the target.

### Monsters — parametric morphologies

`MonsterMjx` runs Ant's locomotion task (forward x-velocity + survive − ctrl −
contact, Ant-style observation) on a body generated from a `MonsterSpec`
(`mjx_envs/monsters.py`): a torso plus limbs, each limb a chain of actuated
capsule segments with per-segment lengths, tilts, joint ranges, and motor
gears. `radial(n, leg)` / `bilateral(leg)` build symmetric layouts in one line.

Env ids are `Monster-<name>-v0`, where `<name>` is a preset (`quad`,
`hexapod`, `biped_tail`, `spider8`) or any spec written to
`assets/monsters/<name>.spec.json` — e.g. a random morphology from
`--sample`. Obs/action sizes vary per monster (one policy per monster). The
full spec is embedded in `task_spec`, so each run's `config.json` records the
exact body it trained on, and the export rebuilds that body from the run's
config (not the current presets).

```bash
python -m mjx_envs.monsters --list                 # presets + env ids
python -m mjx_envs.monsters --preview quad         # sanity check + viewer (macOS: mjpython)
python -m mjx_envs.monsters --sample --seed 3      # random monster -> assets/monsters/
python train_mjx.py --env Monster-quad-v0
python export_onnx.py --run runs/monster-quad/<run>
```

Health is Ant-style (finite state + torso z inside a range derived from the
spec's standing height, overridable via `healthy_z_range`) plus the anti-flip
upright check below. Monsters are custom envs (no gym counterpart, no parity
gate) but are pure MuJoCo, so exports keep the deterministic parity trace and
the C-MuJoCo sim2sim transfer check.

### Anti-flip (upright termination)

The food envs set `healthy_upright_min` (default 0.0): a run is only "healthy"
while the torso's local +z points generally upward (world-z component of
`xmat[main_body]` > threshold). A flipped ant becomes unhealthy → the episode
ends and resets, instead of flailing on its back (a known quirk of Ant, whose
height-only health check often stays satisfied when flipped). This only affects
**future training**; already-exported policies were trained without it. Set
`healthy_upright_min = None` to disable.

## How a run records its task (granularity)

Every training run writes `runs/<label>/<run_name>/config.json`, which captures:

- `command`, `git_commit`, `created`, `device`, package `versions`
- `args` (CLI, including `--init-from` for warm-starts)
- `hyperparameters` (the PPO recipe)
- **`task_spec`** — the env's distinctive constants (reward weights, reach
  radius, food bonus, spawn range, foot radius / min feet, upright threshold,
  health ranges, …), produced by `env.task_spec()`.

`task_spec` is the key to granularity: two runs of the same env id with
different task constants (e.g. `foot_radius` 0.5 vs 0.3) are otherwise
indistinguishable. It is also copied into the exported `*.stats.json`
(`config.task_spec`) so the viewer/analysis has it too.

When you tune a task, prefer either a **new env class** (best — self-documenting
and shows up as its own entry in the viewer) or, for a quick sweep, change the
constant and rely on `task_spec` + `git_commit` to tell the runs apart.

## Exporting to the viewer

```bash
# faithful ports (gym-introspected, sim2sim transfer check runs):
python export_onnx.py --env Ant-v5 --all

# custom tasks (routed through a bespoke path; obs +food dims, no C-MuJoCo food):
python export_onnx.py --run runs/antfood/<run> --out web-next/public
python export_onnx.py --run runs/antfood2leg/<run> --out web-next/public
```

The custom-env export reuses the gym-validated Ant base for the shared
physics/observation, appends the food observation, and (for AntFood2Leg) records
the ankle geom ids the browser needs to reproduce the two-foot test. Its parity
trace is synthetic (in-distribution obs → action) since C-MuJoCo has no food.

## Training (GPU)

See `LAMBDA.md` for the Lambda Cloud flow. Warm-start a new task from an
existing policy with the same observation size:

```bash
scripts/lambda.sh train <ip> AntFood2Leg-v5 --seed 1 --init-from init_forager.pkl
```

Note: `scripts/lambda.sh setup` excludes `runs/`, so copy the source checkpoint
to the box separately before a warm-start.
