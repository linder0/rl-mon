"""Export trained PPO policies to browser assets for the web/ viewer.

Turns runs under runs/<env>/ into everything the in-browser dashboard needs:

    web/public/policies/<id>.onnx        # policy MLP with obs-normalization baked in
    web/public/policies/<id>.json        # sim spec + parity trace (to reproduce the rollout)
    web/public/policies/<id>.stats.json  # config, hyperparameters, training/eval curves
    web/public/models/<model>.xml        # the MuJoCo model (physics + geometry)
    web/public/policies/index.json       # grouped catalog the app reads

where <id> = "<env>__<run_name>__<final|best>", e.g.
"walker2d__ppo_walker2d_3M_s1__best".

The ONNX graph takes the RAW gymnasium observation and returns the deterministic
action (Gaussian mean), with VecNormalize normalization folded in as constants.

Usage:
    python export_onnx.py                       # latest run of Hopper (final+best)
    python export_onnx.py --env Walker2d-v5     # latest Walker run
    python export_onnx.py --env Walker2d-v5 --all   # every Walker run with a model
    python export_onnx.py --all-envs            # every run of every env
    python export_onnx.py --run runs/hopper/lambda_a10 --variant final
"""

import argparse
import csv
import json
import os
import shutil

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import mujoco
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from rl_common import DEFAULT_ENV_ID, env_from_config, env_label, resolve_latest

DEFAULT_OUT = "web/public"
CURVE_POINTS = 300  # downsample training/eval curves to at most this many points


def parse_args():
    p = argparse.ArgumentParser(description="Export trained policies for the web viewer")
    p.add_argument("--env", type=str, default=DEFAULT_ENV_ID,
                   help="Gymnasium env id (used to find runs/<env>/...)")
    p.add_argument("--run", type=str, default=None,
                   help="a single run folder to export (default: runs/<env>/latest)")
    p.add_argument("--all", action="store_true",
                   help="export every run under runs/<env>/ that has a model")
    p.add_argument("--all-envs", action="store_true",
                   help="export every run of every env found under runs/")
    p.add_argument("--variant", choices=["final", "best", "both"], default="both",
                   help="which saved model(s) to export (default: both if present)")
    p.add_argument("--out", type=str, default=DEFAULT_OUT,
                   help="output root (default: web/public)")
    p.add_argument("--parity-steps", type=int, default=64,
                   help="deterministic steps recorded for the JS parity check")
    return p.parse_args()


def has_model(run_dir):
    return (os.path.exists(os.path.join(run_dir, "final_model.zip"))
            or os.path.exists(os.path.join(run_dir, "best_model.zip")))


def discover_runs(label):
    """All real (non-symlink) run dirs under runs/<label>/ that have a model."""
    root = os.path.join("runs", label)
    if not os.path.isdir(root):
        return []
    runs = []
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        if name == "latest" or os.path.islink(d) or not os.path.isdir(d):
            continue
        if has_model(d):
            runs.append(d)
    return runs


def discover_labels():
    if not os.path.isdir("runs"):
        return []
    return sorted(
        n for n in os.listdir("runs") if os.path.isdir(os.path.join("runs", n))
    )


class OnnxablePolicy(nn.Module):
    """Wraps an SB3 PPO policy: raw obs in, deterministic action out, with
    VecNormalize normalization folded into the graph."""

    def __init__(self, policy, obs_mean, obs_var, epsilon, clip_obs):
        super().__init__()
        self.policy = policy
        self.register_buffer("mean", torch.as_tensor(obs_mean, dtype=torch.float32))
        self.register_buffer("var", torch.as_tensor(obs_var, dtype=torch.float32))
        self.epsilon = float(epsilon)
        self.clip_obs = float(clip_obs)

    def forward(self, obs):
        obs = (obs - self.mean) / torch.sqrt(self.var + self.epsilon)
        obs = torch.clamp(obs, -self.clip_obs, self.clip_obs)
        features = self.policy.extract_features(obs)
        latent_pi = self.policy.mlp_extractor.forward_actor(features)
        return self.policy.action_net(latent_pi)


def _activation_name(mod):
    """Map a torch activation module to a name the JS forward pass understands."""
    if isinstance(mod, nn.Tanh):
        return "tanh"
    if isinstance(mod, nn.ReLU):
        return "relu"
    if isinstance(mod, nn.ELU):
        return "elu"
    if isinstance(mod, nn.LeakyReLU):
        return "leaky_relu"
    if isinstance(mod, nn.SiLU):
        return "silu"
    if isinstance(mod, nn.GELU):
        return "gelu"
    return "linear"


def _extract_mlp(seq, head):
    """Serialize an SB3 extractor branch (a Sequential of Linear/activation
    pairs) plus its output head (a final Linear) into an ordered list of dense
    layers. Weights are [out, in]; the browser normalizes the raw obs (see
    meta.normalization) then runs these layers in order to reproduce every
    neuron's activation for the network visualization."""
    layers = []

    def add_linear(lin, act):
        layers.append({
            "w": lin.weight.detach().cpu().numpy().astype(np.float32).round(5).tolist(),
            "b": lin.bias.detach().cpu().numpy().astype(np.float32).round(5).tolist(),
            "act": act,
            "in": int(lin.in_features),
            "out": int(lin.out_features),
        })

    pending = None
    for mod in list(seq):
        if isinstance(mod, nn.Linear):
            if pending is not None:
                add_linear(pending, "linear")
            pending = mod
        else:
            if pending is not None:
                add_linear(pending, _activation_name(mod))
                pending = None
    if pending is not None:
        add_linear(pending, "linear")

    if isinstance(head, nn.Linear):
        add_linear(head, "linear")
    return layers


def extract_policy_net(policy):
    """Actor MLP: mlp_extractor.policy_net -> action_net (action mean)."""
    return _extract_mlp(
        getattr(policy.mlp_extractor, "policy_net", []),
        getattr(policy, "action_net", None),
    )


def extract_value_net(policy):
    """Critic MLP: mlp_extractor.value_net -> value_net (scalar state value)."""
    return _extract_mlp(
        getattr(policy.mlp_extractor, "value_net", []),
        getattr(policy, "value_net", None),
    )


def _addr_to_joint_names(model):
    """Map each qpos / qvel address to its owning joint's name, so the viewer
    can label observation neurons meaningfully."""
    qpos_names = [f"qpos[{i}]" for i in range(model.nq)]
    qvel_names = [f"qvel[{i}]" for i in range(model.nv)]
    n_qpos = {0: 7, 1: 4, 2: 1, 3: 1}  # free, ball, slide, hinge
    n_qvel = {0: 6, 1: 3, 2: 1, 3: 1}
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint{j}"
        jt = int(model.jnt_type[j])
        qa = int(model.jnt_qposadr[j])
        for k in range(n_qpos.get(jt, 1)):
            if qa + k < model.nq:
                qpos_names[qa + k] = name + (f"[{k}]" if n_qpos.get(jt, 1) > 1 else "")
        da = int(model.jnt_dofadr[j])
        for k in range(n_qvel.get(jt, 1)):
            if da + k < model.nv:
                qvel_names[da + k] = name + (f"[{k}]" if n_qvel.get(jt, 1) > 1 else "")
    return qpos_names, qvel_names


def build_obs_labels(model, components):
    """Human-readable name for every observation index, following the same
    component order the browser reconstructs the obs in."""
    qpos_names, qvel_names = _addr_to_joint_names(model)
    labels = []
    for c in components:
        if c["kind"] == "qpos":
            for i in range(int(c.get("start", 0)), model.nq):
                labels.append(f"{qpos_names[i]} pos")
        elif c["kind"] == "qvel":
            for i in range(int(c.get("start", 0)), model.nv):
                labels.append(f"{qvel_names[i]} vel")
        elif c["kind"] == "cfrc_ext":
            for b in range(int(c.get("start_body", 1)), model.nbody):
                bn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or f"body{b}"
                for comp in ("fx", "fy", "fz", "tx", "ty", "tz"):
                    labels.append(f"{bn} {comp}")
    return labels


def build_action_labels(model):
    """Name for every action index: the actuator name, or (when actuators are
    unnamed, as in many MuJoCo MJCFs) the joint the actuator drives."""
    labels = []
    for a in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
        if not name:
            trn = int(model.actuator_trnid[a, 0])
            joint = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, trn)
                     if trn >= 0 else None)
            name = joint or f"act{a}"
        labels.append(name)
    return labels


def load_vecnormalize_stats(stats_path, env_id):
    if not os.path.exists(stats_path):
        raise SystemExit(
            f"No VecNormalize stats at {stats_path}. The web viewer needs them "
            "to match training-time observations.")
    base = make_vec_env(env_id, n_envs=1)
    vec = VecNormalize.load(stats_path, base)
    mean = vec.obs_rms.mean.astype(np.float32)
    var = vec.obs_rms.var.astype(np.float32)
    eps = float(vec.epsilon)
    clip = float(vec.clip_obs)
    base.close()
    return mean, var, eps, clip


_ENV_SPEC_CACHE = {}


# Observation components the browser can reconstruct from mjData. Envs whose
# observation uses anything outside this set (e.g. Humanoid's cinert/cvel/
# qfrc_actuator) are not yet supported by the web viewer.
SUPPORTED_OBS_KINDS = {"qpos", "qvel", "cfrc_ext"}


def build_obs_components(u, structure):
    """Describe the env's observation as an ordered list of components the
    browser can rebuild from mjData. Handles the Hopper/Walker/Ant/HalfCheetah
    family (qpos slice + qvel [+ contact forces]). qvel clipping is
    auto-calibrated against the real env below."""
    skip = int(structure.get("skipped_qpos", 0))
    components = [
        {"kind": "qpos", "start": skip, "clip": None},
        {"kind": "qvel", "start": 0, "clip": None},  # clip filled in by calibration
    ]
    if int(structure.get("cfrc_ext", 0)) > 0:
        # Ant obs uses cfrc_ext[1:] (skip worldbody) flattened, clipped to range.
        rng = getattr(u, "_contact_force_range", (-1.0, 1.0))
        components.append({"kind": "cfrc_ext", "start_body": 1, "clip": list(rng)})
    return components


def _read_component(u, comp):
    if comp["kind"] == "qpos":
        return u.data.qpos[comp["start"]:].copy()
    if comp["kind"] == "qvel":
        v = u.data.qvel[comp["start"]:].astype(np.float64).copy()
        if comp["clip"] is not None:
            v = np.clip(v, comp["clip"][0], comp["clip"][1])
        return v
    if comp["kind"] == "cfrc_ext":
        cf = np.clip(u.data.cfrc_ext, comp["clip"][0], comp["clip"][1])
        return cf[comp["start_body"]:].flatten()
    raise ValueError(f"unknown obs component {comp['kind']}")


def build_observation_from_env(u, components):
    return np.concatenate([_read_component(u, c) for c in components]).astype(np.float32)


def calibrate_and_validate_obs(env_id, components, needs_rne):
    """Roll a few random steps in the real env and confirm our component-based
    reconstruction matches gymnasium's observation exactly. Auto-picks the qvel
    clip (some envs clip to +/-10, others don't). Raises if unsupported."""
    e = gym.make(env_id)
    u = e.unwrapped
    obs, _ = e.reset(seed=0)

    def max_err_for(qvel_clip):
        for c in components:
            if c["kind"] == "qvel":
                c["clip"] = [-qvel_clip, qvel_clip] if qvel_clip else None
        ee = gym.make(env_id)
        uu = ee.unwrapped
        o, _ = ee.reset(seed=0)
        worst = 0.0
        for _ in range(25):
            a = ee.action_space.sample()
            o, _, term, trunc, _ = ee.step(a)
            rec = build_observation_from_env(uu, components)
            if rec.shape != o.shape:
                ee.close()
                return float("inf")
            worst = max(worst, float(np.max(np.abs(rec - o))))
            if term or trunc:
                o, _ = ee.reset()
        ee.close()
        return worst

    best_clip, best_err = None, float("inf")
    for clip in (None, 10.0):
        err = max_err_for(clip)
        if err < best_err:
            best_err, best_clip = err, clip
    for c in components:
        if c["kind"] == "qvel":
            c["clip"] = [-best_clip, best_clip] if best_clip else None
    e.close()
    if best_err > 1e-4:
        raise SystemExit(
            f"[{env_id}] web-viewer observation reconstruction does not match "
            f"gymnasium (max err {best_err:.3g}). This env likely uses obs "
            "components beyond qpos/qvel/cfrc_ext (e.g. Humanoid), which the "
            "viewer does not support yet.")
    return best_err


def introspect_env(env_id):
    """Pull the exact simulation/observation/termination spec from the gym env,
    generalized across the MuJoCo locomotion family (Hopper/Walker/Ant/...)."""
    if env_id in _ENV_SPEC_CACHE:
        return _ENV_SPEC_CACHE[env_id]
    e = gym.make(env_id)
    u = e.unwrapped
    structure = getattr(u, "observation_structure", {})
    skip = int(structure.get("skipped_qpos",
               int(getattr(u, "_exclude_current_positions_from_observation", 0))))

    unsupported = {k for k, v in structure.items()
                   if k != "skipped_qpos" and v and k not in SUPPORTED_OBS_KINDS}
    if unsupported:
        e.close()
        raise SystemExit(
            f"[{env_id}] observation uses components the web viewer can't rebuild "
            f"yet: {sorted(unsupported)}. Supported: {sorted(SUPPORTED_OBS_KINDS)}.")

    components = build_obs_components(u, structure)
    needs_rne = any(c["kind"] == "cfrc_ext" for c in components)

    angle_range = getattr(u, "_healthy_angle_range", None)
    state_range = getattr(u, "_healthy_state_range", None)
    spec = {
        "env_id": env_id,
        "model_xml": os.path.basename(u.fullpath),
        "obs_dim": int(e.observation_space.shape[0]),
        "act_dim": int(e.action_space.shape[0]),
        "nq": int(u.model.nq),
        "nv": int(u.model.nv),
        "nu": int(u.model.nu),
        "frame_skip": int(u.frame_skip),
        "timestep": float(u.model.opt.timestep),
        "dt": float(u.dt),
        "action_low": e.action_space.low.astype(float).tolist(),
        "action_high": e.action_space.high.astype(float).tolist(),
        "obs_components": components,
        "needs_rne": needs_rne,
        "init_qpos": u.init_qpos.astype(float).tolist(),
        "init_qvel": u.init_qvel.astype(float).tolist(),
        "reset_noise_scale": float(getattr(u, "_reset_noise_scale", 0.0)),
        "healthy": {
            "terminate_when_unhealthy": bool(
                getattr(u, "_terminate_when_unhealthy", True)),
            # The torso height sits at the first non-excluded qpos index.
            "z_index": skip,
            "z_range": list(getattr(u, "_healthy_z_range", (-np.inf, np.inf))),
            "angle_index": skip + 1,
            "angle_range": list(angle_range) if angle_range is not None else None,
            "state_range": list(state_range) if state_range is not None else None,
        },
        "obs_labels": build_obs_labels(u.model, components),
        "action_labels": build_action_labels(u.model),
        "fullpath": u.fullpath,
    }
    e.close()
    # Calibrate qvel clip + verify reconstruction matches gym exactly.
    calibrate_and_validate_obs(env_id, components, needs_rne)
    _ENV_SPEC_CACHE[env_id] = spec
    return spec


def _read_component_data(model, data, comp):
    if comp["kind"] == "qpos":
        return np.asarray(data.qpos[comp["start"]:]).copy()
    if comp["kind"] == "qvel":
        v = np.asarray(data.qvel[comp["start"]:]).astype(np.float64).copy()
        if comp["clip"] is not None:
            v = np.clip(v, comp["clip"][0], comp["clip"][1])
        return v
    if comp["kind"] == "cfrc_ext":
        cf = np.clip(np.asarray(data.cfrc_ext), comp["clip"][0], comp["clip"][1])
        return cf[comp["start_body"]:].flatten()
    raise ValueError(f"unknown obs component {comp['kind']}")


def build_observation(model, data, components):
    return np.concatenate(
        [_read_component_data(model, data, c) for c in components]
    ).astype(np.float32)


def deterministic_parity_trace(onnxable, spec, n_steps):
    model = mujoco.MjModel.from_xml_path(spec["fullpath"])
    data = mujoco.MjData(model)
    data.qpos[:] = np.asarray(spec["init_qpos"])
    data.qvel[:] = np.asarray(spec["init_qvel"])
    mujoco.mj_forward(model, data)

    components = spec["obs_components"]
    frame_skip = spec["frame_skip"]
    needs_rne = spec["needs_rne"]
    trace = []
    with torch.no_grad():
        for _ in range(n_steps):
            obs = build_observation(model, data, components)
            action = onnxable(torch.as_tensor(obs)[None]).numpy()[0].astype(np.float32)
            trace.append({"obs": obs.tolist(), "action": action.tolist()})
            data.ctrl[:] = action
            mujoco.mj_step(model, data, nstep=frame_skip)
            if needs_rne:
                # Matches gym's _step_mujoco_simulation so cfrc_ext is populated.
                mujoco.mj_rnePostConstraint(model, data)
    return trace


def copy_model_assets(spec, models_dir):
    os.makedirs(models_dir, exist_ok=True)
    dst = os.path.join(models_dir, spec["model_xml"])
    if not os.path.exists(dst):
        shutil.copyfile(spec["fullpath"], dst)
    return dst


def downsample(xs, ys, n=CURVE_POINTS):
    """Evenly subsample two parallel lists down to at most n points."""
    m = len(xs)
    if m <= n:
        return list(xs), list(ys)
    idx = np.linspace(0, m - 1, n).round().astype(int)
    return [xs[i] for i in idx], [ys[i] for i in idx]


def read_tfevents_scalars(run_dir):
    """Read scalar series from a run's TensorBoard events file, if present.
    Returns a getter series(tag) -> (steps, values) or None, or {} if there is
    no events file / tensorboard is unavailable. Fallback for older runs that
    predate progress.csv / evaluations.npz."""
    import glob
    files = sorted(glob.glob(os.path.join(run_dir, "events.out.tfevents*")))
    if not files:
        return {}
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except Exception:
        return {}
    try:
        ea = EventAccumulator(files[-1], size_guidance={"scalars": 0})
        ea.Reload()
    except Exception:
        return {}
    tags = set(ea.Tags().get("scalars", []))

    def series(tag):
        if tag not in tags:
            return None
        s = ea.Scalars(tag)
        return [int(p.step) for p in s], [float(p.value) for p in s]

    return {"series": series}


def read_stats(run_dir):
    """Parse config.json, progress.csv, and evaluations.npz into curves +
    a compact summary for the dashboard. Falls back to the TensorBoard events
    file for runs that predate the CSV/npz logs."""
    cfg = {}
    cfg_path = os.path.join(run_dir, "config.json")
    if os.path.exists(cfg_path):
        try:
            cfg = json.load(open(cfg_path))
        except Exception:
            cfg = {}

    # Read all progress.csv rows once (many scalar columns per iteration).
    rows = []
    prog = os.path.join(run_dir, "progress.csv")
    if os.path.exists(prog):
        with open(prog) as f:
            rows = list(csv.DictReader(f))

    tf_state = {"loaded": None}  # lazily-loaded TensorBoard fallback

    def tf_scalars():
        if tf_state["loaded"] is None:
            tf_state["loaded"] = read_tfevents_scalars(run_dir) or {}
        return tf_state["loaded"]

    def series(csv_key, tf_tag=None):
        """(steps, values) for a scalar, from progress.csv if present else from
        the TensorBoard events file. Downsampled."""
        xs, ys = [], []
        for r in rows:
            ts = r.get("time/total_timesteps", "")
            v = r.get(csv_key, "")
            if ts and v not in ("", None):
                try:
                    xs.append(int(float(ts)))
                    ys.append(float(v))
                except ValueError:
                    pass
        if not xs:
            tf = tf_scalars()
            got = tf["series"](tf_tag or csv_key) if tf else None
            if got:
                xs, ys = got
        return downsample(xs, ys)

    train_t, train_rew = series("rollout/ep_rew_mean")
    _, train_len = series("rollout/ep_len_mean")

    # Eval curve from evaluations.npz (mean +/- std across eval episodes).
    eval_t, eval_mean, eval_std, eval_len = [], [], [], []
    npz = os.path.join(run_dir, "eval", "evaluations.npz")
    if os.path.exists(npz):
        d = np.load(npz)
        ts = d["timesteps"].astype(int).tolist()
        res, lens = d["results"], d["ep_lengths"]
        eval_t, eval_mean = downsample(ts, res.mean(axis=1).tolist())
        _, eval_std = downsample(ts, res.std(axis=1).tolist())
        _, eval_len = downsample(ts, lens.mean(axis=1).tolist())
    else:
        tf = tf_scalars()
        em = tf["series"]("eval/mean_reward") if tf else None
        if em:
            eval_t, eval_mean = downsample(*em)
            el = tf["series"]("eval/mean_ep_length")
            if el:
                _, eval_len = downsample(*el)

    # PPO training-health diagnostics. `entropy` is negated entropy_loss so a
    # higher value means more exploration (SB3 logs the loss, which is negative).
    def diag_series(csv_key, negate=False):
        t, v = series(csv_key)
        if negate:
            v = [-x for x in v]
        return {"t": t, "v": v}

    diag = {
        "approx_kl": diag_series("train/approx_kl"),
        "explained_variance": diag_series("train/explained_variance"),
        "entropy": diag_series("train/entropy_loss", negate=True),
        "action_std": diag_series("train/std"),
        "clip_fraction": diag_series("train/clip_fraction"),
        "value_loss": diag_series("train/value_loss"),
    }

    summary = {
        "timesteps": (train_t[-1] if train_t else cfg.get("args", {}).get("timesteps")),
        "train_ep_rew_final": (round(train_rew[-1], 1) if train_rew else None),
        "final_eval_mean": (round(eval_mean[-1], 1) if eval_mean else None),
        "best_eval_mean": (round(max(eval_mean), 1) if eval_mean else None),
        "eval_ep_len_final": (round(eval_len[-1], 0) if eval_len else None),
        "seed": cfg.get("args", {}).get("seed"),
    }

    return {
        "config": {
            "env_id": cfg.get("env_id"),
            "command": cfg.get("command"),
            "created": cfg.get("created"),
            "device": cfg.get("device"),
            "hyperparameters": cfg.get("hyperparameters", {}),
            "args": cfg.get("args", {}),
            "versions": cfg.get("versions", {}),
        },
        "curves": {
            "train": {"t": train_t, "reward": train_rew, "ep_len": train_len},
            "eval": {"t": eval_t, "mean": eval_mean, "std": eval_std, "ep_len": eval_len},
            "diag": diag,
        },
        "summary": summary,
    }


def json_safe(obj, big=1e30):
    if isinstance(obj, dict):
        return {k: json_safe(v, big) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v, big) for v in obj]
    if isinstance(obj, float):
        if obj == float("inf"):
            return big
        if obj == float("-inf"):
            return -big
        if obj != obj:
            return 0.0
    return obj


def export_variant(run_dir, env_id, variant, out, parity_steps):
    """Export a single (run, variant). Returns the run entry dict, or None if
    that variant's model does not exist."""
    model_name = "best_model.zip" if variant == "best" else "final_model.zip"
    model_path = os.path.join(run_dir, model_name)
    if not os.path.exists(model_path):
        return None
    stats_path = os.path.join(run_dir, "vecnormalize.pkl")

    label = env_label(env_id)
    run_name = os.path.basename(os.path.normpath(run_dir))
    entry_id = f"{label}__{run_name}__{variant}"

    policies_dir = os.path.join(out, "policies")
    models_dir = os.path.join(out, "models")
    os.makedirs(policies_dir, exist_ok=True)

    spec = introspect_env(env_id)
    mean, var, eps, clip = load_vecnormalize_stats(stats_path, env_id)

    model = PPO.load(model_path, device="cpu")
    policy = model.policy.to("cpu").eval()
    onnxable = OnnxablePolicy(policy, mean, var, eps, clip).eval()

    onnx_path = os.path.join(policies_dir, f"{entry_id}.onnx")
    dummy = torch.zeros(1, spec["obs_dim"], dtype=torch.float32)
    torch.onnx.export(
        onnxable, dummy, onnx_path,
        input_names=["obs"], output_names=["action"],
        opset_version=17,
        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
        dynamo=False,
    )
    copy_model_assets(spec, models_dir)
    trace = deterministic_parity_trace(onnxable, spec, parity_steps)
    stats = read_stats(run_dir)

    meta = {k: v for k, v in spec.items() if k != "fullpath"}
    meta["normalization"] = {
        "mean": mean.astype(float).tolist(),
        "var": var.astype(float).tolist(),
        "epsilon": eps,
        "clip_obs": clip,
    }
    meta["policy_onnx"] = f"{entry_id}.onnx"
    meta["policy_net"] = {
        "input_dim": spec["obs_dim"],
        "layers": extract_policy_net(policy),
    }
    meta["value_net"] = {
        "input_dim": spec["obs_dim"],
        "layers": extract_value_net(policy),
    }
    meta["run"] = {
        "id": entry_id,
        "label": label,
        "run_name": run_name,
        "variant": variant,
        "run_dir": os.path.abspath(run_dir),
    }
    meta["summary"] = stats["summary"]
    meta["parity_trace"] = trace

    # Compact (no indent): this file embeds the parity trace and both networks'
    # weight matrices, so pretty-printing would bloat it ~3x for no benefit.
    with open(os.path.join(policies_dir, f"{entry_id}.json"), "w") as f:
        json.dump(json_safe(meta), f, separators=(",", ":"), allow_nan=False)
    with open(os.path.join(policies_dir, f"{entry_id}.stats.json"), "w") as f:
        json.dump(json_safe(stats), f, indent=2, allow_nan=False)

    print(f"  exported {entry_id}  "
          f"(eval best {stats['summary'].get('best_eval_mean')}, "
          f"final {stats['summary'].get('final_eval_mean')})")
    return {
        "id": entry_id,
        "run_name": run_name,
        "variant": variant,
        "onnx": f"policies/{entry_id}.onnx",
        "meta": f"policies/{entry_id}.json",
        "stats": f"policies/{entry_id}.stats.json",
        "model_xml": f"models/{meta['model_xml']}",
        "summary": stats["summary"],
    }


def export_run(run_dir, env_id, variant, out, parity_steps):
    variants = ["final", "best"] if variant == "both" else [variant]
    entries = []
    for v in variants:
        entry = export_variant(run_dir, env_id, v, out, parity_steps)
        if entry:
            entries.append(entry)
    return entries


def write_index(policies_dir):
    """Rebuild policies/index.json by scanning exported meta files, grouped by
    environment. This keeps every previously exported run listed."""
    envs = {}
    for name in sorted(os.listdir(policies_dir)):
        if not name.endswith(".json") or name.endswith(".stats.json") or name == "index.json":
            continue
        try:
            meta = json.load(open(os.path.join(policies_dir, name)))
        except Exception:
            continue
        run = meta.get("run")
        if not run:
            continue
        env_id = meta.get("env_id")
        entry = {
            "id": run["id"],
            "run_name": run["run_name"],
            "variant": run["variant"],
            "onnx": f"policies/{run['id']}.onnx",
            "meta": f"policies/{name}",
            "stats": f"policies/{run['id']}.stats.json",
            "model_xml": f"models/{meta.get('model_xml')}",
            "summary": meta.get("summary", {}),
        }
        g = envs.setdefault(env_id, {"env_id": env_id, "label": run["label"], "runs": []})
        g["runs"].append(entry)

    for g in envs.values():
        g["runs"].sort(key=lambda e: (e["run_name"], e["variant"]))
    index = {"envs": [envs[k] for k in sorted(envs)]}
    with open(os.path.join(policies_dir, "index.json"), "w") as f:
        json.dump(index, f, indent=2)
    total = sum(len(g["runs"]) for g in index["envs"])
    print(f"\nindex.json: {total} run(s) across {len(index['envs'])} env(s)")


def main():
    args = parse_args()
    policies_dir = os.path.join(args.out, "policies")
    os.makedirs(policies_dir, exist_ok=True)

    # Build the list of (run_dir, env_id) to export.
    targets = []
    if args.all_envs:
        for label in discover_labels():
            for run_dir in discover_runs(label):
                env_id = env_from_config(run_dir) or f"{label.capitalize()}-v5"
                targets.append((run_dir, env_id))
    elif args.all:
        label = env_label(args.env)
        for run_dir in discover_runs(label):
            targets.append((run_dir, env_from_config(run_dir) or args.env))
    elif args.run:
        targets.append((args.run, env_from_config(args.run) or args.env))
    else:
        run_dir = resolve_latest(env_label(args.env))
        targets.append((run_dir, env_from_config(run_dir) or args.env))

    if not targets:
        raise SystemExit("No runs with models found. Train one first: python train.py")

    exported = 0
    for run_dir, env_id in targets:
        if not has_model(run_dir):
            print(f"[skip] {run_dir} (no model)")
            continue
        print(f"{env_id}  <-  {run_dir}")
        entries = export_run(run_dir, env_id, args.variant, args.out, args.parity_steps)
        exported += len(entries)

    write_index(policies_dir)
    print(f"Exported {exported} policy file(s).")
    print("Start the viewer with:  cd web && npm run dev")


if __name__ == "__main__":
    main()
