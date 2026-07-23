"""Parametric monster morphologies: a MonsterSpec (torso + limbs) rendered to
a MuJoCo model via mujoco.MjSpec.

This module is deliberately dependency-light (mujoco + stdlib only — no
jax/brax) so it can be imported by both the training side (MonsterMjx in
locomotion.py) and the export side (export_onnx.py, which runs in the
torch venv without jax).

A monster is a free-joint torso with limbs attached. Each limb is a chain of
capsule segments: a passive "mount" capsule from the torso to the first joint,
then actuated segments. The first actuated segment swings about z (like Ant's
hip); the rest hinge about the horizontal axis perpendicular to the limb's
heading (like Ant's ankle), so they lift/lower the limb within its vertical
plane. Rest pose is the spec's geometry (all joints at 0, inside their ranges),
so a freshly spawned monster stands at its geometric height and settles.

Generated MJCF mirrors ant.xml's conventions (RK4 @ 0.01 s, joint armature/
damping 1, geom density 5, limb geoms collide only with the floor, motors with
gear on every joint, a trackcom camera) so everything downstream — MJX
training, the ONNX export, and the browser viewer — treats a monster exactly
like it treats Ant.

Naming is deterministic: bodies/joints/geoms are `<limb>_seg<k>[_joint|_geom]`
and the mount capsule is `<limb>_mount_geom`, so feet and segments stay
addressable for future contact-based tasks (see foot_geom_names).

CLI (works in either venv; the interactive viewer needs `mjpython` on macOS):

    python -m mjx_envs.monsters --list
    python -m mjx_envs.monsters --preview quad
    python -m mjx_envs.monsters --sample --seed 3 --preview
    python -m mjx_envs.monsters --write-assets            # all presets
    python -m mjx_envs.monsters --write-assets spider8
"""

import argparse
import dataclasses
import json
import math
import os
import re
from dataclasses import dataclass, field, replace
from typing import Optional, Tuple

import mujoco

# Shared with MonsterMjx (locomotion.py) and introspect_monster (export_onnx.py)
# so the training env, the exported spec, and the browser all agree.
FRAME_SKIP = 5           # Ant's control period (dt = 0.05 s)
RESET_NOISE_SCALE = 0.1  # Ant's reset noise
TIMESTEP = 0.01
CLEARANCE = 0.05         # spawn gap between the lowest resting point and the floor

ENV_PREFIX = "Monster-"
ENV_VERSION = "v0"

# Repo-root-relative home for generated monsters (<name>.xml + <name>.spec.json).
ASSETS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "assets", "monsters")

SPEC_FORMAT = "monster-spec-v1"


# ---------------------------------------------------------------------------
# Specs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LimbSpec:
    """One limb: a chain of capsule segments hanging off the torso.

    Geometry is polar: `yaw_deg` is the limb's heading in the torso's xy plane,
    and each segment tilts `pitches_deg[k]` below horizontal (negative = up, so
    tails can curl upward). Joints rest at 0 = the drawn geometry, with ranges
    relative to that rest pose."""

    yaw_deg: float                       # heading in the torso's xy plane
    lengths: Tuple[float, ...]           # actuated segment lengths, root -> tip
    pitches_deg: Tuple[float, ...]       # per-segment tilt below horizontal
    mount_length: float = 0.2            # passive capsule torso -> first joint
    mount_pitch_deg: float = 0.0
    radius: float = 0.08                 # capsule radius for every segment
    swing_range_deg: Tuple[float, float] = (-35.0, 35.0)  # first joint (about z)
    lift_range_deg: Tuple[float, float] = (-45.0, 45.0)   # later joints (in-plane)
    gear: float = 150.0                  # motor gear for every joint
    name: Optional[str] = None           # defaults to limb<i> at build time

    def __post_init__(self):
        if len(self.lengths) != len(self.pitches_deg):
            raise ValueError(
                f"limb {self.name!r}: lengths ({len(self.lengths)}) and "
                f"pitches_deg ({len(self.pitches_deg)}) must have equal length")
        if not self.lengths:
            raise ValueError(f"limb {self.name!r}: needs at least one segment")


@dataclass(frozen=True)
class MonsterSpec:
    """A whole monster: torso shape + limbs. `torso_length` > 0 turns the
    torso into a capsule along x (a sphere otherwise). `healthy_z_range`
    overrides the geometry-derived default (see default_healthy_z_range)."""

    name: str
    limbs: Tuple[LimbSpec, ...]
    torso_radius: float = 0.25
    torso_length: float = 0.0
    rgba: Tuple[float, float, float, float] = (0.8, 0.6, 0.4, 1.0)
    healthy_z_range: Optional[Tuple[float, float]] = None

    def __post_init__(self):
        if not self.limbs:
            raise ValueError(f"monster {self.name!r}: needs at least one limb")
        names = [l.name for l in self.limbs if l.name]
        if len(names) != len(set(names)):
            raise ValueError(f"monster {self.name!r}: duplicate limb names")


# -- symmetry helpers --------------------------------------------------------

def radial(n: int, limb: LimbSpec, yaw0: float = 0.0) -> Tuple[LimbSpec, ...]:
    """n copies of `limb` spread evenly around the torso, starting at yaw0."""
    base = limb.name or "leg"
    return tuple(
        replace(limb, yaw_deg=yaw0 + i * 360.0 / n, name=f"{base}{i + 1}")
        for i in range(n))


def bilateral(limb: LimbSpec) -> Tuple[LimbSpec, LimbSpec]:
    """A left/right mirrored pair (yaw negated across the xz plane)."""
    base = limb.name or "leg"
    return (replace(limb, name=f"{base}_l"),
            replace(limb, yaw_deg=-limb.yaw_deg, name=f"{base}_r"))


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _set_damping(joint, value: float) -> None:
    """MjsJoint.damping is a scalar up to mujoco 3.9 and a per-DOF [3] vector
    from 3.10; uniform damping means the same physics either way."""
    try:
        joint.damping = [value, value, value]
    except TypeError:
        joint.damping = value


def _dir(yaw_deg: float, pitch_deg: float) -> Tuple[float, float, float]:
    """Unit vector at heading `yaw_deg`, tilted `pitch_deg` below horizontal."""
    yaw, pitch = math.radians(yaw_deg), math.radians(pitch_deg)
    return (math.cos(yaw) * math.cos(pitch),
            math.sin(yaw) * math.cos(pitch),
            -math.sin(pitch))


def _limb_segments(limb: LimbSpec):
    """[(length, pitch_deg)] including the mount as element 0."""
    return ([(limb.mount_length, limb.mount_pitch_deg)]
            + list(zip(limb.lengths, limb.pitches_deg)))


def _limb_drop(limb: LimbSpec) -> float:
    """Deepest point (below torso center) of the limb's resting geometry."""
    z, lowest = 0.0, 0.0
    for length, pitch in _limb_segments(limb):
        z -= length * math.sin(math.radians(pitch))
        lowest = min(lowest, z)
    return -lowest + limb.radius


def torso_height(m: MonsterSpec) -> float:
    """Spawn height: torso center such that the lowest resting limb point (or
    the torso itself) clears the floor by CLEARANCE."""
    drop = max(_limb_drop(l) for l in m.limbs)
    return max(m.torso_radius, drop) + CLEARANCE


def default_healthy_z_range(m: MonsterSpec) -> Tuple[float, float]:
    """Geometry-derived health band for the torso height, in the spirit of
    Ant's (0.2, 1.0) vs its 0.75 spawn: collapsed => unhealthy, and a generous
    ceiling (the anti-flip upright check does the real work)."""
    z0 = torso_height(m)
    return (max(0.05, round(0.3 * z0, 3)), round(3.0 * z0, 3))


def foot_geom_names(m: MonsterSpec) -> Tuple[str, ...]:
    """The tip (last-segment) geom of every limb — the 'feet', for future
    contact-based tasks (mirrors AntFood2Leg's ankle-geom list)."""
    names = []
    for i, limb in enumerate(m.limbs):
        base = limb.name or f"limb{i + 1}"
        names.append(f"{base}_seg{len(limb.lengths)}_geom")
    return tuple(names)


# ---------------------------------------------------------------------------
# Spec -> MuJoCo model
# ---------------------------------------------------------------------------

def spec_to_mjspec(m: MonsterSpec) -> mujoco.MjSpec:
    """Build the MjSpec: ant.xml's world (floor/light/skybox/defaults) around
    a generated torso + limb tree with one motor per joint."""
    s = mujoco.MjSpec()
    s.modelname = m.name
    s.compiler.degree = True
    s.compiler.inertiafromgeom = mujoco.mjtInertiaFromGeom.mjINERTIAFROMGEOM_TRUE
    s.option.integrator = mujoco.mjtIntegrator.mjINT_RK4
    s.option.timestep = TIMESTEP

    d = s.default
    d.joint.armature = 1.0
    _set_damping(d.joint, 1.0)
    d.joint.limited = mujoco.mjtLimited.mjLIMITED_TRUE
    d.geom.conaffinity = 0      # limbs collide with the floor only, like Ant
    d.geom.condim = 3
    d.geom.density = 5.0
    d.geom.friction = [1.0, 0.5, 0.5]
    d.geom.margin = 0.01
    d.geom.rgba = list(m.rgba)

    # World dressing, straight from ant.xml.
    sky = s.add_texture(name="sky")
    sky.type = mujoco.mjtTexture.mjTEXTURE_SKYBOX
    sky.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
    sky.rgb1, sky.rgb2 = [1, 1, 1], [0, 0, 0]
    sky.width = sky.height = 100
    plane_tex = s.add_texture(name="texplane")
    plane_tex.type = mujoco.mjtTexture.mjTEXTURE_2D
    plane_tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
    plane_tex.rgb1, plane_tex.rgb2 = [0, 0, 0], [0.8, 0.8, 0.8]
    plane_tex.width = plane_tex.height = 100
    mat = s.add_material(name="MatPlane")
    mat.reflectance, mat.shininess, mat.specular = 0.5, 1.0, 1.0
    mat.texrepeat = [60, 60]
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "texplane"

    light = s.worldbody.add_light(pos=[0, 0, 1.3], dir=[0, 0, -1])
    light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    light.diffuse = [1, 1, 1]
    light.specular = [0.1, 0.1, 0.1]
    light.cutoff = 100.0

    floor = s.worldbody.add_geom(name="floor")
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.conaffinity = 1
    floor.material = "MatPlane"
    floor.rgba = [0.8, 0.9, 0.8, 1]
    floor.size = [40, 40, 40]

    torso = s.worldbody.add_body(name="torso", pos=[0, 0, torso_height(m)])
    cam = torso.add_camera(name="track", pos=[0, -3, 0.3])
    cam.mode = mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM
    cam.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES
    cam.alt.xyaxes = [1, 0, 0, 0, 0, 1]
    # The root free joint must escape the default joint class (armature/
    # damping 1, limited), exactly as ant.xml's root does.
    root = torso.add_freejoint(name="root")
    root.armature = 0.0
    _set_damping(root, 0.0)
    root.limited = mujoco.mjtLimited.mjLIMITED_FALSE
    root.margin = 0.01
    tg = torso.add_geom(name="torso_geom")
    if m.torso_length > 0:
        tg.type = mujoco.mjtGeom.mjGEOM_CAPSULE
        half = m.torso_length / 2.0
        tg.fromto = [-half, 0, 0, half, 0, 0]
        tg.size = [m.torso_radius, 0, 0]
    else:
        tg.type = mujoco.mjtGeom.mjGEOM_SPHERE
        tg.size = [m.torso_radius, 0, 0]

    for i, limb in enumerate(m.limbs):
        _add_limb(s, torso, limb, limb.name or f"limb{i + 1}")
    return s


def _add_limb(s: mujoco.MjSpec, torso, limb: LimbSpec, name: str) -> None:
    root = torso.add_body(name=name, pos=[0, 0, 0])
    mount_tip = [limb.mount_length * c
                 for c in _dir(limb.yaw_deg, limb.mount_pitch_deg)]
    mount = root.add_geom(name=f"{name}_mount_geom")
    mount.type = mujoco.mjtGeom.mjGEOM_CAPSULE
    mount.fromto = [0, 0, 0] + mount_tip
    mount.size = [limb.radius, 0, 0]

    # "Lift" hinge axis: horizontal, perpendicular to the limb's heading, so
    # positive rotation raises the tip (Ant's ankle axes follow this formula).
    yaw = math.radians(limb.yaw_deg)
    lift_axis = [-math.sin(yaw), math.cos(yaw), 0]

    prev_body, prev_vec = root, mount_tip
    for k, (length, pitch) in enumerate(zip(limb.lengths, limb.pitches_deg)):
        body = prev_body.add_body(name=f"{name}_seg{k + 1}", pos=prev_vec)
        joint = body.add_joint(name=f"{name}_seg{k + 1}_joint")
        joint.type = mujoco.mjtJoint.mjJNT_HINGE
        if k == 0:
            joint.axis = [0, 0, 1]
            joint.range = list(limb.swing_range_deg)
        else:
            joint.axis = lift_axis
            joint.range = list(limb.lift_range_deg)
        seg_vec = [length * c for c in _dir(limb.yaw_deg, pitch)]
        geom = body.add_geom(name=f"{name}_seg{k + 1}_geom")
        geom.type = mujoco.mjtGeom.mjGEOM_CAPSULE
        geom.fromto = [0, 0, 0] + seg_vec
        geom.size = [limb.radius, 0, 0]

        motor = s.add_actuator(name=f"{name}_seg{k + 1}_motor")
        motor.target = joint.name
        motor.trntype = mujoco.mjtTrn.mjTRN_JOINT
        motor.gear = [limb.gear, 0, 0, 0, 0, 0]
        motor.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
        motor.ctrlrange = [-1.0, 1.0]

        prev_body, prev_vec = body, seg_vec


def spec_to_model(m: MonsterSpec) -> mujoco.MjModel:
    return spec_to_mjspec(m).compile()


def spec_to_xml(m: MonsterSpec) -> str:
    s = spec_to_mjspec(m)
    s.compile()  # to_xml requires a compiled spec
    xml = s.to_xml()
    # mujoco >= 3.10 serializes joint damping per-DOF ('1 1 1'); collapse
    # uniform vectors back to the scalar form so the XML also loads in older
    # parsers — in particular the viewer's WASM build (MuJoCo 3.3.x).
    xml = re.sub(r'damping="([\d.eE+-]+) \1 \1"', r'damping="\1"', xml)
    return xml


# ---------------------------------------------------------------------------
# JSON (de)serialization
# ---------------------------------------------------------------------------

def spec_to_dict(m: MonsterSpec) -> dict:
    return {"format": SPEC_FORMAT, **dataclasses.asdict(m)}


def _limb_from_dict(d: dict) -> LimbSpec:
    kwargs = {f.name: d[f.name] for f in dataclasses.fields(LimbSpec) if f.name in d}
    for key in ("lengths", "pitches_deg", "swing_range_deg", "lift_range_deg"):
        if key in kwargs:
            kwargs[key] = tuple(kwargs[key])
    return LimbSpec(**kwargs)


def spec_from_dict(d: dict) -> MonsterSpec:
    fmt = d.get("format", SPEC_FORMAT)
    if fmt != SPEC_FORMAT:
        raise ValueError(f"unsupported monster spec format: {fmt!r}")
    kwargs = {f.name: d[f.name] for f in dataclasses.fields(MonsterSpec) if f.name in d}
    kwargs["limbs"] = tuple(_limb_from_dict(l) for l in kwargs["limbs"])
    kwargs["rgba"] = tuple(kwargs.get("rgba", (0.8, 0.6, 0.4, 1.0)))
    if kwargs.get("healthy_z_range") is not None:
        kwargs["healthy_z_range"] = tuple(kwargs["healthy_z_range"])
    return MonsterSpec(**kwargs)


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

def _make_presets() -> dict:
    # quad: an Ant relative — 4 radial legs, hip swing + one lift segment.
    quad = MonsterSpec(
        name="quad",
        limbs=radial(4, LimbSpec(
            yaw_deg=0, mount_length=0.28,
            lengths=(0.28, 0.57), pitches_deg=(0.0, 25.0),
            radius=0.08, gear=150.0), yaw0=45.0))

    # hexapod: 6 shorter legs around a capsule body — statically stable.
    hexapod = MonsterSpec(
        name="hexapod",
        torso_radius=0.18, torso_length=0.5,
        rgba=(0.45, 0.7, 0.4, 1.0),
        limbs=radial(6, LimbSpec(
            yaw_deg=0, mount_length=0.18,
            lengths=(0.22, 0.38), pitches_deg=(10.0, 50.0),
            radius=0.06, gear=120.0), yaw0=30.0))

    # biped_tail: two wide-stance legs pitched steeply down, plus a two-segment
    # counterbalancing tail curling up and back.
    leg = LimbSpec(
        yaw_deg=90.0, mount_length=0.12,
        lengths=(0.35, 0.35), pitches_deg=(55.0, 80.0),
        radius=0.07, gear=180.0,
        swing_range_deg=(-25.0, 25.0), lift_range_deg=(-50.0, 50.0), name="leg")
    tail = LimbSpec(
        yaw_deg=180.0, mount_length=0.15,
        lengths=(0.35, 0.3), pitches_deg=(-25.0, -10.0),
        radius=0.05, gear=80.0,
        swing_range_deg=(-40.0, 40.0), lift_range_deg=(-40.0, 40.0), name="tail")
    biped_tail = MonsterSpec(
        name="biped_tail",
        torso_radius=0.2,
        rgba=(0.75, 0.45, 0.65, 1.0),
        limbs=bilateral(leg) + (tail,))

    # spider8: 8 thin legs that arch up from the body then reach down.
    spider8 = MonsterSpec(
        name="spider8",
        torso_radius=0.22,
        rgba=(0.3, 0.3, 0.35, 1.0),
        limbs=radial(8, LimbSpec(
            yaw_deg=0, mount_length=0.2,
            lengths=(0.3, 0.45), pitches_deg=(-25.0, 60.0),
            radius=0.05, gear=100.0), yaw0=22.5))

    return {m.name: m for m in (quad, hexapod, biped_tail, spider8)}


MONSTERS = _make_presets()


# ---------------------------------------------------------------------------
# Random morphologies
# ---------------------------------------------------------------------------

def sample_spec(seed: int, name: Optional[str] = None) -> MonsterSpec:
    """A random (seeded) morphology: radial or bilateral limb layout, 1-3
    actuated segments per limb, random sizes/gears. Deterministic per seed."""
    import random
    rng = random.Random(seed)

    n_segments = rng.randint(1, 3)
    lengths = tuple(round(rng.uniform(0.18, 0.5), 3) for _ in range(n_segments))
    pitches = tuple(round(rng.uniform(-25.0, 60.0), 1) for _ in range(n_segments))
    limb = LimbSpec(
        yaw_deg=0,
        mount_length=round(rng.uniform(0.1, 0.3), 3),
        mount_pitch_deg=round(rng.uniform(-15.0, 15.0), 1),
        lengths=lengths, pitches_deg=pitches,
        radius=round(rng.uniform(0.05, 0.09), 3),
        gear=round(rng.uniform(80.0, 200.0), 0))

    if rng.random() < 0.5:
        n = rng.randint(3, 8)
        limbs = radial(n, limb, yaw0=rng.uniform(0.0, 360.0 / n))
    else:
        pairs = rng.randint(1, 3)
        limbs = ()
        for p in range(pairs):
            yaw = rng.uniform(30.0, 150.0)
            limbs += bilateral(replace(limb, yaw_deg=yaw, name=f"leg{p + 1}"))
        if rng.random() < 0.5:
            limbs += (replace(
                limb, yaw_deg=180.0, name="tail",
                lengths=tuple(round(x * 0.8, 3) for x in lengths),
                pitches_deg=tuple(-abs(p) * 0.5 for p in pitches),
                gear=round(limb.gear * 0.6, 0)),)

    rgba = (round(rng.uniform(0.2, 0.9), 2), round(rng.uniform(0.2, 0.9), 2),
            round(rng.uniform(0.2, 0.9), 2), 1.0)
    return MonsterSpec(
        name=name or f"rand{seed}",
        torso_radius=round(rng.uniform(0.15, 0.3), 3),
        torso_length=(round(rng.uniform(0.3, 0.6), 3)
                      if rng.random() < 0.4 else 0.0),
        rgba=rgba, limbs=limbs)


# ---------------------------------------------------------------------------
# Env-id mapping and spec loading (shared by locomotion.py and export_onnx.py)
# ---------------------------------------------------------------------------

def env_id_for(name: str) -> str:
    return f"{ENV_PREFIX}{name}-{ENV_VERSION}"


def name_from_env_id(env_id: str) -> Optional[str]:
    """'Monster-quad-v0' -> 'quad'; None when env_id is not a monster id."""
    if not env_id.startswith(ENV_PREFIX):
        return None
    stem = env_id[len(ENV_PREFIX):].rsplit("-v", 1)[0]
    return stem or None


def load_spec(name: str, assets_dir: str = ASSETS_DIR) -> MonsterSpec:
    """A preset, or a previously written <assets_dir>/<name>.spec.json (how
    sampled monsters become trainable by env id)."""
    if name in MONSTERS:
        return MONSTERS[name]
    path = os.path.join(assets_dir, f"{name}.spec.json")
    if os.path.exists(path):
        with open(path) as f:
            return spec_from_dict(json.load(f))
    raise ValueError(
        f"unknown monster {name!r}: not a preset {sorted(MONSTERS)} and no "
        f"spec file at {path} (write one with --write-assets or --sample)")


def write_assets(m: MonsterSpec, assets_dir: str = ASSETS_DIR):
    """Write <name>.xml + <name>.spec.json; returns (xml_path, json_path)."""
    os.makedirs(assets_dir, exist_ok=True)
    xml_path = os.path.join(assets_dir, f"{m.name}.xml")
    json_path = os.path.join(assets_dir, f"{m.name}.spec.json")
    with open(xml_path, "w") as f:
        f.write(spec_to_xml(m))
    with open(json_path, "w") as f:
        json.dump(spec_to_dict(m), f, indent=2)
    return xml_path, json_path


def ensure_xml(m: MonsterSpec, assets_dir: str = ASSETS_DIR) -> str:
    """The monster's XML on disk (regenerated every call so the file always
    matches the current spec/generator). Returns the xml path."""
    return write_assets(m, assets_dir)[0]


# ---------------------------------------------------------------------------
# Sanity check + CLI
# ---------------------------------------------------------------------------

def sanity_check(m: MonsterSpec, steps: int = 200, seed: int = 0) -> dict:
    """Compile and run `steps` random-action control periods in C-MuJoCo:
    everything must stay finite and the torso must not fall through the floor
    or launch. Cheap gate before burning GPU time on a broken body."""
    import numpy as np
    model = spec_to_model(m)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    rng = np.random.default_rng(seed)
    lo, hi = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
    min_z, max_z = float(data.qpos[2]), float(data.qpos[2])
    for _ in range(steps):
        data.ctrl[:] = rng.uniform(lo, hi)
        mujoco.mj_step(model, data, nstep=FRAME_SKIP)
        if not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()):
            raise RuntimeError(f"{m.name}: non-finite state after random actions")
        z = float(data.qpos[2])
        min_z, max_z = min(min_z, z), max(max_z, z)
    if max_z > 10.0 or min_z < -1.0:
        raise RuntimeError(
            f"{m.name}: torso z left [{min_z:.2f}, {max_z:.2f}] — the body is "
            "unstable (check limb geometry / gear)")
    obs_dim = (model.nq - 2) + model.nv + 6 * (model.nbody - 1)
    return {
        "nq": model.nq, "nv": model.nv, "nu": model.nu,
        "obs_dim": obs_dim,
        "spawn_z": round(torso_height(m), 3),
        "torso_z_range_seen": (round(min_z, 3), round(max_z, 3)),
        "healthy_z_range": m.healthy_z_range or default_healthy_z_range(m),
    }


def _launch_viewer(m: MonsterSpec) -> None:
    model = spec_to_model(m)
    data = mujoco.MjData(model)
    try:
        import mujoco.viewer
        mujoco.viewer.launch(model, data)
    except RuntimeError as e:
        raise SystemExit(
            f"viewer failed to launch ({e}).\nOn macOS the interactive viewer "
            "needs the mjpython launcher:\n"
            f"  .venv/bin/mjpython -m mjx_envs.monsters --preview {m.name}")


def main(argv=None):
    p = argparse.ArgumentParser(description="Generate/preview parametric monsters")
    p.add_argument("--list", action="store_true", help="list preset monsters")
    p.add_argument("--preview", type=str, default=None, metavar="NAME",
                   help="sanity-check NAME (preset or assets spec) and open "
                        "the interactive viewer")
    p.add_argument("--sample", action="store_true",
                   help="generate a random monster (rand<seed>) and write its "
                        "assets; combine with --preview-sample to view it")
    p.add_argument("--seed", type=int, default=0, help="seed for --sample")
    p.add_argument("--preview-sample", action="store_true",
                   help="with --sample: also open the viewer")
    p.add_argument("--write-assets", nargs="*", default=None, metavar="NAME",
                   help="write <name>.xml + <name>.spec.json to the assets dir "
                        "(no names = all presets)")
    p.add_argument("--assets-dir", type=str, default=ASSETS_DIR)
    p.add_argument("--steps", type=int, default=200,
                   help="random-action control periods for the sanity check")
    p.add_argument("--no-viewer", action="store_true",
                   help="with --preview: sanity-check only")
    args = p.parse_args(argv)

    did_something = False

    if args.list:
        did_something = True
        for name, m in sorted(MONSTERS.items()):
            model = spec_to_model(m)
            print(f"  {name:12s} limbs={len(m.limbs)} joints={model.nu} "
                  f"env id: {env_id_for(name)}")

    if args.write_assets is not None:
        did_something = True
        names = args.write_assets or sorted(MONSTERS)
        for name in names:
            m = load_spec(name, args.assets_dir)
            xml_path, json_path = write_assets(m, args.assets_dir)
            print(f"  wrote {xml_path} and {json_path}")

    if args.sample:
        did_something = True
        m = sample_spec(args.seed)
        report = sanity_check(m, steps=args.steps, seed=args.seed)
        xml_path, json_path = write_assets(m, args.assets_dir)
        print(f"sampled {m.name}: {report}")
        print(f"  wrote {xml_path} and {json_path}")
        print(f"  train it with: python train_mjx.py --env {env_id_for(m.name)}")
        if args.preview_sample:
            _launch_viewer(m)

    if args.preview:
        did_something = True
        m = load_spec(args.preview, args.assets_dir)
        report = sanity_check(m, steps=args.steps)
        print(f"{m.name}: {report}")
        print(f"  env id: {env_id_for(m.name)}")
        if not args.no_viewer:
            _launch_viewer(m)

    if not did_something:
        p.print_help()


if __name__ == "__main__":
    main()
