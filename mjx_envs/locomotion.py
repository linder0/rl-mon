"""Faithful MJX ports of the gymnasium `-v5` MuJoCo locomotion envs.

These Brax `PipelineEnv`s load the *same* MJCF models gymnasium ships and
replicate its observation, reward, termination, and reset logic exactly, so a
policy trained here transfers to C-MuJoCo (enjoy.py, the web viewer) with the
same observation contract the export pipeline already understands.

Ground truth: gymnasium/envs/mujoco/{hopper,walker2d,ant}_v5.py. Every constant
below mirrors those files' defaults. The only intentional difference is the
physics engine (MJX vs C MuJoCo) — obs/reward *construction* is checked to
match exactly by mjx_envs/validate.py; *dynamics* parity is empirical (see the
transfer check in export_onnx.py).
"""

import os

import gymnasium.envs.mujoco as _gym_mujoco
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf as brax_mjcf

from mjx_envs import monsters

ASSETS_DIR = os.path.join(os.path.dirname(_gym_mujoco.__file__), "assets")


class GymLocomotionEnv(PipelineEnv):
    """Base class for the gym-v5 ports: qpos-slice obs, forward reward,
    healthy termination, uniform reset noise. Subclasses fill in constants."""

    # Mirrors of the gymnasium -v5 constructor defaults.
    xml_file: str
    frame_skip: int
    forward_reward_weight: float = 1.0
    ctrl_cost_weight: float = 1e-3
    healthy_reward: float = 1.0
    reset_noise_scale: float = 5e-3
    skipped_qpos: int = 1  # exclude_current_positions_from_observation

    def __init__(self):
        path = os.path.join(ASSETS_DIR, self.xml_file)
        self._init_from_model(mujoco.MjModel.from_xml_path(path))

    def _init_from_model(self, mj_model):
        """Finish construction from an already-built MjModel (the gym asset
        for the ports; an in-memory generated model for MonsterMjx)."""
        sys = brax_mjcf.load_model(mj_model)
        super().__init__(sys=sys, backend="mjx", n_frames=self.frame_skip)
        self._init_qpos = jnp.asarray(mj_model.qpos0)
        self._init_qvel = jnp.zeros(mj_model.nv)
        self._ctrl_low = jnp.asarray(mj_model.actuator_ctrlrange[:, 0])
        self._ctrl_high = jnp.asarray(mj_model.actuator_ctrlrange[:, 1])
        self.mj_model = mj_model

    # -- pieces shared with validate.py (duck-typed on .qpos/.qvel/...) ------

    def obs_from_data(self, data) -> jnp.ndarray:
        position = jnp.asarray(data.qpos)[self.skipped_qpos:]
        velocity = jnp.clip(jnp.asarray(data.qvel), -10.0, 10.0)
        return jnp.concatenate((position, velocity))

    def is_healthy(self, data) -> jnp.ndarray:
        raise NotImplementedError

    def forward_pos(self, data) -> jnp.ndarray:
        """Coordinate whose displacement defines the forward reward."""
        return jnp.asarray(data.qpos)[0]

    def reward_terms(self, x_velocity, healthy, action, data):
        forward_reward = self.forward_reward_weight * x_velocity
        healthy_reward = healthy * self.healthy_reward
        ctrl_cost = self.ctrl_cost_weight * jnp.sum(jnp.square(action))
        reward = forward_reward + healthy_reward - ctrl_cost
        metrics = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_survive": healthy_reward,
        }
        return reward, metrics

    def task_spec(self):
        """Distinctive task parameters for this env, recorded per-run by
        train_mjx.py so runs stay self-describing and reproducible even as the
        reward/termination/task constants are tuned across variants."""
        return {
            "env_class": type(self).__name__,
            "xml_file": self.xml_file,
            "frame_skip": self.frame_skip,
            "reset_noise_scale": self.reset_noise_scale,
            "skipped_qpos": self.skipped_qpos,
            "reward": {
                "forward_reward_weight": self.forward_reward_weight,
                "ctrl_cost_weight": self.ctrl_cost_weight,
                "healthy_reward": self.healthy_reward,
            },
        }

    # -- reset / step ---------------------------------------------------------

    def _reset_noise(self, rng):
        """Hopper/Walker2d: uniform noise on both qpos and qvel."""
        s = self.reset_noise_scale
        rng_pos, rng_vel = jax.random.split(rng)
        qpos = self._init_qpos + jax.random.uniform(
            rng_pos, (self.sys.nq,), minval=-s, maxval=s)
        qvel = self._init_qvel + jax.random.uniform(
            rng_vel, (self.sys.nv,), minval=-s, maxval=s)
        return qpos, qvel

    def _post_physics(self, data):
        """Hook for extra computation after stepping (Ant fills cfrc_ext)."""
        return data

    def reset(self, rng: jax.Array) -> State:
        qpos, qvel = self._reset_noise(rng)
        data = self._post_physics(self.pipeline_init(qpos, qvel))
        obs = self.obs_from_data(data)
        zero = jnp.zeros(())
        _, metrics = self.reward_terms(zero, jnp.ones(()), jnp.zeros(self.sys.nu), data)
        metrics = {**metrics, "x_position": self.forward_pos(data) * 0.0,
                   "x_velocity": zero}
        return State(pipeline_state=data, obs=obs, reward=zero, done=zero,
                     metrics=metrics)

    def step(self, state: State, action: jax.Array) -> State:
        data0 = state.pipeline_state
        # MuJoCo clamps ctrl to ctrlrange internally; mirror that explicitly.
        ctrl = jnp.clip(action, self._ctrl_low, self._ctrl_high)
        data = self._post_physics(self.pipeline_step(data0, ctrl))

        x_before = self.forward_pos(data0)
        x_after = self.forward_pos(data)
        x_velocity = (x_after - x_before) / self.dt

        healthy = self.is_healthy(data).astype(jnp.float32)
        # ctrl_cost uses the raw action, as gymnasium does.
        reward, metrics = self.reward_terms(x_velocity, healthy, action, data)
        # Merge over the existing dict: brax's wrappers add their own keys
        # (e.g. AutoResetWrapper's 'reward') that must survive our update.
        metrics = {**state.metrics, **metrics,
                   "x_position": x_after, "x_velocity": x_velocity}

        obs = self.obs_from_data(data)
        done = 1.0 - healthy  # terminate_when_unhealthy=True
        return state.replace(pipeline_state=data, obs=obs, reward=reward,
                             done=done, metrics=metrics)


class HopperMjx(GymLocomotionEnv):
    """Hopper-v5: obs = qpos[1:] + clip(qvel, ±10); healthy needs z > 0.7,
    |angle| < 0.2, and state[2:] within ±100."""

    xml_file = "hopper.xml"
    frame_skip = 4
    healthy_z_range = (0.7, float("inf"))
    healthy_angle_range = (-0.2, 0.2)
    healthy_state_range = (-100.0, 100.0)

    def is_healthy(self, data):
        qpos = jnp.asarray(data.qpos)
        qvel = jnp.asarray(data.qvel)
        state = jnp.concatenate((qpos, qvel))[2:]
        z, angle = qpos[1], qpos[2]
        min_s, max_s = self.healthy_state_range
        healthy_state = jnp.all((min_s < state) & (state < max_s))
        healthy_z = (self.healthy_z_range[0] < z) & (z < self.healthy_z_range[1])
        healthy_angle = ((self.healthy_angle_range[0] < angle)
                         & (angle < self.healthy_angle_range[1]))
        return healthy_state & healthy_z & healthy_angle

    def task_spec(self):
        spec = super().task_spec()
        spec["healthy"] = {
            "z_range": list(self.healthy_z_range),
            "angle_range": list(self.healthy_angle_range),
            "state_range": list(self.healthy_state_range),
        }
        return spec


class Walker2dMjx(GymLocomotionEnv):
    """Walker2d-v5: obs = qpos[1:] + clip(qvel, ±10); healthy needs
    0.8 < z < 2.0 and |angle| < 1.0 (v5 checks only these two)."""

    xml_file = "walker2d_v5.xml"
    frame_skip = 4
    healthy_z_range = (0.8, 2.0)
    healthy_angle_range = (-1.0, 1.0)

    def is_healthy(self, data):
        qpos = jnp.asarray(data.qpos)
        z, angle = qpos[1], qpos[2]
        healthy_z = (self.healthy_z_range[0] < z) & (z < self.healthy_z_range[1])
        healthy_angle = ((self.healthy_angle_range[0] < angle)
                         & (angle < self.healthy_angle_range[1]))
        return healthy_z & healthy_angle

    def task_spec(self):
        spec = super().task_spec()
        spec["healthy"] = {
            "z_range": list(self.healthy_z_range),
            "angle_range": list(self.healthy_angle_range),
        }
        return spec


class AntMjx(GymLocomotionEnv):
    """Ant-v5: obs = qpos[2:] + qvel + clip(cfrc_ext, ±1)[1:] flattened;
    reward includes a contact cost; qvel noise is normal, not uniform.

    cfrc_ext is not populated by mjx.step, so we run mjx.rne_postconstraint
    after stepping — the same call gymnasium makes after mj_step."""

    xml_file = "ant.xml"
    frame_skip = 5
    ctrl_cost_weight = 0.5
    contact_cost_weight = 5e-4
    contact_force_range = (-1.0, 1.0)
    healthy_z_range = (0.2, 1.0)
    reset_noise_scale = 0.1
    skipped_qpos = 2
    main_body = 1  # "torso"

    def _reset_noise(self, rng):
        s = self.reset_noise_scale
        rng_pos, rng_vel = jax.random.split(rng)
        qpos = self._init_qpos + jax.random.uniform(
            rng_pos, (self.sys.nq,), minval=-s, maxval=s)
        qvel = self._init_qvel + s * jax.random.normal(rng_vel, (self.sys.nv,))
        return qpos, qvel

    def _post_physics(self, data):
        return mjx.rne_postconstraint(self.sys, data)

    def contact_forces(self, data):
        lo, hi = self.contact_force_range
        return jnp.clip(jnp.asarray(data.cfrc_ext), lo, hi)

    def obs_from_data(self, data):
        position = jnp.asarray(data.qpos)[self.skipped_qpos:]
        velocity = jnp.asarray(data.qvel)  # NOT clipped in Ant-v5
        contact_force = self.contact_forces(data)[1:].ravel()
        return jnp.concatenate((position, velocity, contact_force))

    def is_healthy(self, data):
        qpos = jnp.asarray(data.qpos)
        qvel = jnp.asarray(data.qvel)
        state = jnp.concatenate((qpos, qvel))
        z = qpos[2]
        finite = jnp.all(jnp.isfinite(state))
        z_ok = (self.healthy_z_range[0] <= z) & (z <= self.healthy_z_range[1])
        return finite & z_ok

    def forward_pos(self, data):
        return jnp.asarray(data.xpos)[self.main_body, 0]

    def reward_terms(self, x_velocity, healthy, action, data):
        forward_reward = self.forward_reward_weight * x_velocity
        healthy_reward = healthy * self.healthy_reward
        ctrl_cost = self.ctrl_cost_weight * jnp.sum(jnp.square(action))
        contact_cost = self.contact_cost_weight * jnp.sum(
            jnp.square(self.contact_forces(data)))
        reward = forward_reward + healthy_reward - ctrl_cost - contact_cost
        metrics = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_contact": -contact_cost,
            "reward_survive": healthy_reward,
        }
        return reward, metrics

    def task_spec(self):
        spec = super().task_spec()
        spec["reward"]["contact_cost_weight"] = self.contact_cost_weight
        spec["contact_force_range"] = list(self.contact_force_range)
        spec["healthy"] = {"z_range": list(self.healthy_z_range)}
        spec["main_body"] = self.main_body
        return spec


class AntFoodMjx(AntMjx):
    """Ant that forages for randomly spawned "food" instead of just running +x.

    This is a *custom* task, not a gymnasium port, so it is deliberately kept
    out of the parity gate (mjx_envs/validate.py, which iterates ``ENVS``).

    Food is a 2-D ground target carried in ``state.info["food"]``. The dense
    reward rewards closing the distance to the food (replacing Ant's pure
    forward-velocity term); reaching it (torso within ``reach_radius``) grants a
    sparse bonus and respawns the food at a fresh nearby location. The
    observation is the standard Ant obs with the food's position *relative to the
    torso* (2 extra dims) appended, so the policy can actually see its target.

    Auto-reset handling: brax's ``AutoResetWrapper`` restores ``pipeline_state``
    and ``obs`` to the episode's first state on ``done`` but does NOT touch our
    custom ``info`` keys. We therefore stash ``first_food`` at reset and re-sync
    ``info["food"]`` to it whenever ``info["steps"] == 0`` (the first step of an
    episode), keeping obs/reward consistent with the restored physics state.
    """

    # Weight on the "closed distance per second" term (mirrors forward_reward).
    food_reward_weight = 1.0
    food_bonus = 10.0            # sparse reward for reaching a food
    reach_radius = 1.0           # torso-to-food xy distance counted as "reached"
    spawn_radius_range = (3.0, 8.0)  # new food spawns this far from the torso
    # Anti-flip: also require the torso's local +z to point generally upward
    # (world-z component > this). None disables it. Only affects *future*
    # training — a flipped ant becomes "unhealthy" and the episode resets
    # instead of letting it flail on its back. Not a gym-parity env, so this is
    # free to diverge from Ant-v5.
    healthy_upright_min = 0.0

    def _torso_xy(self, data):
        return jnp.asarray(data.xpos)[self.main_body, :2]

    def _sample_food(self, rng, center):
        """Random point on an annulus around ``center``; returns (xy, rng)."""
        rng, k_r, k_theta = jax.random.split(rng, 3)
        lo, hi = self.spawn_radius_range
        r = jax.random.uniform(k_r, (), minval=lo, maxval=hi)
        theta = jax.random.uniform(k_theta, (), minval=0.0, maxval=2.0 * jnp.pi)
        offset = jnp.array([r * jnp.cos(theta), r * jnp.sin(theta)])
        return center + offset, rng

    def _obs_with_food(self, data, food_xy):
        base = self.obs_from_data(data)
        rel = food_xy - self._torso_xy(data)  # target in the ant's frame
        return jnp.concatenate((base, rel))

    def _reached(self, data, food_xy):
        """1.0 when the food counts as collected. Base task: torso within
        ``reach_radius``. Subclasses can require a more specific contact."""
        dist = jnp.linalg.norm(food_xy - self._torso_xy(data))
        return (dist < self.reach_radius).astype(jnp.float32)

    def is_healthy(self, data):
        base = super().is_healthy(data)
        if self.healthy_upright_min is None:
            return base
        # xmat[main_body] is the torso rotation; element [2,2] is the world-z
        # component of the torso's local +z axis (≈+1 upright, ≈-1 flipped).
        up_z = jnp.asarray(data.xmat)[self.main_body].reshape(-1)[8]
        return base & (up_z > self.healthy_upright_min)

    def task_spec(self):
        spec = super().task_spec()
        spec["task"] = "forage: collect food by reaching it (torso proximity)"
        spec["food"] = {
            "food_reward_weight": self.food_reward_weight,
            "food_bonus": self.food_bonus,
            "reach_radius": self.reach_radius,
            "spawn_radius_range": list(self.spawn_radius_range),
        }
        spec["healthy"]["upright_min"] = self.healthy_upright_min
        return spec

    def _food_metrics(self, reward_food, ctrl_cost, contact_cost,
                      healthy_reward, food_dist, reached, x_position,
                      x_velocity):
        return {
            "reward_food": reward_food,
            "reward_ctrl": -ctrl_cost,
            "reward_contact": -contact_cost,
            "reward_survive": healthy_reward,
            "food_dist": food_dist,
            "food_reached": reached,
            "x_position": x_position,
            "x_velocity": x_velocity,
        }

    def reset(self, rng: jax.Array) -> State:
        rng, rng_reset = jax.random.split(rng)
        qpos, qvel = self._reset_noise(rng_reset)
        data = self._post_physics(self.pipeline_init(qpos, qvel))
        center = self._torso_xy(data)
        food_xy, rng = self._sample_food(rng, center)

        obs = self._obs_with_food(data, food_xy)
        zero = jnp.zeros(())
        metrics = self._food_metrics(
            reward_food=zero, ctrl_cost=zero, contact_cost=zero,
            healthy_reward=zero,
            food_dist=jnp.linalg.norm(food_xy - center),
            reached=zero, x_position=zero, x_velocity=zero)
        info = {"food": food_xy, "first_food": food_xy, "rng": rng}
        return State(pipeline_state=data, obs=obs, reward=zero, done=zero,
                     metrics=metrics, info=info)

    def step(self, state: State, action: jax.Array) -> State:
        data0 = state.pipeline_state

        # On the first step of an episode the AutoResetWrapper has restored the
        # physics to the episode start but left our food stale — re-sync it.
        if "steps" in state.info:
            fresh = (state.info["steps"] == 0)
            food_xy = jnp.where(fresh, state.info["first_food"], state.info["food"])
        else:
            food_xy = state.info["food"]
        rng = state.info["rng"]

        ctrl = jnp.clip(action, self._ctrl_low, self._ctrl_high)
        data = self._post_physics(self.pipeline_step(data0, ctrl))

        xy_before = self._torso_xy(data0)
        xy_after = self._torso_xy(data)
        dist_before = jnp.linalg.norm(food_xy - xy_before)
        dist_after = jnp.linalg.norm(food_xy - xy_after)
        approach_speed = (dist_before - dist_after) / self.dt

        healthy = self.is_healthy(data).astype(jnp.float32)
        reached = self._reached(data, food_xy)

        # Respawn the food (relative to where the ant now is) once reached.
        new_food, rng = self._sample_food(rng, xy_after)
        food_next = jnp.where(reached > 0, new_food, food_xy)

        forward_reward = self.food_reward_weight * approach_speed
        healthy_reward = healthy * self.healthy_reward
        ctrl_cost = self.ctrl_cost_weight * jnp.sum(jnp.square(action))
        contact_cost = self.contact_cost_weight * jnp.sum(
            jnp.square(self.contact_forces(data)))
        reward = (forward_reward + self.food_bonus * reached + healthy_reward
                  - ctrl_cost - contact_cost)

        obs = self._obs_with_food(data, food_next)
        done = 1.0 - healthy
        metrics = {**state.metrics, **self._food_metrics(
            reward_food=forward_reward + self.food_bonus * reached,
            ctrl_cost=ctrl_cost, contact_cost=contact_cost,
            healthy_reward=healthy_reward,
            food_dist=jnp.linalg.norm(food_next - xy_after),
            reached=reached, x_position=xy_after[0], x_velocity=approach_speed)}
        info = {**state.info, "food": food_next, "rng": rng}
        return state.replace(pipeline_state=data, obs=obs, reward=reward,
                             done=done, metrics=metrics, info=info)


class AntFood2LegMjx(AntFoodMjx):
    """AntFood variant where the food only counts as collected when the ant
    plants at least ``min_feet`` of its feet on it — i.e. it must actually stand
    over the target with two legs, not merely wander its torso across it.

    "Feet" are the four ankle capsule geoms; a foot is "on" the food when the
    geom's world xy is within ``foot_radius`` of the food. The dense approach
    reward stays torso-based (to guide the ant there); only the pickup bonus and
    respawn use the stricter two-foot test. The observation is unchanged (still
    the food vector relative to the torso), so a food policy can warm-start it."""

    foot_radius = 0.5
    min_feet = 2
    foot_geom_names = ("left_ankle_geom", "right_ankle_geom",
                       "third_ankle_geom", "fourth_ankle_geom")

    def __init__(self):
        super().__init__()
        self._foot_geom_ids = jnp.asarray(
            [mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, n)
             for n in self.foot_geom_names])

    def _reached(self, data, food_xy):
        feet_xy = jnp.asarray(data.geom_xpos)[self._foot_geom_ids, :2]
        dists = jnp.linalg.norm(feet_xy - food_xy[None, :], axis=1)
        n_on = jnp.sum(dists < self.foot_radius)
        return (n_on >= self.min_feet).astype(jnp.float32)

    def task_spec(self):
        spec = super().task_spec()
        spec["task"] = (f"forage: collect food with >= {self.min_feet} feet "
                        "within foot_radius")
        spec["food"]["foot_radius"] = self.foot_radius
        spec["food"]["min_feet"] = self.min_feet
        spec["food"]["foot_geom_names"] = list(self.foot_geom_names)
        return spec


class AntGetUpMjx(AntMjx):
    """Recovery task: the ant starts fallen at a random orientation and is
    rewarded for righting itself into an upright, standing posture. There is no
    forward/food objective and — unlike every other env here — NO orientation or
    height termination: the ant keeps the whole episode to get up, so it can
    actually learn to recover instead of being reset the instant it is not
    upright. Only a non-finite state ends the episode.

    Observation is the standard Ant observation (the torso quaternion in
    qpos[3:7] tells the policy which way is up), so obs_dim is unchanged (105)."""

    ctrl_cost_weight = 0.01   # much lower than Ant's 0.5: getting up needs effort
    stand_height = 0.55       # upright torso height (≈ Ant's init height)
    start_height = 0.5        # drop height at reset; random orientation tumbles it
    upright_weight = 1.0      # reward the torso "up" axis pointing up
    height_weight = 0.5       # reward the torso at standing height
    stand_bonus = 1.0         # extra when fully upright AND standing

    def _reset_noise(self, rng):
        rng, r_quat, r_joint, r_vel = jax.random.split(rng, 4)
        quat = jax.random.normal(r_quat, (4,))        # ~uniform random orientation
        quat = quat / jnp.linalg.norm(quat)
        s = self.reset_noise_scale
        qpos = self._init_qpos
        qpos = qpos.at[2].set(self.start_height)
        qpos = qpos.at[3:7].set(quat)
        qpos = qpos.at[7:].add(
            jax.random.uniform(r_joint, (self.sys.nq - 7,), minval=-s, maxval=s))
        qvel = self._init_qvel + s * jax.random.normal(r_vel, (self.sys.nv,))
        return qpos, qvel

    def is_healthy(self, data):
        # Never terminate on pose — only on a blown-up (non-finite) state.
        state = jnp.concatenate((jnp.asarray(data.qpos), jnp.asarray(data.qvel)))
        return jnp.all(jnp.isfinite(state))

    def _torso_up_z(self, data):
        # World-z component of the torso's local +z axis: +1 upright, -1 flipped.
        return jnp.asarray(data.xmat)[self.main_body].reshape(-1)[8]

    def reward_terms(self, x_velocity, healthy, action, data):
        up_z = self._torso_up_z(data)
        z = jnp.asarray(data.xpos)[self.main_body, 2]
        upright = 0.5 * (up_z + 1.0)                       # 0 flipped .. 1 upright
        height = jnp.clip(z / self.stand_height, 0.0, 1.0)
        standing = ((up_z > 0.9) & (z > 0.9 * self.stand_height)).astype(jnp.float32)
        ctrl_cost = self.ctrl_cost_weight * jnp.sum(jnp.square(action))
        reward = (self.upright_weight * upright + self.height_weight * height
                  + self.stand_bonus * standing - ctrl_cost)
        metrics = {
            "reward_upright": self.upright_weight * upright,
            "reward_height": self.height_weight * height,
            "reward_stand": self.stand_bonus * standing,
            "reward_ctrl": -ctrl_cost,
        }
        return reward, metrics

    def task_spec(self):
        spec = super().task_spec()
        spec["task"] = "recover: right itself to upright/standing from a fallen start"
        spec["reward"].update({
            "upright_weight": self.upright_weight,
            "height_weight": self.height_weight,
            "stand_bonus": self.stand_bonus,
        })
        spec["stand_height"] = self.stand_height
        spec["start_height"] = self.start_height
        spec["healthy"] = {"terminate_when_unhealthy": False, "finite_only": True}
        return spec


class MonsterMjx(AntMjx):
    """Forward locomotion for a generated monster morphology (mjx_envs/
    monsters.py). The task is Ant's — run +x, stay healthy, pay ctrl and
    contact costs — but the body is built from a MonsterSpec instead of
    ant.xml, so obs/action sizes vary per monster.

    A *custom* env (not a gymnasium port): kept out of the parity gate.
    Health = Ant's finite + z-range check (the range derived from the spec's
    standing geometry unless it overrides healthy_z_range), plus the same
    anti-flip upright test the food envs use, since a monster on its back
    often still satisfies a height-only check."""

    healthy_upright_min = 0.0  # None disables the upright requirement

    def __init__(self, spec):
        self.monster_spec = spec
        self.xml_file = f"monsters/{spec.name}.xml"  # instance-level, generated
        self.healthy_z_range = (spec.healthy_z_range
                                or monsters.default_healthy_z_range(spec))
        self._init_from_model(monsters.spec_to_model(spec))

    def is_healthy(self, data):
        base = super().is_healthy(data)
        if self.healthy_upright_min is None:
            return base
        up_z = jnp.asarray(data.xmat)[self.main_body].reshape(-1)[8]
        return base & (up_z > self.healthy_upright_min)

    def task_spec(self):
        spec = super().task_spec()
        spec["task"] = "monster locomotion: run +x with a generated morphology"
        spec["healthy"]["upright_min"] = self.healthy_upright_min
        # The full morphology, so the run is reproducible from config.json
        # alone even if the preset/generator changes later.
        spec["monster"] = monsters.spec_to_dict(self.monster_spec)
        return spec


ENVS = {
    "Hopper-v5": HopperMjx,
    "Walker2d-v5": Walker2dMjx,
    "Ant-v5": AntMjx,
}

# Custom (non-gymnasium) tasks. Kept separate from ENVS so the parity gate
# (validate.py) only ever runs against the faithful ports.
CUSTOM_ENVS = {
    "AntFood-v5": AntFoodMjx,
    "AntFood2Leg-v5": AntFood2LegMjx,
    "AntGetUp-v5": AntGetUpMjx,
}

ALL_ENVS = {**ENVS, **CUSTOM_ENVS}


def make_env(env_id: str) -> GymLocomotionEnv:
    # Monster-<name>-v0 ids resolve through the monster registry (presets or
    # assets/monsters/<name>.spec.json), so any generated morphology is
    # trainable by id without touching this table.
    monster_name = monsters.name_from_env_id(env_id)
    if monster_name is not None:
        return MonsterMjx(monsters.load_spec(monster_name))
    if env_id not in ALL_ENVS:
        raise ValueError(
            f"No MJX env for {env_id!r}. Available: {sorted(ALL_ENVS)} "
            f"plus {monsters.ENV_PREFIX}<name>-{monsters.ENV_VERSION} for "
            f"monsters {sorted(monsters.MONSTERS)} or any "
            "assets/monsters/<name>.spec.json")
    return ALL_ENVS[env_id]()
