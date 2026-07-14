import loadMujoco from "mujoco-js";
import type { MainModule, MjModel, MjData } from "mujoco-js";
import type { EnvMeta } from "./types";

let mujocoPromise: Promise<MainModule> | null = null;

/** Load the MuJoCo WASM module once and mount a writable virtual filesystem. */
async function getMujoco(): Promise<MainModule> {
  if (!mujocoPromise) {
    mujocoPromise = loadMujoco().then((m) => {
      const fs = m as unknown as { FS: any; MEMFS: any };
      fs.FS.mkdir("/working");
      fs.FS.mount(fs.MEMFS, { root: "." }, "/working");
      return m;
    });
  }
  return mujocoPromise;
}

/** A single-agent MuJoCo simulation that steps exactly like the Gymnasium env
 * the policy was trained in (same observation, frame_skip, and health checks). */
export class MujocoSim {
  readonly mujoco: MainModule;
  readonly model: MjModel;
  readonly data: MjData;
  readonly meta: EnvMeta;

  /** Foraging target (world x, y), tracked in JS for the AntFood task. Only
   * meaningful when meta.food is set; see spawnFood/updateFood/getObs. */
  foodX = 0;
  foodY = 0;
  get hasFood(): boolean {
    return this.meta.food != null;
  }

  private constructor(mujoco: MainModule, model: MjModel, data: MjData, meta: EnvMeta) {
    this.mujoco = mujoco;
    this.model = model;
    this.data = data;
    this.meta = meta;
  }

  static async create(meta: EnvMeta, xmlUrl: string): Promise<MujocoSim> {
    const mujoco = await getMujoco();
    const xml = await (await fetch(xmlUrl)).text();
    const path = `/working/${meta.model_xml}`;
    (mujoco as unknown as { FS: any }).FS.writeFile(path, xml);
    const model = mujoco.MjModel.loadFromXML(path);
    const data = new mujoco.MjData(model);
    const sim = new MujocoSim(mujoco, model, data, meta);
    sim.reset(false);
    return sim;
  }

  /** Reset to the fixed initial state. With noise (default) matches gym's
   * uniform reset perturbation in magnitude; without noise gives the exact
   * deterministic start used by the parity trace. */
  reset(withNoise = true): void {
    const { init_qpos, init_qvel, reset_noise_scale, nq, nv } = this.meta;
    const qpos = this.data.qpos as Float64Array;
    const qvel = this.data.qvel as Float64Array;
    const s = withNoise ? reset_noise_scale : 0;
    for (let i = 0; i < nq; i++) qpos[i] = init_qpos[i] + s * (Math.random() * 2 - 1);
    for (let i = 0; i < nv; i++) qvel[i] = init_qvel[i] + s * (Math.random() * 2 - 1);
    // AntGetUp: drop the ant fallen at a random orientation so it must recover
    // (mirrors AntGetUpMjx._reset_noise). Free joint quat lives at qpos[3:7].
    if (this.meta.getup && withNoise) {
      qpos[2] = this.meta.getup.start_height;
      const g = () => {
        let u = 0, v = 0;
        while (u === 0) u = Math.random();
        while (v === 0) v = Math.random();
        return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
      };
      const q = [g(), g(), g(), g()];
      const n = Math.hypot(q[0], q[1], q[2], q[3]) || 1;
      for (let i = 0; i < 4; i++) qpos[3 + i] = q[i] / n;
    }
    const ctrl = this.data.ctrl as Float64Array;
    for (let i = 0; i < this.meta.nu; i++) ctrl[i] = 0;
    this.mujoco.mj_forward(this.model, this.data);
    if (this.meta.food) this.spawnFood();
  }

  /** Torso (main-body) world xy, the reference frame for the food target. */
  private torsoXY(): [number, number] {
    const b = this.meta.food?.main_body ?? 1;
    const xpos = this.data.xpos as Float64Array;
    return [xpos[b * 3 + 0], xpos[b * 3 + 1]];
  }

  /** Viewer override of the spawn annulus' outer radius (meters); null keeps
   * the trained spawn_max. Lets the UI probe generalization to farther food. */
  private foodSpawnMax: number | null = null;

  /** Set (or clear) the spawn-distance override and respawn the food so the
   * change is immediately visible. */
  setFoodSpawnMax(max: number | null): void {
    if (!this.meta.food) return;
    this.foodSpawnMax = max;
    this.spawnFood();
  }

  /** Spawn the food on an annulus around the torso, mirroring AntFoodMjx. */
  private spawnFood(): void {
    const f = this.meta.food!;
    const [cx, cy] = this.torsoXY();
    const max = this.foodSpawnMax ?? f.spawn_max;
    const min = Math.min(f.spawn_min, max);
    const r = min + Math.random() * (max - min);
    const theta = Math.random() * 2 * Math.PI;
    this.foodX = cx + r * Math.cos(theta);
    this.foodY = cy + r * Math.sin(theta);
  }

  /** True when the food counts as collected: either the torso is within
   * reach_radius (base task) or >= min_feet ankle geoms are within foot_radius
   * (AntFood2Leg). Mirrors AntFood(2Leg)Mjx._reached. */
  private foodReached(): boolean {
    const f = this.meta.food!;
    if (f.foot_geoms && f.foot_geoms.length) {
      const r = f.foot_radius ?? 0.5;
      const need = f.min_feet ?? 2;
      const gx = this.data.geom_xpos as Float64Array;
      let on = 0;
      for (const g of f.foot_geoms) {
        if (Math.hypot(this.foodX - gx[g * 3 + 0], this.foodY - gx[g * 3 + 1]) < r) on++;
      }
      return on >= need;
    }
    const [tx, ty] = this.torsoXY();
    return Math.hypot(this.foodX - tx, this.foodY - ty) < f.reach_radius;
  }

  /** If the food has been reached, respawn it and report the pickup. Call once
   * per control step, before getObs (matches the training step order). */
  updateFood(): boolean {
    if (!this.meta.food) return false;
    if (this.foodReached()) {
      this.spawnFood();
      return true;
    }
    return false;
  }

  /** Build the observation exactly as gymnasium's `_get_obs`, from the ordered
   * component spec (qpos slice + qvel [+ clipped contact forces]). */
  getObs(out?: Float32Array): Float32Array {
    const obs = out ?? new Float32Array(this.meta.obs_dim);
    let k = 0;
    for (const c of this.meta.obs_components) {
      if (c.kind === "qpos") {
        const qpos = this.data.qpos as Float64Array;
        for (let i = c.start ?? 0; i < this.meta.nq; i++) obs[k++] = qpos[i];
      } else if (c.kind === "qvel") {
        const qvel = this.data.qvel as Float64Array;
        const lo = c.clip ? c.clip[0] : -Infinity;
        const hi = c.clip ? c.clip[1] : Infinity;
        for (let i = c.start ?? 0; i < this.meta.nv; i++) {
          const v = qvel[i];
          obs[k++] = v < lo ? lo : v > hi ? hi : v;
        }
      } else if (c.kind === "cfrc_ext") {
        // (nbody x 6) external contact forces, skip worldbody rows, clipped.
        const cf = this.data.cfrc_ext as Float64Array;
        const lo = c.clip ? c.clip[0] : -Infinity;
        const hi = c.clip ? c.clip[1] : Infinity;
        for (let i = (c.start_body ?? 0) * 6; i < cf.length; i++) {
          const v = cf[i];
          obs[k++] = v < lo ? lo : v > hi ? hi : v;
        }
      } else if (c.kind === "food") {
        // Food target relative to the torso (see AntFoodMjx._obs_with_food).
        const [tx, ty] = this.torsoXY();
        obs[k++] = this.foodX - tx;
        obs[k++] = this.foodY - ty;
      }
    }
    return obs;
  }

  /** Latch a new action into ctrl. MuJoCo clamps ctrl to each actuator's
   * ctrlrange internally, matching gym. The control is then held constant for
   * `frame_skip` substeps (zero-order hold), exactly like gym's do_simulation
   * — but the caller advances physics one substep at a time (see SimLoop) so
   * envs with long control periods (Ant: 50 ms) still render smoothly. */
  setCtrl(action: Float32Array | number[]): void {
    const ctrl = this.data.ctrl as Float64Array;
    for (let i = 0; i < this.meta.nu; i++) ctrl[i] = action[i];
  }

  /** Advance the physics by a single model timestep with the latched ctrl. */
  substep(): void {
    this.mujoco.mj_step(this.model, this.data);
  }

  /** Populate cfrc_ext (contact forces) for the observation, like gym does
   * after its frame_skip'd mj_step. Call before getObs when needs_rne. */
  computeContactForces(): void {
    this.mujoco.mj_rnePostConstraint(this.model, this.data);
  }

  /** Mirror of gymnasium's `is_healthy` across the locomotion family: requires
   * finite state, torso height in range, and (when the env defines them) a
   * torso-angle range and a per-state-value range. */
  isHealthy(): boolean {
    const h = this.meta.healthy;
    const qpos = this.data.qpos as Float64Array;
    const qvel = this.data.qvel as Float64Array;

    for (let i = 0; i < this.meta.nq; i++) if (!Number.isFinite(qpos[i])) return false;
    for (let i = 0; i < this.meta.nv; i++) if (!Number.isFinite(qvel[i])) return false;

    const z = qpos[h.z_index];
    if (!(z > h.z_range[0] && z < h.z_range[1])) return false;

    if (h.angle_range) {
      const angle = qpos[h.angle_index];
      if (!(angle > h.angle_range[0] && angle < h.angle_range[1])) return false;
    }

    if (h.state_range) {
      const [lo, hi] = h.state_range;
      // state = concat(qpos, qvel); healthy check applies to state[2:].
      for (let i = 2; i < this.meta.nq; i++) if (!(qpos[i] > lo && qpos[i] < hi)) return false;
      for (let i = 0; i < this.meta.nv; i++) if (!(qvel[i] > lo && qvel[i] < hi)) return false;
    }
    return true;
  }

  /** x-position of the root joint, useful for a follow camera. */
  rootX(): number {
    return (this.data.qpos as Float64Array)[0];
  }

  /** World-z component of the torso's local +z axis: +1 upright, -1 flipped.
   * Used by the combined forage+recover controller to decide which policy to
   * run. Reads xmat (per-body 3x3 rotation) for the food task's main body. */
  torsoUpZ(): number {
    const mb = this.meta.food?.main_body ?? 1;
    const xmat = this.data.xmat as Float64Array;
    return xmat[mb * 9 + 8];
  }

  /** Shove the ant: pop it up and spin the torso so it tumbles over. Lets the
   * viewer demonstrate the recovery policy on demand. Acts on the free joint's
   * velocity dofs (linear 0..2, angular 3..5). */
  knockOver(): void {
    const qvel = this.data.qvel as Float64Array;
    qvel[2] += 2.5;
    qvel[3] += (Math.random() * 2 - 1) * 12;
    qvel[4] += (Math.random() * 2 - 1) * 12;
    this.mujoco.mj_forward(this.model, this.data);
  }

  /** Free the underlying MuJoCo objects. Call when swapping this sim out so the
   * WASM heap does not grow unbounded across model switches. */
  dispose(): void {
    try { (this.data as unknown as { delete?: () => void }).delete?.(); } catch { /* noop */ }
    try { (this.model as unknown as { delete?: () => void }).delete?.(); } catch { /* noop */ }
  }
}
