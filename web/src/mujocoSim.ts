import loadMujoco from "mujoco-js";
import type { MainModule, MjModel, MjData } from "mujoco-js";
import type { EnvMeta } from "./types";

let mujocoPromise: Promise<MainModule> | null = null;

/** Load the MuJoCo WASM module once and mount a writable virtual filesystem. */
async function getMujoco(): Promise<MainModule> {
  if (!mujocoPromise) {
    mujocoPromise = loadMujoco().then((m) => {
      const fs = (m as unknown as { FS: any; MEMFS: any });
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
    const ctrl = this.data.ctrl as Float64Array;
    for (let i = 0; i < this.meta.nu; i++) ctrl[i] = 0;
    this.mujoco.mj_forward(this.model, this.data);
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
      }
    }
    return obs;
  }

  /** Apply an action and advance the physics `frame_skip` steps. MuJoCo clamps
   * ctrl to each actuator's ctrlrange internally, matching gym. */
  applyAction(action: Float32Array | number[]): void {
    const ctrl = this.data.ctrl as Float64Array;
    for (let i = 0; i < this.meta.nu; i++) ctrl[i] = action[i];
    for (let f = 0; f < this.meta.frame_skip; f++) {
      this.mujoco.mj_step(this.model, this.data);
    }
    if (this.meta.needs_rne) {
      // Populate cfrc_ext (contact forces) for the observation, like gym does.
      this.mujoco.mj_rnePostConstraint(this.model, this.data);
    }
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

  /** Free the underlying MuJoCo objects. Call when swapping this sim out so the
   * WASM heap does not grow unbounded across model switches. */
  dispose(): void {
    try { (this.data as unknown as { delete?: () => void }).delete?.(); } catch { /* noop */ }
    try { (this.model as unknown as { delete?: () => void }).delete?.(); } catch { /* noop */ }
  }
}
