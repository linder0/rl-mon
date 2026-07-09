import type { MujocoSim } from "./mujocoSim";
import type { Policy } from "./policy";
import type { Renderer } from "./renderer";

const MAX_EPISODE_STEPS = 1000; // gym TimeLimit for these tasks

export interface LoopStats {
  fps: number;
  step: number;
  episode: number;
  distance: number;
  healthy: boolean;
}

export interface ParityResult {
  steps: number;
  maxActionError: number;
}

/** Drives the obs -> policy -> step -> render loop in real time (scaled by
 * `speed`), auto-resetting when the agent falls or hits the time limit. */
export class SimLoop {
  private sim: MujocoSim;
  private policy: Policy;
  private readonly renderer: Renderer;

  playing = true;
  speed = 1.0;

  private obsBuf: Float32Array;
  private acc = 0;
  private last = performance.now();
  private stepCount = 0;
  private episode = 1;
  private disposed = false;
  private stepping = false;

  private fps = 0;
  private frameTimes: number[] = [];

  onStats: (s: LoopStats) => void = () => {};
  /** Fired each control step with the observation fed to the policy and the
   * resulting action. Used to drive the live network visualization. */
  onControl: (obs: Float32Array, action: Float32Array) => void = () => {};

  constructor(sim: MujocoSim, policy: Policy, renderer: Renderer) {
    this.sim = sim;
    this.policy = policy;
    this.renderer = renderer;
    this.obsBuf = new Float32Array(sim.meta.obs_dim);
    this.renderer.setModel(sim);
  }

  /** Swap in a different agent without recreating the renderer. The previous
   * simulation's MuJoCo objects are freed to bound WASM heap growth. */
  setAgent(sim: MujocoSim, policy: Policy): void {
    const old = this.sim;
    this.sim = sim;
    this.policy = policy;
    this.obsBuf = new Float32Array(sim.meta.obs_dim);
    this.renderer.setModel(sim);
    this.reset();
    if (old && old !== sim) old.dispose();
  }

  reset(): void {
    this.sim.reset(true);
    this.stepCount = 0;
    this.acc = 0;
  }

  start(): void {
    this.last = performance.now();
    requestAnimationFrame(this.frame);
  }

  dispose(): void {
    this.disposed = true;
  }

  /** Feed the recorded (Python) observations through the JS policy and compare
   * actions. This isolates the obs-normalization + ONNX inference path from any
   * physics-engine differences, so it is an exact correctness gate. */
  async verifyPolicyParity(): Promise<ParityResult> {
    const trace = this.sim.meta.parity_trace;
    let maxErr = 0;
    for (const step of trace) {
      const obs = Float32Array.from(step.obs);
      const action = await this.policy.act(obs);
      for (let i = 0; i < action.length; i++) {
        maxErr = Math.max(maxErr, Math.abs(action[i] - step.action[i]));
      }
    }
    return { steps: trace.length, maxActionError: maxErr };
  }

  private async controlStep(): Promise<void> {
    // Capture the agent so a swap mid-inference can't apply a stale action to a
    // freshly-loaded (different-dimensioned) simulation.
    const sim = this.sim;
    const policy = this.policy;
    const obs = sim.getObs(this.obsBuf);
    const action = await policy.act(obs);
    if (this.sim !== sim || this.policy !== policy) return;

    sim.applyAction(action);
    this.onControl(obs, action);
    this.stepCount++;

    const unhealthy =
      sim.meta.healthy.terminate_when_unhealthy && !sim.isHealthy();
    if (unhealthy || this.stepCount >= MAX_EPISODE_STEPS) {
      this.episode++;
      this.reset();
    }
  }

  private frame = async (): Promise<void> => {
    if (this.disposed) return;
    // The whole frame is wrapped so a transient error (e.g. a MuJoCo typed-array
    // view detached by WASM heap growth during a model switch) can never break
    // the requestAnimationFrame chain and force a page reload.
    try {
      const now = performance.now();
      const realDt = Math.min((now - this.last) / 1000, 0.1);
      this.last = now;

      this.frameTimes.push(now);
      while (this.frameTimes.length > 30) this.frameTimes.shift();
      if (this.frameTimes.length > 1) {
        const span = (now - this.frameTimes[0]) / 1000;
        this.fps = (this.frameTimes.length - 1) / span;
      }

      if (this.playing && !this.stepping) {
        this.stepping = true;
        try {
          const dt = this.sim.meta.dt;
          this.acc += realDt * this.speed;
          let did = 0;
          while (this.acc >= dt && did < 8) {
            await this.controlStep();
            this.acc -= dt;
            did++;
          }
          if (this.acc > dt * 8) this.acc = 0; // avoid spiral of death
        } finally {
          this.stepping = false;
        }
      }

      this.renderer.update(this.sim);
      this.renderer.render();

      this.onStats({
        fps: this.fps,
        step: this.stepCount,
        episode: this.episode,
        distance: this.sim.rootX(),
        healthy: this.sim.isHealthy(),
      });
    } catch (err) {
      console.warn("frame skipped due to error:", err);
    } finally {
      if (!this.disposed) requestAnimationFrame(this.frame);
    }
  };
}
