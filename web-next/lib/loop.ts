import type { MujocoSim } from "./mujocoSim";
import type { Policy } from "./policy";
import type { RendererLike } from "./rendererLike";

const MAX_EPISODE_STEPS = 1000; // gym TimeLimit for these tasks

export interface LoopStats {
  fps: number;
  step: number;
  episode: number;
  distance: number;
  healthy: boolean;
  /** Food pickups this episode (AntFood task only; undefined otherwise). */
  pickups?: number;
  /** True while the recovery (get-up) policy is driving in combined mode. */
  recovering?: boolean;
}

/** A secondary policy that takes over when the ant is flipped, composing a
 * trained get-up skill with the main (forage) policy — a hand-written
 * hierarchical switch. `obsDim` is the recovery policy's input width; it reads
 * the first `obsDim` dims of the main observation (the shared Ant obs). */
export interface Recovery {
  policy: Policy;
  obsDim: number;
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
  private readonly renderer: RendererLike;

  playing = true;
  speed = 1.0;

  /** Optional get-up policy for combined forage+recover mode (null = off). */
  recovery: Recovery | null = null;
  private recovering = false;
  private readonly recoverOn = 0.4; // switch to get-up when up-z drops below
  private readonly recoverOff = 0.85; // switch back to forage above (hysteresis)

  private obsBuf: Float32Array;
  private acc = 0;
  private substepsLeft = 0; // physics substeps remaining in the control period
  private last = performance.now();
  private stepCount = 0;
  private episode = 1;
  private pickups = 0;
  private disposed = false;
  private stepping = false;

  private fps = 0;
  private frameTimes: number[] = [];

  onStats: (s: LoopStats) => void = () => {};
  /** Fired each control step with the observation fed to the policy and the
   * resulting action. Used to drive the live network visualization. */
  onControl: (obs: Float32Array, action: Float32Array) => void = () => {};

  constructor(sim: MujocoSim, policy: Policy, renderer: RendererLike) {
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

  /** Swap only the policy, keeping the running simulation and its current pose.
   * Used by "learning mode" so the same body visibly changes behavior as you
   * scrub across training checkpoints (no reset, so the transition is seen). */
  setPolicy(policy: Policy): void {
    this.policy = policy;
  }

  reset(): void {
    this.sim.reset(true);
    this.stepCount = 0;
    this.pickups = 0;
    this.acc = 0;
    this.substepsLeft = 0; // next substep starts with a fresh policy query
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

  /** Start of a control period: build the observation, query the policy, and
   * latch the resulting ctrl (held for the next frame_skip substeps, exactly
   * like gym's do_simulation zero-order hold). Returns false if the agent was
   * swapped out mid-inference. */
  private async controlBoundary(): Promise<boolean> {
    // Capture the agent so a swap mid-inference can't apply a stale action to
    // a freshly-loaded (different-dimensioned) simulation.
    const sim = this.sim;
    const policy = this.policy;
    const recovery = this.recovery;
    if (sim.meta.needs_rne) sim.computeContactForces();
    // Check/respawn the foraging target on the post-step pose, then observe it
    // (matches the training step order in AntFoodMjx.step).
    if (sim.hasFood && sim.updateFood()) this.pickups++;
    const obs = sim.getObs(this.obsBuf);

    // Combined mode: hand control to the get-up policy while flipped, hand it
    // back once upright again (hysteresis avoids rapid toggling at the border).
    if (recovery) {
      const up = sim.torsoUpZ();
      if (!this.recovering && up < this.recoverOn) this.recovering = true;
      else if (this.recovering && up > this.recoverOff) this.recovering = false;
    } else {
      this.recovering = false;
    }

    const action =
      recovery && this.recovering
        ? await recovery.policy.act(obs.subarray(0, recovery.obsDim))
        : await policy.act(obs);
    if (this.sim !== sim || this.policy !== policy) return false;

    sim.setCtrl(action);
    this.onControl(obs, action);
    return true;
  }

  /** Shove the ant over (combined-mode demo of the recovery policy). */
  knockOver(): void {
    this.sim.knockOver();
  }

  /** Adjust the foraging spawn distance live (AntFood tasks; no-op otherwise). */
  setFoodSpawnMax(max: number | null): void {
    this.sim.setFoodSpawnMax(max);
  }

  /** End of a control period: episode bookkeeping, matching gym's post-step
   * health/time-limit checks. */
  private endOfControlPeriod(): void {
    this.stepCount++;
    // In combined mode we never end the episode for being unhealthy — the whole
    // point is to let the get-up policy recover from a flip instead of resetting.
    const unhealthy =
      !this.recovery &&
      this.sim.meta.healthy.terminate_when_unhealthy &&
      !this.sim.isHealthy();
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
          // Advance physics one model timestep at a time so the rendered pose
          // updates every frame even when the control period is long (Ant's
          // dt is 50 ms — stepping it whole made motion visibly chunky). The
          // policy is only queried at control boundaries (every frame_skip
          // substeps), so the simulated trajectory is identical to gym's.
          const { timestep, frame_skip } = this.sim.meta;
          const maxSubsteps = 8 * frame_skip; // same cap as 8 control steps
          this.acc += realDt * this.speed;
          let did = 0;
          while (this.acc >= timestep && did < maxSubsteps) {
            if (this.substepsLeft === 0) {
              if (!(await this.controlBoundary())) break;
              this.substepsLeft = frame_skip;
            }
            this.sim.substep();
            this.substepsLeft--;
            if (this.substepsLeft === 0) this.endOfControlPeriod();
            this.acc -= timestep;
            did++;
          }
          if (this.acc > timestep * maxSubsteps) this.acc = 0; // avoid spiral of death
        } finally {
          this.stepping = false;
        }
      }

      this.renderer.update(this.sim);
      await this.renderer.render();

      this.onStats({
        fps: this.fps,
        step: this.stepCount,
        episode: this.episode,
        distance: this.sim.rootX(),
        healthy: this.sim.isHealthy(),
        pickups: this.sim.hasFood ? this.pickups : undefined,
        recovering: this.recovery ? this.recovering : undefined,
      });
    } catch (err) {
      console.warn("frame skipped due to error:", err);
    } finally {
      if (!this.disposed) requestAnimationFrame(this.frame);
    }
  };
}
