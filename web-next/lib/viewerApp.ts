import { MujocoSim } from "./mujocoSim";
import { Policy } from "./policy";
import { SimLoop, type LoopStats, type ParityResult } from "./loop";
import { PolicyNet } from "./net";
import { NetVizController } from "./netvizController";
import { WebGPURenderer, type Backend, type ForceVizCfg } from "@/components/scene/webgpuRenderer";
import { asset, loadJson, defaultVariant, type Variant } from "./catalog";
import type { Theme } from "./theme";
import type {
  EnvMeta,
  RunAgg,
  RunEntry,
  Timeline,
  TimelineFrame,
} from "./types";

export type { Variant };

/** "Learning mode" state pushed to the UI: whether the live run has a
 * checkpoint timeline, whether it's engaged, the frames, and the current one. */
export interface LearningState {
  available: boolean;
  active: boolean;
  frames: TimelineFrame[];
  index: number;
}

/** Env-specific task capabilities of the loaded run, driving the custom "Task"
 * controls in the UI (knock-over for get-up, spawn slider for foraging). */
export interface TaskInfo {
  /** Foraging task: the trained spawn annulus (meters), for slider defaults. */
  food: { spawnMin: number; spawnMax: number } | null;
  /** Get-up recovery task (knock-over makes sense any time). */
  getup: boolean;
}

/** Scene colors that can be customized per theme (0xRRGGBB). */
export interface SceneColors {
  bg: number;
  ground: number;
  grid: number;
  agent: number;
}

/** Persisted UI preferences applied when the renderer/loop come online. */
export interface ViewerInit {
  speed?: number;
  follow?: boolean;
  netCollapsed?: boolean;
  gridOn?: boolean;
  showForces?: boolean;
  forceViz?: ForceVizCfg;
  bloom?: number;
  /** Per-theme scene color overrides; the entry for the active theme is applied
   * on startup and whenever the theme changes. */
  colorOverrides?: Partial<Record<Theme, SceneColors>>;
}

export interface ViewerCallbacks {
  onLoading: (text: string | null) => void;
  onError: (message: string) => void;
  /** The variant actually loaded (requested one may be missing for this run). */
  onVariant: (v: Variant) => void;
  onLiveStats: (s: LoopStats) => void;
  onParity: (r: ParityResult) => void;
  onValue: (v: number) => void;
  onBackend: (b: Backend) => void;
  onSceneColors: (c: SceneColors) => void;
  onLearning: (state: LearningState) => void;
  onTask: (t: TaskInfo) => void;
}

/** Sim controller for one iteration (training run): owns the renderer, sim
 * loop, and network visualizations for the run it's given, and emits live
 * display data through callbacks. Catalog/stats concerns live in catalog.ts
 * and the React pages — this class only ever sees its own run. */
export class ViewerApp {
  private readonly renderer: WebGPURenderer;
  private readonly actorViz: NetVizController;
  private readonly criticViz: NetVizController;
  private readonly cb: ViewerCallbacks;
  private readonly init: ViewerInit;
  private theme: Theme;
  private readonly colorOverrides: Partial<Record<Theme, SceneColors>>;

  private loop: SimLoop | null = null;
  private run: RunAgg | null = null;
  private switchToken = 0;
  private disposed = false;

  private lastStatsEmit = 0;

  // ---- combined forage+recover mode ----
  private recoveryEntry: RunEntry | null = null; // the get-up policy to compose
  private recoveryOn = false;

  // ---- task controls ----
  private foodSpawnMax: number | null = null; // survives variant/run switches

  // ---- learning mode (checkpoint timeline) ----
  private basePolicy: Policy | null = null; // the loaded run's own policy
  private baseActDim = 0;
  private learningActive = false;
  private learningFrames: TimelineFrame[] = [];
  private learningIndex = 0;
  private learningToken = 0;
  private timeline: Timeline | null = null;
  private readonly framePolicyCache = new Map<string, Policy>();

  constructor(
    sceneCanvas: HTMLCanvasElement,
    actorCanvas: HTMLCanvasElement,
    criticCanvas: HTMLCanvasElement,
    callbacks: ViewerCallbacks,
    theme: Theme = "dark",
    init: ViewerInit = {},
  ) {
    this.cb = callbacks;
    this.init = init;
    this.theme = theme;
    this.colorOverrides = init.colorOverrides ?? {};
    this.renderer = new WebGPURenderer(sceneCanvas, theme);
    this.actorViz = new NetVizController(actorCanvas, theme);
    this.criticViz = new NetVizController(criticCanvas, theme);
    this.criticViz.onValue = (v) => this.cb.onValue(v);
    if (init.follow !== undefined) this.renderer.follow = init.follow;
    if (init.netCollapsed !== undefined) this.setNetCollapsed(init.netCollapsed);
  }

  /** Initialize the renderer and load the given run. `recoveryEntry` is the
   * get-up policy composed in combined mode (pass null when not applicable
   * to this run's env). */
  async start(run: RunAgg, recoveryEntry: RunEntry | null): Promise<void> {
    this.run = run;
    this.recoveryEntry = recoveryEntry;

    try {
      this.cb.onLoading("Initializing renderer…");
      const backend = await this.renderer.init();
      this.cb.onBackend(backend);
      if (this.init.gridOn !== undefined) this.renderer.setGridVisible(this.init.gridOn);
      if (this.init.forceViz !== undefined) this.renderer.setForceViz(this.init.forceViz);
      if (this.init.showForces !== undefined) this.renderer.setShowForces(this.init.showForces);
      if (this.init.bloom !== undefined) this.renderer.setBloomStrength(this.init.bloom);
      this.applyColorOverride();
    } catch (err) {
      console.error(err);
      this.cb.onError(`Failed to initialize the GPU renderer:<br><br><code>${String(err)}</code>`);
      return;
    }

    try {
      this.cb.onLoading("Loading MuJoCo…");
      await this.switchToVariant(defaultVariant(run));
    } catch (err) {
      console.error(err);
      this.cb.onError(`Failed to start the simulation:<br><br><code>${String(err)}</code>`);
    }
  }

  dispose(): void {
    this.disposed = true;
    this.loop?.dispose();
    this.actorViz.dispose();
    this.criticViz.dispose();
    this.renderer.dispose();
  }

  // ---- controls (public API for the UI) ----

  setPlaying(playing: boolean): void {
    if (this.loop) this.loop.playing = playing;
  }
  reset(): void {
    this.loop?.reset();
  }
  setSpeed(speed: number): void {
    if (this.loop) this.loop.speed = speed;
  }
  setFollow(follow: boolean): void {
    this.renderer.follow = follow;
  }
  setTheme(theme: Theme): void {
    this.theme = theme;
    this.renderer.setTheme(theme);
    this.actorViz.setTheme(theme);
    this.criticViz.setTheme(theme);
    this.applyColorOverride();
  }

  /** Apply the saved color override for the active theme (or reset to the theme
   * defaults when there's none), then report the effective colors to the UI.
   * Call after the renderer is ready and after every theme switch. */
  private applyColorOverride(): void {
    const o = this.colorOverrides[this.theme];
    if (o) {
      this.renderer.setBackgroundColor(o.bg);
      this.renderer.setGroundColor(o.ground);
      this.renderer.setGridColor(o.grid);
      this.renderer.setAgentColor(o.agent);
    } else {
      // setTheme/constructor already reset bg/ground/grid to theme defaults; the
      // agent accent is tracked separately, so clear it explicitly.
      this.renderer.setAgentColor(null);
    }
    this.cb.onSceneColors(this.renderer.sceneColors());
  }

  /** Record a per-theme color override so it's re-applied on theme switches and
   * model reloads. The UI owns persistence; this keeps the app's copy in sync. */
  setColorOverride(theme: Theme, colors: SceneColors): void {
    this.colorOverrides[theme] = colors;
  }
  setNetCollapsed(collapsed: boolean): void {
    this.actorViz.setCollapsed(collapsed);
    this.criticViz.setCollapsed(collapsed);
  }

  // ---- live scene appearance (Scene panel) ----
  setBackgroundColor(hex: number): void {
    this.renderer.setBackgroundColor(hex);
  }
  setGroundColor(hex: number): void {
    this.renderer.setGroundColor(hex);
  }
  setGridColor(hex: number): void {
    this.renderer.setGridColor(hex);
  }
  setGridVisible(on: boolean): void {
    this.renderer.setGridVisible(on);
  }
  setShowForces(on: boolean): void {
    this.renderer.setShowForces(on);
  }
  setForceViz(cfg: ForceVizCfg): void {
    this.renderer.setForceViz(cfg);
  }
  setAgentColor(hex: number | null): void {
    this.renderer.setAgentColor(hex);
  }
  setBloomStrength(v: number): void {
    this.renderer.setBloomStrength(v);
  }
  sceneColors(): { bg: number; ground: number; grid: number; agent: number } {
    return this.renderer.sceneColors();
  }

  setVariant(variant: Variant): void {
    void this.switchToVariant(variant);
  }

  /** Switch to a different iteration (run) of the same env — the iteration
   * dropdown. Renderer and loop persist; the agent and timeline swap out. */
  setRun(run: RunAgg, variant?: Variant): void {
    this.run = run;
    this.timeline = null; // the checkpoint timeline is per-run
    void this.switchToVariant(variant ?? defaultVariant(run));
  }

  /** Enable/disable combined mode: compose the current forager with the trained
   * get-up policy, which takes over whenever the ant is flipped. */
  setRecovery(on: boolean): void {
    void this.applyRecovery(on);
  }
  private async applyRecovery(on: boolean): Promise<void> {
    this.recoveryOn = on;
    if (!this.loop) return;
    if (!on || !this.recoveryEntry) {
      this.loop.recovery = null;
      return;
    }
    try {
      const meta = await loadJson<EnvMeta>(this.recoveryEntry.meta);
      const policy = await Policy.create(asset(this.recoveryEntry.onnx), meta.act_dim);
      // Guard against a variant switch that turned recovery off mid-load.
      if (this.recoveryOn && this.loop) {
        this.loop.recovery = { policy, obsDim: meta.obs_dim };
      }
    } catch (err) {
      console.error(err);
      this.cb.onError(`Failed to load the get-up policy:<br><br><code>${String(err)}</code>`);
    }
  }

  /** Shove the ant over (recovery-policy demo, or the get-up task itself). */
  knockOver(): void {
    this.loop?.knockOver();
  }

  /** Override the foraging spawn distance (meters); null restores the trained
   * range. Remembered across variant/iteration switches within the page. */
  setFoodSpawnMax(max: number | null): void {
    this.foodSpawnMax = max;
    this.loop?.setFoodSpawnMax(max);
  }

  // ---- internals ----

  private async loadAgent(
    entry: RunEntry,
  ): Promise<{ sim: MujocoSim; policy: Policy; meta: EnvMeta }> {
    const meta = await loadJson<EnvMeta>(entry.meta);
    const sim = await MujocoSim.create(meta, asset(entry.model_xml));
    const policy = await Policy.create(asset(entry.onnx), meta.act_dim);
    return { sim, policy, meta };
  }

  private async switchToVariant(variant: Variant): Promise<void> {
    const run = this.run;
    if (!run) return;
    const entry = run.variants[variant] ?? run.variants.best ?? run.variants.final;
    if (!entry) return;
    const resolvedVariant = entry.variant as Variant;

    const token = ++this.switchToken;
    this.cb.onLoading(`Loading ${run.runName} (${resolvedVariant})…`);

    let agent: { sim: MujocoSim; policy: Policy; meta: EnvMeta };
    try {
      agent = await this.loadAgent(entry);
    } catch (err) {
      if (token === this.switchToken) {
        console.error(err);
        this.cb.onError(`Failed to load ${run.runName}:<br><br><code>${String(err)}</code>`);
      }
      return;
    }
    if (token !== this.switchToken || this.disposed) { agent.sim.dispose(); return; }

    // Switching variants leaves learning mode; this policy becomes the base.
    this.learningActive = false;
    this.learningFrames = [];
    this.learningIndex = 0;
    this.learningToken++;
    this.framePolicyCache.clear();
    this.basePolicy = agent.policy;
    this.baseActDim = agent.meta.act_dim;

    if (this.loop) this.loop.setAgent(agent.sim, agent.policy);
    else {
      this.loop = new SimLoop(agent.sim, agent.policy, this.renderer);
      if (this.init.speed !== undefined) this.loop.speed = this.init.speed;
      this.loop.onStats = (s) => this.emitStats(s);
      this.loop.onControl = (obs) => {
        this.actorViz.setObs(obs);
        this.criticViz.setObs(obs);
      };
      this.loop.start();
    }

    // The fresh sim starts with the trained spawn range; re-apply the UI's
    // override so the slider setting survives variant/iteration switches.
    if (this.foodSpawnMax != null && agent.meta.food) {
      this.loop.setFoodSpawnMax(this.foodSpawnMax);
    }

    const labels = { obs: agent.meta.obs_labels, act: agent.meta.action_labels };
    this.actorViz.setNet(PolicyNet.actor(agent.meta), labels);
    this.criticViz.setNet(PolicyNet.critic(agent.meta), { obs: labels.obs, output: "value" });

    this.cb.onTask({
      food: agent.meta.food
        ? { spawnMin: agent.meta.food.spawn_min, spawnMax: agent.meta.food.spawn_max }
        : null,
      getup: !!agent.meta.getup,
    });
    this.cb.onVariant(resolvedVariant);
    this.emitLearning();

    const parity = await this.loop.verifyPolicyParity();
    if (token === this.switchToken) this.cb.onParity(parity);
    this.cb.onLoading(null);
  }

  // ---- learning mode (public API for the UI) ----

  private emitLearning(): void {
    this.cb.onLearning({
      available: !!this.run?.timeline,
      active: this.learningActive,
      frames: this.learningFrames,
      index: this.learningIndex,
    });
  }

  /** Engage/disengage the checkpoint scrubber for the current run. Turning it on
   * loads the timeline manifest and shows the first (untrained) checkpoint;
   * turning it off restores the run's own trained policy. */
  async setLearning(on: boolean): Promise<void> {
    if (on) {
      if (!this.run?.timeline || !this.loop) return;
      const token = ++this.learningToken;
      this.cb.onLoading("Loading checkpoints…");
      let timeline: Timeline;
      try {
        timeline = this.timeline ?? (await loadJson<Timeline>(this.run.timeline));
        this.timeline = timeline;
      } catch (err) {
        console.error(err);
        this.cb.onError(`Failed to load checkpoints:<br><br><code>${String(err)}</code>`);
        return;
      }
      if (token !== this.learningToken || this.disposed) return;
      this.learningActive = true;
      this.learningFrames = timeline.frames;
      this.learningIndex = 0;
      this.cb.onLoading(null);
      this.emitLearning();
      await this.showFrame(0);
    } else {
      this.learningActive = false;
      this.learningIndex = 0;
      this.learningToken++;
      if (this.loop && this.basePolicy) this.loop.setPolicy(this.basePolicy);
      this.emitLearning();
    }
  }

  /** Show the checkpoint at timeline index `i` (swaps the policy live). */
  async showFrame(i: number): Promise<void> {
    if (!this.learningActive || !this.loop) return;
    const frame = this.learningFrames[i];
    if (!frame) return;
    this.learningIndex = i;
    this.emitLearning();
    const token = ++this.learningToken;
    let policy = this.framePolicyCache.get(frame.onnx);
    if (!policy) {
      try {
        policy = await Policy.create(asset(frame.onnx), this.baseActDim);
      } catch (err) {
        console.error(err);
        return;
      }
      this.framePolicyCache.set(frame.onnx, policy);
    }
    if (token !== this.learningToken || !this.learningActive) return;
    this.loop.setPolicy(policy);
  }

  /** Throttle the per-frame stats to ~6/s so we don't thrash React state. */
  private emitStats(s: LoopStats): void {
    const now = performance.now();
    if (now - this.lastStatsEmit < 160) return;
    this.lastStatsEmit = now;
    this.cb.onLiveStats(s);
  }
}
