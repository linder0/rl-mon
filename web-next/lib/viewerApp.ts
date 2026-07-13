import { MujocoSim } from "./mujocoSim";
import { Policy } from "./policy";
import { SimLoop, type LoopStats, type ParityResult } from "./loop";
import { PolicyNet } from "./net";
import { NetVizController } from "./netvizController";
import { WebGPURenderer, type Backend } from "@/components/scene/webgpuRenderer";
import { RUN_PALETTE, SERIES_COLORS } from "./palette";
import type { Theme } from "./theme";
import type { Series, ChartPayload } from "./chartTypes";
import type {
  EnvGroup,
  EnvMeta,
  PolicyIndex,
  RunAgg,
  RunEntry,
  StatsData,
  Timeline,
  TimelineFrame,
} from "./types";

export type Variant = "final" | "best";
export interface LiveSel {
  runName: string;
  variant: Variant;
}

/** "Learning mode" state pushed to the UI: whether the live run has a
 * checkpoint timeline, whether it's engaged, the frames, and the current one. */
export interface LearningState {
  available: boolean;
  active: boolean;
  frames: TimelineFrame[];
  index: number;
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
  bloom?: number;
  /** Per-theme scene color overrides; the entry for the active theme is applied
   * on startup and whenever the theme changes. */
  colorOverrides?: Partial<Record<Theme, SceneColors>>;
}

export interface ViewerCallbacks {
  onLoading: (text: string | null) => void;
  onError: (message: string) => void;
  onEnvs: (envs: EnvGroup[], currentEnvId: string) => void;
  onRunTable: (runs: RunAgg[], live: LiveSel, compared: string[]) => void;
  onSummary: (run: RunAgg, variant: Variant) => void;
  onConfig: (stats: StatsData) => void;
  onCharts: (payload: ChartPayload) => void;
  onLiveStats: (s: LoopStats) => void;
  onParity: (r: ParityResult) => void;
  onValue: (v: number) => void;
  onBackend: (b: Backend) => void;
  onSceneColors: (c: SceneColors) => void;
  onLearning: (state: LearningState) => void;
  /** Whether combined forage+recover mode is available for the current env (a
   * get-up policy exists and the env is a foraging task). */
  onRecoveryAvailable: (available: boolean) => void;
}

/** Env ids the combined forage+recover controller applies to, and the env
 * whose policy supplies the get-up skill. */
const RECOVERY_ENVS = new Set(["AntFood-v5", "AntFood2Leg-v5"]);
const RECOVERY_SOURCE_ENV = "AntGetUp-v5";

const asset = (p: string): string => (p.startsWith("/") ? p : `/${p}`);

async function loadJson<T>(path: string): Promise<T> {
  return (await fetch(asset(path))).json() as Promise<T>;
}

/** Framework-agnostic orchestrator for the viewer: owns the renderer, sim loop,
 * and network visualizations, and emits display data through callbacks so a
 * React (or any) shell can render the dashboard. Ported from the original
 * main.ts + UI wiring. */
export class ViewerApp {
  private readonly renderer: WebGPURenderer;
  private readonly actorViz: NetVizController;
  private readonly criticViz: NetVizController;
  private readonly cb: ViewerCallbacks;
  private readonly init: ViewerInit;
  private theme: Theme;
  private readonly colorOverrides: Partial<Record<Theme, SceneColors>>;

  private loop: SimLoop | null = null;
  private readonly statsCache = new Map<string, StatsData>();

  private envs: EnvGroup[] = [];
  private readonly aggByEnv = new Map<string, RunAgg[]>();
  private currentEnv = "";
  private runsForEnv: RunAgg[] = [];
  private byName = new Map<string, RunAgg>();
  private live: LiveSel = { runName: "", variant: "best" };
  private liveStats: StatsData | null = null;
  private readonly compared = new Set<string>();
  private readonly rememberedVariant = new Map<string, Variant>();
  private switchToken = 0;
  private disposed = false;

  private lastStatsEmit = 0;

  // ---- combined forage+recover mode ----
  private recoveryEntry: RunEntry | null = null; // the get-up policy to compose
  private recoveryOn = false;

  // ---- learning mode (checkpoint timeline) ----
  private basePolicy: Policy | null = null; // the loaded run's own policy
  private baseActDim = 0;
  private learningActive = false;
  private learningFrames: TimelineFrame[] = [];
  private learningIndex = 0;
  private learningToken = 0;
  private readonly timelineCache = new Map<string, Timeline>();
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

  async start(): Promise<void> {
    try {
      this.cb.onLoading("Initializing renderer…");
      const backend = await this.renderer.init();
      this.cb.onBackend(backend);
      if (this.init.gridOn !== undefined) this.renderer.setGridVisible(this.init.gridOn);
      if (this.init.bloom !== undefined) this.renderer.setBloomStrength(this.init.bloom);
      this.applyColorOverride();
    } catch (err) {
      console.error(err);
      this.cb.onError(`Failed to initialize the GPU renderer:<br><br><code>${String(err)}</code>`);
      return;
    }

    let index: PolicyIndex;
    try {
      index = await loadJson<PolicyIndex>("policies/index.json");
    } catch {
      this.cb.onError(
        "Could not load <code>policies/index.json</code>. Export policies first:" +
          "<br><br><code>python export_onnx.py --all-envs</code>",
      );
      return;
    }

    this.aggregate(index);
    if (this.envs.length === 0 || this.envs.every((e) => e.runs.length === 0)) {
      this.cb.onError(
        "No runs found. Export them from the repo root:<br><br>" +
          "<code>python export_onnx.py --all-envs</code>",
      );
      return;
    }

    this.currentEnv = this.envs[0].env_id;
    this.cb.onEnvs(this.envs, this.currentEnv);

    try {
      this.cb.onLoading("Loading MuJoCo…");
      await this.selectEnv(this.currentEnv);
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
  setAgentColor(hex: number | null): void {
    this.renderer.setAgentColor(hex);
  }
  setBloomStrength(v: number): void {
    this.renderer.setBloomStrength(v);
  }
  sceneColors(): { bg: number; ground: number; grid: number; agent: number } {
    return this.renderer.sceneColors();
  }
  selectEnvId(envId: string): void {
    void this.selectEnv(envId);
  }
  pickRun(runName: string): void {
    void this.switchToRun(runName, this.defaultVariant(this.byName.get(runName)!));
  }
  setVariant(runName: string, variant: Variant): void {
    void this.switchToRun(runName, variant);
  }
  toggleCompare(runName: string): void {
    if (this.compared.has(runName)) this.compared.delete(runName);
    else this.compared.add(runName);
    this.cb.onRunTable(this.runsForEnv, this.live, [...this.compared]);
    void this.buildCharts();
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
      // Guard against an env switch that turned recovery off mid-load.
      if (this.recoveryOn && this.loop) {
        this.loop.recovery = { policy, obsDim: meta.obs_dim };
      }
    } catch (err) {
      console.error(err);
      this.cb.onError(`Failed to load the get-up policy:<br><br><code>${String(err)}</code>`);
    }
  }

  /** Shove the ant over (to demo the recovery policy in combined mode). */
  knockOver(): void {
    this.loop?.knockOver();
  }

  // ---- internals (ported from main.ts) ----

  private async loadAgent(
    entry: RunEntry,
  ): Promise<{ sim: MujocoSim; policy: Policy; meta: EnvMeta }> {
    const meta = await loadJson<EnvMeta>(entry.meta);
    const sim = await MujocoSim.create(meta, asset(entry.model_xml));
    const policy = await Policy.create(asset(entry.onnx), meta.act_dim);
    return { sim, policy, meta };
  }

  private aggregate(index: PolicyIndex): void {
    this.envs = index.envs ?? [];
    for (const env of this.envs) {
      const map = new Map<string, RunAgg>();
      for (const run of env.runs) {
        let agg = map.get(run.run_name);
        if (!agg) {
          agg = {
            runName: run.run_name,
            envId: env.env_id,
            color: RUN_PALETTE[map.size % RUN_PALETTE.length],
            seed: run.summary?.seed ?? null,
            summary: run.summary,
            statsPath: run.stats,
            variants: {},
          };
          map.set(run.run_name, agg);
        }
        agg.variants[run.variant as Variant] = run;
        if (run.timeline) agg.timeline = run.timeline;
      }
      this.aggByEnv.set(env.env_id, [...map.values()]);
    }

    // Find the get-up policy (best variant) to compose in combined mode.
    const src = this.envs.find((e) => e.env_id === RECOVERY_SOURCE_ENV);
    const runs = src?.runs ?? [];
    this.recoveryEntry =
      runs.find((r) => r.variant === "best") ?? runs[0] ?? null;
  }

  private recoveryApplies(envId: string): boolean {
    return this.recoveryEntry != null && RECOVERY_ENVS.has(envId);
  }

  private defaultVariant(agg: RunAgg): Variant {
    const remembered = this.rememberedVariant.get(agg.runName);
    if (remembered && agg.variants[remembered]) return remembered;
    return agg.variants.best ? "best" : "final";
  }

  private async statsFor(agg: RunAgg): Promise<StatsData> {
    const cached = this.statsCache.get(agg.statsPath);
    if (cached) return cached;
    const stats = await loadJson<StatsData>(agg.statsPath);
    this.statsCache.set(agg.statsPath, stats);
    return stats;
  }

  private async buildCharts(): Promise<void> {
    if (!this.liveStats) return;
    const names = [this.live.runName, ...[...this.compared].filter((n) => n !== this.live.runName)];
    const evalOverlay: Series[] = [];
    const epLenOverlay: Series[] = [];

    for (const name of names) {
      const agg = this.byName.get(name);
      if (!agg) continue;
      const stats = await this.statsFor(agg);
      const isLive = name === this.live.runName;
      const ev = stats.curves.eval;
      if (ev.t.length) {
        const band =
          isLive && ev.std && ev.std.length === ev.mean.length
            ? { lo: ev.mean.map((m, i) => m - ev.std[i]), hi: ev.mean.map((m, i) => m + ev.std[i]) }
            : undefined;
        evalOverlay.push({ name, color: agg.color, x: ev.t, y: ev.mean, band });
      }
      // Episode length: prefer eval survival, fall back to rollout episode length.
      if (ev.t.length && ev.ep_len && ev.ep_len.length === ev.t.length) {
        epLenOverlay.push({ name, color: agg.color, x: ev.t, y: ev.ep_len });
      } else if (stats.curves.train.ep_len && stats.curves.train.ep_len.length) {
        epLenOverlay.push({ name, color: agg.color, x: stats.curves.train.t, y: stats.curves.train.ep_len });
      }
    }

    const d = this.liveStats.curves.diag ?? {};
    const diag = (key: string, color: string): Series[] =>
      d[key] && d[key].t.length ? [{ name: key, color, x: d[key].t, y: d[key].v }] : [];

    this.cb.onCharts({
      evalOverlay,
      epLenOverlay,
      train: [{
        name: "ep_rew_mean", color: SERIES_COLORS.reward,
        x: this.liveStats.curves.train.t, y: this.liveStats.curves.train.reward,
      }],
      approxKl: diag("approx_kl", SERIES_COLORS.approxKl),
      explainedVar: diag("explained_variance", SERIES_COLORS.explainedVar),
      entropy: diag("entropy", SERIES_COLORS.entropy),
    });
  }

  private async selectEnv(envId: string): Promise<void> {
    this.currentEnv = envId;
    this.runsForEnv = this.aggByEnv.get(envId) ?? [];
    this.byName = new Map(this.runsForEnv.map((r) => [r.runName, r]));
    this.compared.clear();
    // Combined mode is per-env: clear it on switch and tell the UI whether it's
    // offered here.
    this.recoveryOn = false;
    if (this.loop) this.loop.recovery = null;
    this.cb.onRecoveryAvailable(this.recoveryApplies(envId));
    if (this.runsForEnv.length === 0) return;
    // Default to the run with the best eval (fall back to first).
    const best = [...this.runsForEnv].sort(
      (a, b) => (b.summary.best_eval_mean ?? -Infinity) - (a.summary.best_eval_mean ?? -Infinity),
    )[0];
    this.cb.onRunTable(this.runsForEnv, { runName: best.runName, variant: this.defaultVariant(best) }, [...this.compared]);
    await this.switchToRun(best.runName, this.defaultVariant(best));
  }

  private async switchToRun(runName: string, variant: Variant): Promise<void> {
    const agg = this.byName.get(runName);
    if (!agg) return;
    const entry = agg.variants[variant] ?? agg.variants.best ?? agg.variants.final;
    if (!entry) return;
    const resolvedVariant = entry.variant as Variant;
    this.rememberedVariant.set(runName, resolvedVariant);

    const token = ++this.switchToken;
    this.cb.onLoading(`Loading ${runName} (${resolvedVariant})…`);

    let agent: { sim: MujocoSim; policy: Policy; meta: EnvMeta };
    let stats: StatsData;
    try {
      [agent, stats] = await Promise.all([this.loadAgent(entry), this.statsFor(agg)]);
    } catch (err) {
      if (token === this.switchToken) {
        console.error(err);
        this.cb.onError(`Failed to load ${runName}:<br><br><code>${String(err)}</code>`);
      }
      return;
    }
    if (token !== this.switchToken || this.disposed) { agent.sim.dispose(); return; }

    // Switching runs leaves learning mode; this run's own policy is the base.
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

    const labels = { obs: agent.meta.obs_labels, act: agent.meta.action_labels };
    this.actorViz.setNet(PolicyNet.actor(agent.meta), labels);
    this.criticViz.setNet(PolicyNet.critic(agent.meta), { obs: labels.obs, output: "value" });

    this.live = { runName, variant: resolvedVariant };
    this.liveStats = stats;
    this.cb.onSummary(agg, resolvedVariant);
    this.cb.onConfig(stats);
    this.cb.onRunTable(this.runsForEnv, this.live, [...this.compared]);
    await this.buildCharts();

    this.emitLearning();

    const parity = await this.loop.verifyPolicyParity();
    if (token === this.switchToken) this.cb.onParity(parity);
    this.cb.onLoading(null);
  }

  // ---- learning mode (public API for the UI) ----

  private emitLearning(): void {
    const agg = this.byName.get(this.live.runName);
    this.cb.onLearning({
      available: !!agg?.timeline,
      active: this.learningActive,
      frames: this.learningFrames,
      index: this.learningIndex,
    });
  }

  /** Engage/disengage the checkpoint scrubber for the current run. Turning it on
   * loads the timeline manifest and shows the first (untrained) checkpoint;
   * turning it off restores the run's own trained policy. */
  async setLearning(on: boolean): Promise<void> {
    const agg = this.byName.get(this.live.runName);
    if (on) {
      if (!agg?.timeline || !this.loop) return;
      const token = ++this.learningToken;
      this.cb.onLoading("Loading checkpoints…");
      let timeline: Timeline;
      try {
        timeline = await this.loadTimeline(agg.timeline);
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

  private async loadTimeline(path: string): Promise<Timeline> {
    const cached = this.timelineCache.get(path);
    if (cached) return cached;
    const tl = await loadJson<Timeline>(path);
    this.timelineCache.set(path, tl);
    return tl;
  }

  /** Throttle the per-frame stats to ~6/s so we don't thrash React state. */
  private emitStats(s: LoopStats): void {
    const now = performance.now();
    if (now - this.lastStatsEmit < 160) return;
    this.lastStatsEmit = now;
    this.cb.onLiveStats(s);
  }
}
