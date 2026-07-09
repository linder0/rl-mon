import "./style.css";
import { MujocoSim } from "./mujocoSim";
import { Policy } from "./policy";
import { Renderer } from "./renderer";
import { SimLoop } from "./loop";
import { PolicyNet } from "./net";
import { NetViz } from "./netviz";
import { UI } from "./ui";
import type { EnvGroup, EnvMeta, PolicyIndex, RunAgg, RunEntry, StatsData } from "./types";
import type { Series } from "./chart";

const BASE = import.meta.env.BASE_URL;
const asset = (p: string) => `${BASE}${p}`;

const PALETTE = [
  "#6ea8fe", "#b78bff", "#59d499", "#ffb454",
  "#ff6b6b", "#4dd0e1", "#f06292", "#aed581",
];

const ui = new UI();
const canvas = document.getElementById("scene") as HTMLCanvasElement;
const renderer = new Renderer(canvas);
const actorViz = new NetViz(document.getElementById("netcanvas") as HTMLCanvasElement);
const criticViz = new NetViz(document.getElementById("netcanvas2") as HTMLCanvasElement);
criticViz.onValue = (v) => ui.setValue(v);

let loop: SimLoop | null = null;
const statsCache = new Map<string, StatsData>();

// State.
let envs: EnvGroup[] = [];
const aggByEnv = new Map<string, RunAgg[]>();
let currentEnv = "";
let runsForEnv: RunAgg[] = [];
let byName = new Map<string, RunAgg>();
let live = { runName: "", variant: "best" as "final" | "best" };
let liveStats: StatsData | null = null;
const compared = new Set<string>();
const rememberedVariant = new Map<string, "final" | "best">();
let switchToken = 0;

async function loadJson<T>(path: string): Promise<T> {
  return (await fetch(asset(path))).json() as Promise<T>;
}

async function statsFor(agg: RunAgg): Promise<StatsData> {
  const cached = statsCache.get(agg.statsPath);
  if (cached) return cached;
  const stats = await loadJson<StatsData>(agg.statsPath);
  statsCache.set(agg.statsPath, stats);
  return stats;
}

async function loadAgent(
  entry: RunEntry,
): Promise<{ sim: MujocoSim; policy: Policy; meta: EnvMeta }> {
  const meta = await loadJson<EnvMeta>(entry.meta);
  const sim = await MujocoSim.create(meta, asset(entry.model_xml));
  const policy = await Policy.create(asset(entry.onnx), meta.act_dim);
  return { sim, policy, meta };
}

/** Group index entries by run_name, collecting final/best variants + a color. */
function aggregate(index: PolicyIndex): void {
  envs = index.envs ?? [];
  for (const env of envs) {
    const map = new Map<string, RunAgg>();
    for (const run of env.runs) {
      let agg = map.get(run.run_name);
      if (!agg) {
        agg = {
          runName: run.run_name,
          envId: env.env_id,
          color: PALETTE[map.size % PALETTE.length],
          seed: run.summary?.seed ?? null,
          summary: run.summary,
          statsPath: run.stats,
          variants: {},
        };
        map.set(run.run_name, agg);
      }
      agg.variants[run.variant as "final" | "best"] = run;
    }
    aggByEnv.set(env.env_id, [...map.values()]);
  }
}

function defaultVariant(agg: RunAgg): "final" | "best" {
  const remembered = rememberedVariant.get(agg.runName);
  if (remembered && agg.variants[remembered]) return remembered;
  return agg.variants.best ? "best" : "final";
}

async function buildCharts(): Promise<void> {
  if (!liveStats) return;
  const names = [live.runName, ...[...compared].filter((n) => n !== live.runName)];
  const evalOverlay: Series[] = [];
  const epLenOverlay: Series[] = [];

  for (const name of names) {
    const agg = byName.get(name);
    if (!agg) continue;
    const stats = await statsFor(agg);
    const isLive = name === live.runName;
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

  const d = liveStats.curves.diag ?? {};
  const diag = (key: string, color: string): Series[] =>
    d[key] && d[key].t.length ? [{ name: key, color, x: d[key].t, y: d[key].v }] : [];

  ui.renderCharts({
    evalOverlay,
    epLenOverlay,
    train: [{ name: "ep_rew_mean", color: "#b78bff", x: liveStats.curves.train.t, y: liveStats.curves.train.reward }],
    approxKl: diag("approx_kl", "#ffb454"),
    explainedVar: diag("explained_variance", "#59d499"),
    entropy: diag("entropy", "#4dd0e1"),
  });
}

async function switchToRun(runName: string, variant: "final" | "best"): Promise<void> {
  const agg = byName.get(runName);
  if (!agg) return;
  const entry = agg.variants[variant] ?? agg.variants.best ?? agg.variants.final;
  if (!entry) return;
  const resolvedVariant = entry.variant as "final" | "best";
  rememberedVariant.set(runName, resolvedVariant);

  const token = ++switchToken;
  ui.setLoading(`Loading ${runName} (${resolvedVariant})…`);

  let agent: { sim: MujocoSim; policy: Policy; meta: EnvMeta };
  let stats: StatsData;
  try {
    [agent, stats] = await Promise.all([loadAgent(entry), statsFor(agg)]);
  } catch (err) {
    if (token === switchToken) {
      console.error(err);
      ui.showError(`Failed to load ${runName}:<br><br><code>${String(err)}</code>`);
    }
    return;
  }
  if (token !== switchToken) { agent.sim.dispose(); return; }

  if (loop) loop.setAgent(agent.sim, agent.policy);
  else {
    loop = new SimLoop(agent.sim, agent.policy, renderer);
    loop.onStats = (s) => ui.renderLiveStats(s);
    loop.onControl = (obs) => {
      actorViz.setObs(obs);
      criticViz.setObs(obs);
    };
    loop.start();
  }

  const labels = { obs: agent.meta.obs_labels, act: agent.meta.action_labels };
  actorViz.setNet(PolicyNet.actor(agent.meta), labels);
  criticViz.setNet(PolicyNet.critic(agent.meta), { obs: labels.obs, output: "value" });

  live = { runName, variant: resolvedVariant };
  liveStats = stats;
  ui.renderSummary(agg, resolvedVariant);
  ui.renderConfig(stats);
  ui.renderRunTable(runsForEnv, live, compared);
  await buildCharts();

  const parity = await loop.verifyPolicyParity();
  if (token === switchToken) ui.setParity(parity);
  ui.setLoading(null);
}

async function selectEnv(envId: string): Promise<void> {
  currentEnv = envId;
  runsForEnv = aggByEnv.get(envId) ?? [];
  byName = new Map(runsForEnv.map((r) => [r.runName, r]));
  compared.clear();
  if (runsForEnv.length === 0) return;
  // Default to the run with the best eval (fall back to first).
  const best = [...runsForEnv].sort(
    (a, b) => (b.summary.best_eval_mean ?? -Infinity) - (a.summary.best_eval_mean ?? -Infinity),
  )[0];
  ui.renderRunTable(runsForEnv, { runName: best.runName, variant: defaultVariant(best) }, compared);
  await switchToRun(best.runName, defaultVariant(best));
}

async function main(): Promise<void> {
  let index: PolicyIndex;
  try {
    index = await loadJson<PolicyIndex>("policies/index.json");
  } catch {
    ui.showError(
      "Could not load <code>policies/index.json</code>. Export policies first:" +
        "<br><br><code>python export_onnx.py --all-envs</code>",
    );
    return;
  }

  aggregate(index);
  if (envs.length === 0 || envs.every((e) => e.runs.length === 0)) {
    ui.showError(
      "No runs found. Export them from the repo root:<br><br>" +
        "<code>python export_onnx.py --all-envs</code>",
    );
    return;
  }

  currentEnv = envs[0].env_id;
  ui.setEnvs(envs, currentEnv);

  ui.bind({
    onSelectEnv: (id) => void selectEnv(id),
    onPickRun: (runName) => void switchToRun(runName, defaultVariant(byName.get(runName)!)),
    onVariant: (runName, v) => void switchToRun(runName, v),
    onToggleCompare: (runName) => {
      if (compared.has(runName)) compared.delete(runName);
      else compared.add(runName);
      ui.renderRunTable(runsForEnv, live, compared);
      void buildCharts();
    },
    onTogglePlay: (playing) => { if (loop) loop.playing = playing; },
    onReset: () => loop?.reset(),
    onSpeed: (speed) => { if (loop) loop.speed = speed; },
    onFollow: (follow) => { renderer.follow = follow; },
    onToggleNet: (collapsed) => {
      actorViz.setCollapsed(collapsed);
      criticViz.setCollapsed(collapsed);
    },
  });

  try {
    ui.setLoading("Loading MuJoCo…");
    await selectEnv(currentEnv);
  } catch (err) {
    console.error(err);
    ui.showError(`Failed to start the simulation:<br><br><code>${String(err)}</code>`);
  }
}

void main();
