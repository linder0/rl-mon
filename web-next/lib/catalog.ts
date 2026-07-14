/**
 * Catalog data layer: loads the exported policy index and per-run stats from
 * the static JSON under public/policies/, and derives the project/iteration
 * hierarchy (project = environment, iteration = training run) used by the
 * routed pages. Pure data — no sim, renderer, or React in here.
 */

import { RUN_PALETTE } from "./palette";
import type {
  EnvGroup,
  PolicyIndex,
  RunAgg,
  RunEntry,
  StatsData,
} from "./types";

export type Variant = "final" | "best";

/** Env ids the combined forage+recover controller applies to, and the env
 * whose policy supplies the get-up skill. */
export const RECOVERY_ENVS = new Set(["AntFood-v5", "AntFood2Leg-v5"]);
export const RECOVERY_SOURCE_ENV = "AntGetUp-v5";

export const asset = (p: string): string => (p.startsWith("/") ? p : `/${p}`);

export async function loadJson<T>(path: string): Promise<T> {
  const res = await fetch(asset(path));
  if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`);
  return res.json() as Promise<T>;
}

// ---- index ----

let indexPromise: Promise<PolicyIndex> | null = null;

/** Load (and cache) the exported catalog. The data is static per page load, so
 * every page shares one in-flight/completed fetch. */
export function loadIndex(): Promise<PolicyIndex> {
  if (!indexPromise) {
    indexPromise = loadJson<PolicyIndex>("policies/index.json").catch((err) => {
      indexPromise = null; // allow retry after a failed fetch
      throw err;
    });
  }
  return indexPromise;
}

/** Group an env's exported (run, variant) entries into one RunAgg per run,
 * assigning each run a stable palette color by position. */
export function aggregateRuns(env: EnvGroup): RunAgg[] {
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
  return [...map.values()];
}

export function findEnvByLabel(index: PolicyIndex, label: string): EnvGroup | null {
  return index.envs?.find((e) => e.label === label) ?? null;
}

/** The get-up policy composed in combined forage+recover mode (best variant of
 * the recovery-source env), or null if none is exported. */
export function findRecoveryEntry(index: PolicyIndex): RunEntry | null {
  const src = index.envs?.find((e) => e.env_id === RECOVERY_SOURCE_ENV);
  const runs = src?.runs ?? [];
  return runs.find((r) => r.variant === "best") ?? runs[0] ?? null;
}

export function defaultVariant(agg: RunAgg): Variant {
  return agg.variants.best ? "best" : "final";
}

/** Runs sorted best-first (by best eval mean), the project pages' default order. */
export function sortRunsByEval(runs: RunAgg[]): RunAgg[] {
  return [...runs].sort(
    (a, b) =>
      (b.summary.best_eval_mean ?? -Infinity) - (a.summary.best_eval_mean ?? -Infinity),
  );
}

// ---- stats ----

const statsCache = new Map<string, Promise<StatsData>>();

export function loadStats(path: string): Promise<StatsData> {
  let p = statsCache.get(path);
  if (!p) {
    p = loadJson<StatsData>(path).catch((err) => {
      statsCache.delete(path);
      throw err;
    });
    statsCache.set(path, p);
  }
  return p;
}
