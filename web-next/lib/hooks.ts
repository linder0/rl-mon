"use client";

import { useEffect, useState } from "react";
import { aggregateRuns, findEnvByLabel, loadIndex, loadStats } from "./catalog";
import type { EnvGroup, PolicyIndex, RunAgg, StatsData } from "./types";

export interface CatalogState {
  index: PolicyIndex | null;
  error: string | null;
}

/** The exported catalog (policies/index.json), fetched once per page load. */
export function useCatalog(): CatalogState {
  const [state, setState] = useState<CatalogState>({ index: null, error: null });

  useEffect(() => {
    let alive = true;
    loadIndex()
      .then((index) => alive && setState({ index, error: null }))
      .catch(() =>
        alive &&
        setState({
          index: null,
          error:
            "Could not load policies/index.json. Export policies first: python export_onnx.py --all-envs",
        }),
      );
    return () => {
      alive = false;
    };
  }, []);

  return state;
}

export interface ProjectState {
  env: EnvGroup | null;
  runs: RunAgg[];
  /** True once the catalog has loaded (env === null then means "not found"). */
  ready: boolean;
  error: string | null;
}

/** One project (environment) and its aggregated iterations (runs). */
export function useProject(label: string): ProjectState {
  const { index, error } = useCatalog();
  if (!index) return { env: null, runs: [], ready: false, error };
  const env = findEnvByLabel(index, label);
  return { env, runs: env ? aggregateRuns(env) : [], ready: true, error: null };
}

/** Stats for a set of runs, keyed by run name. Grows as fetches resolve. */
export function useRunStats(runs: RunAgg[]): Map<string, StatsData> {
  const [stats, setStats] = useState<Map<string, StatsData>>(new Map());
  // Refetch only when the actual set of stats files changes, not on every
  // parent render that rebuilds the runs array.
  const key = runs.map((r) => r.statsPath).join("|");

  useEffect(() => {
    let alive = true;
    setStats(new Map());
    for (const run of runs) {
      loadStats(run.statsPath)
        .then((s) => {
          if (!alive) return;
          setStats((prev) => new Map(prev).set(run.runName, s));
        })
        .catch((err) => console.error(`stats for ${run.runName}:`, err));
    }
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);

  return stats;
}
