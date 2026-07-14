/**
 * Metric registry for the batched training charts. Each ChartDef describes
 * where a metric lives in StatsData; ChartGrid maps defs onto uPlot charts for
 * one run (single mode, with bands) or many runs (overlay mode, run colors).
 * Diag metrics not listed here are discovered from the data and appended, so
 * newly exported diagnostics chart themselves without code changes.
 */

import { SERIES_COLORS, ACCENT } from "./palette";
import type { StatsData } from "./types";

/** A single line series with an optional symmetric band (e.g. mean +/- std). */
export interface Series {
  name: string;
  color: string;
  x: number[];
  y: number[];
  band?: { lo: number[]; hi: number[] };
}

export type ChartSource =
  | { kind: "eval" }
  | { kind: "epLen" }
  | { kind: "train" }
  | { kind: "diag"; key: string };

export interface ChartDef {
  id: string;
  title: string;
  hint?: string;
  source: ChartSource;
  /** Fixed color used in single-run mode (overlay mode uses run colors). */
  color: string;
}

export interface ChartRun {
  name: string;
  color: string;
  stats: StatsData;
}

/** Headline outcome charts, always listed first. */
export const OUTCOME_DEFS: ChartDef[] = [
  {
    id: "eval",
    title: "Eval reward",
    hint: "mean ± std",
    source: { kind: "eval" },
    color: SERIES_COLORS.reward,
  },
  {
    id: "epLen",
    title: "Episode length",
    hint: "survival",
    source: { kind: "epLen" },
    color: ACCENT.blue,
  },
];

/** Training diagnostics with curated titles/hints; discovered diag keys not
 * listed here fall back to the raw key. Order here is display order. */
const DIAG_META: Record<string, { title: string; hint?: string; color: string }> = {
  approx_kl: { title: "approx_kl", hint: "update size", color: SERIES_COLORS.approxKl },
  explained_variance: { title: "explained_variance", hint: "→ 1", color: SERIES_COLORS.explainedVar },
  entropy: { title: "policy entropy", hint: "exploration", color: SERIES_COLORS.entropy },
  action_std: { title: "action std", hint: "policy spread", color: ACCENT.pink },
  clip_fraction: { title: "clip fraction", hint: "clipped updates", color: ACCENT.lime },
  value_loss: { title: "value loss", color: ACCENT.red },
};

const TRAIN_DEF: ChartDef = {
  id: "train",
  title: "Episode reward",
  hint: "rollout",
  source: { kind: "train" },
  color: SERIES_COLORS.reward,
};

/** All diagnostic defs present in any of the given stats: curated ones first
 * (in DIAG_META order), then any unrecognized keys the export produced. */
export function diagDefs(statsList: StatsData[]): ChartDef[] {
  const present = new Set<string>();
  for (const s of statsList) {
    for (const key of Object.keys(s.curves.diag ?? {})) {
      if (s.curves.diag[key]?.t.length) present.add(key);
    }
  }
  const defs: ChartDef[] = [];
  for (const key of Object.keys(DIAG_META)) {
    if (present.has(key)) {
      defs.push({ id: key, source: { kind: "diag", key }, ...DIAG_META[key] });
      present.delete(key);
    }
  }
  for (const key of [...present].sort()) {
    defs.push({ id: key, title: key, source: { kind: "diag", key }, color: ACCENT.cyan });
  }
  return defs;
}

export function trainDefs(statsList: StatsData[]): ChartDef[] {
  const has = statsList.some((s) => s.curves.train.reward.length > 0);
  return has ? [TRAIN_DEF] : [];
}

/** Extract one run's series for a chart def; null when the run has no data
 * for that metric. Bands (eval ± std) only make sense in single-run mode. */
export function seriesFor(
  def: ChartDef,
  run: ChartRun,
  opts: { band?: boolean; color?: string } = {},
): Series | null {
  const { curves } = run.stats;
  const color = opts.color ?? run.color;
  switch (def.source.kind) {
    case "eval": {
      const ev = curves.eval;
      if (!ev.t.length) return null;
      const band =
        opts.band && ev.std && ev.std.length === ev.mean.length
          ? {
              lo: ev.mean.map((m, i) => m - ev.std[i]),
              hi: ev.mean.map((m, i) => m + ev.std[i]),
            }
          : undefined;
      return { name: run.name, color, x: ev.t, y: ev.mean, band };
    }
    case "epLen": {
      const ev = curves.eval;
      // Prefer eval survival, fall back to rollout episode length.
      if (ev.t.length && ev.ep_len && ev.ep_len.length === ev.t.length) {
        return { name: run.name, color, x: ev.t, y: ev.ep_len };
      }
      if (curves.train.ep_len?.length) {
        return { name: run.name, color, x: curves.train.t, y: curves.train.ep_len };
      }
      return null;
    }
    case "train": {
      if (!curves.train.reward.length) return null;
      return { name: run.name, color, x: curves.train.t, y: curves.train.reward };
    }
    case "diag": {
      const d = curves.diag?.[def.source.key];
      if (!d || !d.t.length) return null;
      return { name: run.name, color, x: d.t, y: d.v };
    }
  }
}
