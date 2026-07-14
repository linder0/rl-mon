"use client";

import { UPlotChart } from "@/components/charts/UPlotChart";
import {
  OUTCOME_DEFS,
  diagDefs,
  trainDefs,
  seriesFor,
  type ChartDef,
  type ChartRun,
  type Series,
} from "@/lib/chartDefs";

export function ChartBlock({
  title,
  hint,
  children,
}: {
  title: string;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="space-y-1">
      <div className="text-label font-medium">
        {title} {hint && <span className="text-muted-foreground">{hint}</span>}
      </div>
      {children}
    </div>
  );
}

interface ChartGridProps {
  runs: ChartRun[];
  theme: string;
  height?: number;
  /** Grid classes for laying charts out (defaults to a single column). */
  className?: string;
  /** Restrict which chart groups render (default: all). */
  groups?: ("outcome" | "diagnostics")[];
  /** In overlay mode the run name is shown per series; single-run mode uses the
   * def's fixed metric color and hides bands-less legends. */
  singleRunLabel?: boolean;
}

/**
 * The batched chart surface: resolves the metric registry against the given
 * runs' stats and renders every chart that has data. One run = single mode
 * (metric colors, eval band); multiple runs = overlay mode (run colors).
 */
export function ChartGrid({
  runs,
  theme,
  height = 150,
  className = "flex flex-col gap-4",
  groups = ["outcome", "diagnostics"],
}: ChartGridProps) {
  if (runs.length === 0) return null;
  const overlay = runs.length > 1;
  const statsList = runs.map((r) => r.stats);

  const defs: ChartDef[] = [
    ...(groups.includes("outcome") ? OUTCOME_DEFS : []),
    ...(groups.includes("diagnostics")
      ? [...trainDefs(statsList), ...diagDefs(statsList)]
      : []),
  ];

  const charts = defs
    .map((def) => {
      const series = runs
        .map((run) =>
          seriesFor(def, run, {
            band: !overlay && def.source.kind === "eval",
            color: overlay ? run.color : def.color,
          }),
        )
        .filter((s): s is Series => s !== null)
        // Single-run mode: drop the legend by leaving the series unnamed.
        .map((s) => (overlay ? s : { ...s, name: "" }));
      return { def, series };
    })
    .filter(({ series }) => series.length > 0);

  if (charts.length === 0) return null;

  return (
    <div className={className}>
      {charts.map(({ def, series }) => (
        <ChartBlock key={def.id} title={def.title} hint={def.hint}>
          <UPlotChart series={series} height={height} theme={theme} />
        </ChartBlock>
      ))}
    </div>
  );
}
