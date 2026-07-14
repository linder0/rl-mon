"use client";

import { memo } from "react";
import { Panel } from "@/components/ui/panel";
import { Button } from "@/components/ui/button";
import { SectionLabel } from "@/components/ui/section-label";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { ChartGrid } from "@/components/charts/ChartGrid";
import type { ChartRun } from "@/lib/chartDefs";
import type { Variant } from "@/lib/viewerApp";
import type { RunAgg, StatsData } from "@/lib/types";
import { PanelViewToggle, type PanelView } from "@/components/PanelViewToggle";
import { fmt } from "@/lib/format";

interface StatsPanelProps {
  agg: RunAgg | null;
  variant: Variant;
  stats: StatsData | null;
  /** Every iteration in this project (with stats where loaded), for the
   * batched outcome overlays. Colors match the iteration dropdown order. */
  overlayRuns: ChartRun[];
  collapsed: boolean;
  theme: string;
  view: PanelView;
  onSetView: (v: PanelView) => void;
  onToggleCollapsed: () => void;
  onVariant: (v: Variant) => void;
}

/** Build the config table. Handles both trainers — SB3 (train.py) and Brax/MJX
 * (train_mjx.py) store different hyperparameter keys — and drops any row whose
 * value is missing so the panel never shows a column of dashes. */
function configRows(stats: StatsData): [string, string][] {
  const hp = (stats.config.hyperparameters ?? {}) as Record<string, unknown>;
  const args = (stats.config.args ?? {}) as Record<string, unknown>;
  const isBrax =
    hp.discounting !== undefined ||
    hp.num_envs !== undefined ||
    hp.clipping_epsilon !== undefined;
  const netArch =
    (hp.net_arch as unknown) ??
    (hp.policy_kwargs as { net_arch?: unknown } | undefined)?.net_arch;

  const rows: [string, unknown][] = [
    ["env", stats.config.env_id],
    ["trainer", isBrax ? "PPO · Brax/MJX" : "PPO · SB3"],
    ["learning rate", hp.learning_rate],
  ];
  if (isBrax) {
    rows.push(
      ["lr schedule", hp.lr_schedule],
      ["num_envs", hp.num_envs],
      ["batch size", hp.batch_size],
      ["minibatches", hp.num_minibatches],
      ["unroll length", hp.unroll_length],
      ["epochs/batch", hp.num_updates_per_batch],
      ["discounting γ", hp.discounting],
      ["gae λ", hp.gae_lambda],
      ["clip ε", hp.clipping_epsilon],
      ["entropy cost", hp.entropy_cost],
      ["reward scaling", hp.reward_scaling],
    );
  } else {
    rows.push(
      ["n_steps", hp.n_steps],
      ["batch size", hp.batch_size],
      ["n_epochs", hp.n_epochs],
      ["gamma", hp.gamma],
      ["gae_lambda", hp.gae_lambda],
      ["clip range", hp.clip_range],
      ["ent_coef", hp.ent_coef],
      ["n_envs", args.n_envs],
    );
  }
  rows.push(
    ["net arch", Array.isArray(netArch) ? `[${netArch.join(", ")}]` : undefined],
    ["device", stats.config.device],
  );

  return rows
    .filter(([, v]) => v !== undefined && v !== null && v !== "")
    .map(([k, v]) => [k, String(v)]);
}

function StatTile({
  label,
  value,
  accent,
}: {
  label: string;
  value: string;
  accent?: boolean;
}) {
  return (
    <div className="rounded-md bg-muted/50 px-2 py-1.5 text-center">
      <div
        className={`text-sm font-semibold tabular-nums ${accent ? "text-primary" : ""}`}
      >
        {value}
      </div>
      <div className="mt-0.5 text-nano uppercase tracking-wide text-muted-foreground">
        {label}
      </div>
    </div>
  );
}

function StatsPanelImpl(props: StatsPanelProps) {
  const { agg, variant, stats, overlayRuns, collapsed, theme } = props;
  const hasFinal = !!agg?.variants.final;
  const hasBest = !!agg?.variants.best;
  // Outcome charts overlay every iteration when the project has more than one;
  // otherwise the selected run renders alone (with its eval band).
  const outcomeRuns: ChartRun[] =
    overlayRuns.length > 1
      ? overlayRuns
      : agg && stats
        ? [{ name: agg.runName, color: agg.color, stats }]
        : [];

  return (
    <Panel size="lg" className={collapsed ? "" : "h-full"}>
      <Panel.Header>
        <PanelViewToggle view={props.view} onSetView={props.onSetView} />
        <Button
          variant="ghost"
          size="sm"
          className="h-7 shrink-0 px-2 text-xs"
          onClick={props.onToggleCollapsed}
        >
          {collapsed ? "Show" : "Hide"}
        </Button>
      </Panel.Header>

      {!collapsed && (
        <ScrollArea className="min-h-0 flex-1">
          <div className="flex flex-col gap-4 px-3 pb-3">
            {/* Checkpoint variant loaded into the sim. */}
            {agg && (
              <div className="flex items-center justify-between gap-2">
                <SectionLabel className="mb-0">Checkpoint</SectionLabel>
                <div className="flex gap-1">
                  <Button
                    size="sm"
                    variant={variant === "best" ? "default" : "outline"}
                    disabled={!hasBest}
                    className="h-6 px-2 text-xs"
                    onClick={() => props.onVariant("best")}
                  >
                    Best
                  </Button>
                  <Button
                    size="sm"
                    variant={variant === "final" ? "default" : "outline"}
                    disabled={!hasFinal}
                    className="h-6 px-2 text-xs"
                    onClick={() => props.onVariant("final")}
                  >
                    Final
                  </Button>
                </div>
              </div>
            )}

            {agg && (
              <div className="grid grid-cols-3 gap-1.5">
                <StatTile label="best eval" value={fmt(agg.summary.best_eval_mean)} accent />
                <StatTile label="final eval" value={fmt(agg.summary.final_eval_mean)} />
                <StatTile label="ep len" value={fmt(agg.summary.eval_ep_len_final)} />
                <StatTile label="timesteps" value={fmt(agg.summary.timesteps)} />
                <StatTile label="seed" value={fmt(agg.summary.seed)} />
              </div>
            )}

            {outcomeRuns.length > 0 && (
              <>
                <Separator />
                <SectionLabel>
                  Outcome{overlayRuns.length > 1 ? " — all iterations" : ""}
                </SectionLabel>
                <ChartGrid
                  runs={outcomeRuns}
                  theme={theme}
                  height={140}
                  groups={["outcome"]}
                />
              </>
            )}

            {agg && stats && (
              <>
                <Separator />
                <SectionLabel>Training diagnostics</SectionLabel>
                <ChartGrid
                  runs={[{ name: agg.runName, color: agg.color, stats }]}
                  theme={theme}
                  height={120}
                  groups={["diagnostics"]}
                />
              </>
            )}

            {stats && (
              <>
                <Separator />
                <SectionLabel>Config</SectionLabel>
                <div className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-label tabular-nums">
                  {configRows(stats).map(([k, v]) => (
                    <div key={k} className="contents">
                      <span className="text-muted-foreground">{k}</span>
                      <span className="text-right font-medium">{v}</span>
                    </div>
                  ))}
                </div>
              </>
            )}
          </div>
        </ScrollArea>
      )}
    </Panel>
  );
}

export const StatsPanel = memo(StatsPanelImpl);
