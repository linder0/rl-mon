"use client";

import { memo } from "react";
import { UPlotChart } from "@/components/charts/UPlotChart";
import { Panel } from "@/components/ui/panel";
import { Button } from "@/components/ui/button";
import { SectionLabel } from "@/components/ui/section-label";
import { Checkbox } from "@/components/ui/checkbox";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import type { ChartPayload, Series } from "@/lib/chartTypes";
import type { LiveSel, Variant } from "@/lib/viewerApp";
import type { RunAgg, StatsData } from "@/lib/types";
import { PanelViewToggle, type PanelView } from "@/components/PanelViewToggle";

function fmt(v: number | null | undefined, digits = 0): string {
  if (v === null || v === undefined) return "—";
  return v.toLocaleString(undefined, { maximumFractionDigits: digits });
}

interface StatsPanelProps {
  runs: RunAgg[];
  live: LiveSel;
  compared: string[];
  summary: { run: RunAgg; variant: Variant } | null;
  config: StatsData | null;
  charts: ChartPayload | null;
  collapsed: boolean;
  theme: string;
  view: PanelView;
  onSetView: (v: PanelView) => void;
  onToggleCollapsed: () => void;
  onPickRun: (name: string) => void;
  onVariant: (name: string, v: Variant) => void;
  onToggleCompare: (name: string) => void;
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

function ChartBlock({
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

function StatsPanelImpl(props: StatsPanelProps) {
  const { runs, live, compared, summary, config, charts, collapsed, theme } = props;
  const comparedSet = new Set(compared);

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
            {/* Run comparison table */}
            <div className="space-y-1">
              <SectionLabel>Runs</SectionLabel>
              <Table className="text-label">
                <TableHeader>
                  <TableRow className="border-border/60 hover:bg-transparent">
                    <TableHead className="h-6 px-1 text-nano uppercase">Run</TableHead>
                    <TableHead className="h-6 px-1 text-center text-nano uppercase">
                      Best
                    </TableHead>
                    <TableHead className="h-6 px-1 text-center text-nano uppercase">
                      Final
                    </TableHead>
                    <TableHead className="h-6 px-1 text-center text-nano uppercase">
                      Var
                    </TableHead>
                    <TableHead className="h-6 w-6 px-1" />
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {runs.map((run) => {
                    const isLive = run.runName === live.runName;
                    const hasFinal = !!run.variants.final;
                    const hasBest = !!run.variants.best;
                    const checked = comparedSet.has(run.runName) || isLive;
                    return (
                      <TableRow
                        key={run.runName}
                        className={`border-0 ${isLive ? "bg-primary/10" : ""}`}
                      >
                        <TableCell className="px-1 py-1">
                          <button
                            className="flex items-center gap-1.5 text-left hover:text-primary"
                            onClick={() => props.onPickRun(run.runName)}
                            title={run.runName}
                          >
                            <span
                              className="size-2 shrink-0 rounded-[3px]"
                              style={{ background: run.color }}
                            />
                            <span className="flex flex-col leading-tight">
                              <span className="max-w-[110px] truncate">{run.runName}</span>
                              <span className="text-nano text-muted-foreground">
                                seed {run.seed ?? "—"}
                              </span>
                            </span>
                          </button>
                        </TableCell>
                        <TableCell className="px-1 py-1 text-center tabular-nums">
                          {fmt(run.summary.best_eval_mean, 0)}
                        </TableCell>
                        <TableCell className="px-1 py-1 text-center tabular-nums">
                          {fmt(run.summary.final_eval_mean, 0)}
                        </TableCell>
                        <TableCell className="px-1 py-1">
                          <div className="flex justify-center gap-0.5">
                            <Button
                              size="sm"
                              variant={
                                isLive && live.variant === "final" ? "default" : "outline"
                              }
                              disabled={!hasFinal}
                              className="h-5 w-5 p-0 text-nano"
                              onClick={() => props.onVariant(run.runName, "final")}
                            >
                              F
                            </Button>
                            <Button
                              size="sm"
                              variant={
                                isLive && live.variant === "best" ? "default" : "outline"
                              }
                              disabled={!hasBest}
                              className="h-5 w-5 p-0 text-nano"
                              onClick={() => props.onVariant(run.runName, "best")}
                            >
                              B
                            </Button>
                          </div>
                        </TableCell>
                        <TableCell className="px-1 py-1">
                          <Checkbox
                            checked={checked}
                            disabled={isLive}
                            className="size-3.5"
                            onCheckedChange={() => props.onToggleCompare(run.runName)}
                            title="overlay eval curve"
                          />
                        </TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </div>

            {summary && (
              <div className="grid grid-cols-3 gap-1.5">
                <StatTile label="best eval" value={fmt(summary.run.summary.best_eval_mean)} accent />
                <StatTile label="final eval" value={fmt(summary.run.summary.final_eval_mean)} />
                <StatTile label="ep len" value={fmt(summary.run.summary.eval_ep_len_final)} />
                <StatTile label="timesteps" value={fmt(summary.run.summary.timesteps)} />
                <StatTile
                  label="seed"
                  value={
                    summary.run.summary.seed === null || summary.run.summary.seed === undefined
                      ? "—"
                      : String(summary.run.summary.seed)
                  }
                />
              </div>
            )}

            <Separator />
            <SectionLabel>Outcome</SectionLabel>
            <ChartBlock title="Eval reward" hint="mean ± std">
              <UPlotChart series={charts?.evalOverlay ?? []} height={150} theme={theme} />
            </ChartBlock>
            <ChartBlock title="Episode length" hint="survival">
              <UPlotChart series={charts?.epLenOverlay ?? []} height={130} theme={theme} />
            </ChartBlock>

            {(() => {
              // Only render diagnostics that this run actually logged — SB3 and
              // Brax emit different subsets, so empty "no data" boxes would just
              // look broken. Hide the whole section if nothing is available.
              const hasData = (s?: Series[]) => !!s?.some((x) => x.x.length > 0);
              const blocks: React.ReactNode[] = [];
              if (hasData(charts?.train))
                blocks.push(
                  <ChartBlock key="train" title="Episode reward" hint="rollout">
                    <UPlotChart series={charts!.train} height={120} theme={theme} />
                  </ChartBlock>,
                );
              if (hasData(charts?.approxKl))
                blocks.push(
                  <ChartBlock key="kl" title="approx_kl" hint="update size">
                    <UPlotChart series={charts!.approxKl} height={110} theme={theme} />
                  </ChartBlock>,
                );
              if (hasData(charts?.explainedVar))
                blocks.push(
                  <ChartBlock key="ev" title="explained_variance" hint="→ 1">
                    <UPlotChart series={charts!.explainedVar} height={110} theme={theme} />
                  </ChartBlock>,
                );
              if (hasData(charts?.entropy))
                blocks.push(
                  <ChartBlock key="ent" title="policy entropy" hint="exploration">
                    <UPlotChart series={charts!.entropy} height={110} theme={theme} />
                  </ChartBlock>,
                );
              if (blocks.length === 0) return null;
              return (
                <>
                  <Separator />
                  <SectionLabel>Training diagnostics</SectionLabel>
                  {blocks}
                </>
              );
            })()}

            {config && (
              <>
                <Separator />
                <SectionLabel>Config</SectionLabel>
                <div className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-label tabular-nums">
                  {configRows(config).map(([k, v]) => (
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
