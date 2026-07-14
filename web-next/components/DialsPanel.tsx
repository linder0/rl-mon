"use client";

import { memo } from "react";
import { Panel } from "@/components/ui/panel";
import { Button } from "@/components/ui/button";
import type { LoopStats } from "@/lib/loop";
import type { RunAgg } from "@/lib/types";
import type { Variant } from "@/lib/viewerApp";
import { PanelViewToggle, type PanelView } from "@/components/PanelViewToggle";
import { fmt } from "@/lib/format";

const EP_MAX = 1000; // gym TimeLimit for these tasks
const FPS_MAX = 120;
const SPEED_MAX = 3;

interface DialsPanelProps {
  liveStats: LoopStats | null;
  value: number | null;
  speed: number;
  summary: { run: RunAgg; variant: Variant } | null;
  collapsed: boolean;
  view: PanelView;
  onSetView: (v: PanelView) => void;
  onToggleCollapsed: () => void;
}

/** A 270-degree SVG gauge with a centered readout. `colorClass` sets the arc
 * color via currentColor (Tailwind text-* utility). */
function Gauge({
  label,
  display,
  frac,
  colorClass,
}: {
  label: string;
  display: string;
  frac: number;
  colorClass: string;
}) {
  const R = 30;
  const C = 2 * Math.PI * R;
  const sweep = 0.75; // 270° arc
  const used = C * sweep;
  const gap = C * (1 - sweep);
  const p = Math.max(0, Math.min(1, frac));
  return (
    <div className="flex flex-col items-center gap-1">
      <div className="relative">
        <svg viewBox="0 0 80 80" className="size-20 -rotate-[135deg]">
          <circle
            cx="40"
            cy="40"
            r={R}
            fill="none"
            className="text-muted"
            stroke="currentColor"
            strokeWidth="7"
            strokeLinecap="round"
            strokeDasharray={`${used} ${gap}`}
          />
          <circle
            cx="40"
            cy="40"
            r={R}
            fill="none"
            className={colorClass}
            stroke="currentColor"
            strokeWidth="7"
            strokeLinecap="round"
            strokeDasharray={`${used * p} ${C - used * p}`}
          />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="text-sm font-semibold tabular-nums">{display}</span>
        </div>
      </div>
      <span className="text-micro uppercase tracking-wide text-muted-foreground">
        {label}
      </span>
    </div>
  );
}

function BigReadout({
  label,
  value,
  accent,
}: {
  label: string;
  value: string;
  accent?: boolean;
}) {
  return (
    <div className="rounded-lg bg-muted/50 px-3 py-2">
      <div className="text-micro uppercase tracking-wide text-muted-foreground">
        {label}
      </div>
      <div
        className={`text-xl font-semibold tabular-nums ${accent ? "text-primary" : ""}`}
      >
        {value}
      </div>
    </div>
  );
}

function DialsPanelImpl(props: DialsPanelProps) {
  const { liveStats, value, speed, summary, collapsed } = props;
  const fps = liveStats?.fps ?? 0;
  const step = liveStats?.step ?? 0;
  const healthy = liveStats?.healthy ?? true;
  const distance = liveStats?.distance ?? 0;
  const episode = liveStats?.episode ?? 0;
  const pickups = liveStats?.pickups;

  return (
    <Panel size="md">
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
        <div className="flex flex-col gap-3 px-3 pb-4">
          <div className="truncate text-label text-muted-foreground">
            {summary ? `${summary.run.runName} · ${summary.variant}` : "no run"}
          </div>

          {/* Hero: the critic's live value estimate. */}
          <div className="rounded-lg bg-primary/10 px-3 py-3 text-center">
            <div className="text-micro uppercase tracking-wide text-muted-foreground">
              Critic value
            </div>
            <div className="text-3xl font-semibold tabular-nums text-primary">
              {value === null ? "—" : value.toFixed(1)}
            </div>
          </div>

          {/* Live gauges. */}
          <div className="grid grid-cols-3 gap-1">
            <Gauge
              label="episode"
              display={`${Math.min(step, EP_MAX)}`}
              frac={step / EP_MAX}
              colorClass="text-primary"
            />
            <Gauge
              label="fps"
              display={fps.toFixed(0)}
              frac={fps / FPS_MAX}
              colorClass="text-chart-2"
            />
            <Gauge
              label="speed"
              display={`${speed.toFixed(1)}×`}
              frac={speed / SPEED_MAX}
              colorClass="text-chart-4"
            />
          </div>

          {/* Readouts. */}
          <div className="grid grid-cols-2 gap-2">
            <BigReadout label="distance" value={`${distance.toFixed(1)} m`} accent />
            {pickups === undefined ? (
              <BigReadout label="episode #" value={`${episode}`} />
            ) : (
              <BigReadout label="food eaten" value={`${pickups}`} accent />
            )}
          </div>

          {/* Health pill. */}
          <div
            className={`flex items-center justify-center gap-2 rounded-lg py-2 text-sm font-medium ${
              healthy
                ? "bg-success/15 text-success"
                : "bg-destructive/15 text-destructive"
            }`}
          >
            <span
              className={`size-2 rounded-full ${
                healthy ? "bg-success" : "bg-destructive"
              }`}
            />
            {healthy ? "healthy" : "fallen"}
          </div>

          {summary && (
            <div className="grid grid-cols-2 gap-2">
              <BigReadout label="best eval" value={fmt(summary.run.summary.best_eval_mean)} />
              <BigReadout label="final eval" value={fmt(summary.run.summary.final_eval_mean)} />
            </div>
          )}
        </div>
      )}
    </Panel>
  );
}

export const DialsPanel = memo(DialsPanelImpl);
