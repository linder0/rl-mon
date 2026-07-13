"use client";

import { memo } from "react";
import { Moon, Sun } from "lucide-react";
import { Panel } from "@/components/ui/panel";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Separator } from "@/components/ui/separator";
import { PanelViewToggle, type PanelView } from "@/components/PanelViewToggle";

export interface SceneCfg {
  bg: number;
  ground: number;
  grid: number;
  agent: number;
  gridOn: boolean;
  bloom: number;
}

interface ScenePanelProps {
  cfg: SceneCfg;
  theme: string;
  collapsed: boolean;
  view: PanelView;
  onSetView: (v: PanelView) => void;
  onToggleCollapsed: () => void;
  onSetTheme: (t: "light" | "dark") => void;
  onBg: (hex: number) => void;
  onGround: (hex: number) => void;
  onGrid: (hex: number) => void;
  onGridOn: (on: boolean) => void;
  onAgent: (hex: number) => void;
  onBloom: (v: number) => void;
}

const toHex = (n: number): string => `#${(n >>> 0).toString(16).padStart(6, "0").slice(-6)}`;
const toNum = (s: string): number => parseInt(s.replace("#", ""), 16) || 0;

function ColorRow({
  label,
  value,
  onChange,
}: {
  label: string;
  value: number;
  onChange: (hex: number) => void;
}) {
  return (
    <div className="flex items-center justify-between">
      <Label className="text-label text-muted-foreground">{label}</Label>
      <input
        type="color"
        value={toHex(value)}
        onChange={(e) => onChange(toNum(e.target.value))}
        className="h-6 w-9 cursor-pointer rounded border border-border bg-transparent p-0.5"
        aria-label={label}
      />
    </div>
  );
}

function ScenePanelImpl(props: ScenePanelProps) {
  const { cfg, theme, collapsed } = props;

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
          <div className="space-y-1.5">
            <Label className="text-label text-muted-foreground">Theme</Label>
            <div className="grid grid-cols-2 gap-2">
              <Button
                size="sm"
                variant={theme === "light" ? "default" : "outline"}
                className="h-8"
                onClick={() => props.onSetTheme("light")}
              >
                <Sun className="size-3.5" /> Light
              </Button>
              <Button
                size="sm"
                variant={theme === "dark" ? "default" : "outline"}
                className="h-8"
                onClick={() => props.onSetTheme("dark")}
              >
                <Moon className="size-3.5" /> Dark
              </Button>
            </div>
          </div>

          <Separator />

          <div className="space-y-2">
            <Label className="text-label font-medium">Colors</Label>
            <ColorRow label="Background" value={cfg.bg} onChange={props.onBg} />
            <ColorRow label="Ground" value={cfg.ground} onChange={props.onGround} />
            <ColorRow label="Grid" value={cfg.grid} onChange={props.onGrid} />
            <ColorRow label="Agent" value={cfg.agent} onChange={props.onAgent} />
          </div>

          <Separator />

          <div className="flex items-center justify-between">
            <Label htmlFor="gridOn" className="text-label text-muted-foreground">
              Show grid
            </Label>
            <Switch
              id="gridOn"
              size="sm"
              checked={cfg.gridOn}
              onCheckedChange={(c) => props.onGridOn(Boolean(c))}
            />
          </div>

          <div className="space-y-1.5">
            <div className="flex items-center justify-between text-label">
              <span className="text-muted-foreground">Bloom</span>
              <span className="font-medium tabular-nums">{cfg.bloom.toFixed(2)}</span>
            </div>
            <Slider
              value={[cfg.bloom]}
              min={0}
              max={1.5}
              step={0.05}
              onValueChange={(v) => props.onBloom(Array.isArray(v) ? v[0] : (v as number))}
            />
          </div>

          <p className="text-micro leading-tight text-muted-foreground">
            Tweaks are live and saved per theme, so each of Light/Dark keeps its
            own colors across reloads.
          </p>
        </div>
      )}
    </Panel>
  );
}

export const ScenePanel = memo(ScenePanelImpl);
