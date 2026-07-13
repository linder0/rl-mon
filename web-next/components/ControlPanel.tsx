"use client";

import { GraduationCap, LifeBuoy, Moon, Pause, Play, RotateCcw, Sun, Zap } from "lucide-react";
import { Panel } from "@/components/ui/panel";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Separator } from "@/components/ui/separator";
import type { LearningState } from "@/lib/viewerApp";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { LoopStats, ParityResult } from "@/lib/loop";
import type { Backend } from "@/components/scene/webgpuRenderer";
import type { EnvGroup } from "@/lib/types";

interface ControlPanelProps {
  envs: EnvGroup[];
  currentEnv: string;
  onSelectEnv: (id: string) => void;
  playing: boolean;
  onTogglePlay: () => void;
  onReset: () => void;
  speed: number;
  onSpeed: (v: number) => void;
  follow: boolean;
  onFollow: (v: boolean) => void;
  liveStats: LoopStats | null;
  parity: ParityResult | null;
  backend: Backend | null;
  theme: string;
  onToggleTheme: () => void;
  learning: LearningState | null;
  evolvePlaying: boolean;
  onToggleLearning: (on: boolean) => void;
  onToggleEvolve: () => void;
  onScrub: (index: number) => void;
  recoveryAvailable: boolean;
  recoveryOn: boolean;
  onToggleRecovery: (on: boolean) => void;
  onKnockOver: () => void;
}

function formatStep(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(n % 1_000_000 ? 1 : 0)}M`;
  if (n >= 1_000) return `${Math.round(n / 1_000)}k`;
  return String(n);
}

function Stat({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <>
      <span className="text-muted-foreground">{label}</span>
      <span className="text-right font-medium tabular-nums">{children}</span>
    </>
  );
}

export function ControlPanel(props: ControlPanelProps) {
  const { liveStats, parity, backend, theme, learning } = props;
  const parityOk = parity ? parity.maxActionError < 1e-3 : null;
  const learnActive = learning?.active ?? false;
  const frames = learning?.frames ?? [];
  const frame = learnActive ? frames[learning?.index ?? 0] : undefined;

  return (
    <Panel size="sm">
      <div className="flex flex-col gap-3 p-3">
        <div className="flex items-center justify-between">
          <div className="flex items-baseline gap-1.5">
            <span className="text-sm font-semibold tracking-tight">RL MuJoCo</span>
            <span className="text-micro font-medium uppercase tracking-widest text-muted-foreground">
              viewer
            </span>
          </div>
          <Button
            variant="ghost"
            size="icon"
            className="size-7"
            onClick={props.onToggleTheme}
            aria-label="Toggle color theme"
            title={theme === "light" ? "Switch to dark mode" : "Switch to light mode"}
          >
            {theme === "light" ? <Moon className="size-4" /> : <Sun className="size-4" />}
          </Button>
        </div>

        <div className="space-y-1">
          <Label className="text-label text-muted-foreground">Environment</Label>
          <Select value={props.currentEnv} onValueChange={(v) => props.onSelectEnv(String(v))}>
            <SelectTrigger className="h-8 w-full text-sm">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {props.envs.map((e) => (
                <SelectItem key={e.env_id} value={e.env_id}>
                  {e.env_id}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="grid grid-cols-2 gap-2">
          <Button size="sm" className="h-8" onClick={props.onTogglePlay}>
            {props.playing ? <Pause className="size-3.5" /> : <Play className="size-3.5" />}
            {props.playing ? "Pause" : "Play"}
          </Button>
          <Button size="sm" variant="secondary" className="h-8" onClick={props.onReset}>
            <RotateCcw className="size-3.5" />
            Reset
          </Button>
        </div>

        <div className="space-y-1.5">
          <div className="flex items-center justify-between text-label">
            <span className="text-muted-foreground">Speed</span>
            <span className="font-medium tabular-nums">{props.speed.toFixed(1)}×</span>
          </div>
          <Slider
            value={[props.speed]}
            min={0.1}
            max={3}
            step={0.1}
            onValueChange={(v) => props.onSpeed(Array.isArray(v) ? v[0] : (v as number))}
          />
        </div>

        <div className="flex items-center justify-between">
          <Label htmlFor="follow" className="text-label text-muted-foreground">
            Follow camera
          </Label>
          <Switch
            id="follow"
            size="sm"
            checked={props.follow}
            onCheckedChange={(c) => props.onFollow(Boolean(c))}
          />
        </div>

        <Separator />

        <div className="grid grid-cols-2 gap-x-3 gap-y-1 text-label">
          <Stat label="Episode">{liveStats?.episode ?? "—"}</Stat>
          <Stat label="Step">{liveStats?.step ?? "—"}</Stat>
          <Stat label="Distance">
            {liveStats ? `${liveStats.distance.toFixed(2)} m` : "—"}
          </Stat>
          <Stat label="State">
            {liveStats ? (
              <span className={liveStats.healthy ? "text-success" : "text-destructive"}>
                {liveStats.healthy ? "healthy" : "fallen"}
              </span>
            ) : (
              "—"
            )}
          </Stat>
          <Stat label="FPS">{liveStats ? liveStats.fps.toFixed(0) : "—"}</Stat>
          <Stat label="Renderer">{backend ? backend.toUpperCase() : "—"}</Stat>
        </div>

        {props.recoveryAvailable && (
          <>
            <Separator />
            <div className="flex items-center justify-between">
              <Label htmlFor="recovery" className="flex items-center gap-1.5 text-label text-muted-foreground">
                <LifeBuoy className="size-3.5" />
                Auto-recover (get-up)
              </Label>
              <Switch
                id="recovery"
                size="sm"
                checked={props.recoveryOn}
                onCheckedChange={(c) => props.onToggleRecovery(Boolean(c))}
              />
            </div>
            {props.recoveryOn && (
              <>
                <Button
                  size="sm"
                  variant="secondary"
                  className="h-8"
                  onClick={props.onKnockOver}
                >
                  <Zap className="size-3.5" />
                  Knock over
                </Button>
                <p className="text-micro leading-tight text-muted-foreground">
                  {liveStats?.recovering
                    ? "recovering — get-up policy driving"
                    : "foraging — forager policy driving"}
                </p>
              </>
            )}
          </>
        )}

        {learning?.available && (
          <>
            <Separator />
            <div className="flex items-center justify-between">
              <Label htmlFor="learning" className="flex items-center gap-1.5 text-label text-muted-foreground">
                <GraduationCap className="size-3.5" />
                Learning mode
              </Label>
              <Switch
                id="learning"
                size="sm"
                checked={learnActive}
                onCheckedChange={(c) => props.onToggleLearning(Boolean(c))}
              />
            </div>

            {learnActive && frames.length > 0 && (
              <div className="space-y-1.5">
                <div className="flex items-center justify-between text-label">
                  <span className="text-muted-foreground">
                    {frame ? `${formatStep(frame.step)} steps` : "—"}
                  </span>
                  <span className="font-medium tabular-nums">
                    {frame?.reward != null ? `reward ${Math.round(frame.reward)}` : ""}
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <Button
                    size="icon"
                    variant="secondary"
                    className="size-8 shrink-0"
                    onClick={props.onToggleEvolve}
                    aria-label={props.evolvePlaying ? "Pause evolution" : "Play evolution"}
                    title={props.evolvePlaying ? "Pause evolution" : "Watch it evolve"}
                  >
                    {props.evolvePlaying ? <Pause className="size-3.5" /> : <Play className="size-3.5" />}
                  </Button>
                  <Slider
                    className="flex-1"
                    value={[learning?.index ?? 0]}
                    min={0}
                    max={Math.max(frames.length - 1, 0)}
                    step={1}
                    onValueChange={(v) => props.onScrub(Array.isArray(v) ? v[0] : (v as number))}
                  />
                </div>
                <p className="text-micro leading-tight text-muted-foreground">
                  checkpoint {(learning?.index ?? 0) + 1} / {frames.length} · body evolves live
                </p>
              </div>
            )}
          </>
        )}

        {parity && (
          <p
            className={`text-micro leading-tight ${
              parityOk ? "text-success" : "text-destructive"
            }`}
          >
            {parityOk
              ? `parity ok · matches Python (Δ ${parity.maxActionError.toExponential(1)})`
              : `parity warning · Δ ${parity.maxActionError.toExponential(1)}`}
          </p>
        )}
      </div>
    </Panel>
  );
}
