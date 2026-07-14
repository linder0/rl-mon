"use client";

import Link from "next/link";
import {
  ChevronRight,
  GraduationCap,
  LifeBuoy,
  Moon,
  Pause,
  Play,
  RotateCcw,
  Sun,
  Zap,
} from "lucide-react";
import { Panel } from "@/components/ui/panel";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Separator } from "@/components/ui/separator";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { LearningState, TaskInfo } from "@/lib/viewerApp";
import { fmtSteps } from "@/lib/format";
import type { LoopStats, ParityResult } from "@/lib/loop";
import type { Backend } from "@/components/scene/webgpuRenderer";

interface ControlPanelProps {
  /** Project (env label) this viewer is scoped to. */
  label: string;
  /** This project's iterations (run names, best-first) and the loaded one. */
  runNames: string[];
  currentRun: string;
  onSelectRun: (name: string) => void;
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
  /** Env-specific task capabilities (null until the run is loaded). */
  task: TaskInfo | null;
  /** Current food spawn distance (meters) shown on the task slider. */
  foodMax: number;
  onFoodMax: (v: number) => void;
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
          <nav className="flex min-w-0 items-center gap-1 text-label">
            <Link href="/" className="shrink-0 text-muted-foreground hover:text-primary">
              Projects
            </Link>
            <ChevronRight className="size-3 shrink-0 text-muted-foreground" />
            <Link
              href={`/p/${props.label}`}
              className="truncate font-medium hover:text-primary"
            >
              {props.label}
            </Link>
          </nav>
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
          <Label className="text-label text-muted-foreground">Iteration</Label>
          <Select value={props.currentRun} onValueChange={(v) => props.onSelectRun(String(v))}>
            <SelectTrigger className="h-8 w-full text-sm">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {props.runNames.map((name) => (
                <SelectItem key={name} value={name}>
                  {name}
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

        {/* Get-up task: the whole point is recovering from a fall, so offer the
            shove any time (no combined mode needed). */}
        {props.task?.getup && (
          <>
            <Separator />
            <Button size="sm" variant="secondary" className="h-8" onClick={props.onKnockOver}>
              <Zap className="size-3.5" />
              Knock over
            </Button>
            <p className="text-micro leading-tight text-muted-foreground">
              task: right itself after a fall
            </p>
          </>
        )}

        {/* Foraging task: probe generalization by spawning food outside the
            trained range. */}
        {props.task?.food && (
          <>
            <Separator />
            <div className="space-y-1.5">
              <div className="flex items-center justify-between text-label">
                <span className="text-muted-foreground">Food distance</span>
                <span className="font-medium tabular-nums">{props.foodMax.toFixed(1)} m</span>
              </div>
              <Slider
                value={[props.foodMax]}
                min={1}
                max={20}
                step={0.5}
                onValueChange={(v) => props.onFoodMax(Array.isArray(v) ? v[0] : (v as number))}
              />
              <p className="text-micro leading-tight text-muted-foreground">
                trained on {props.task.food.spawnMin.toFixed(0)}–
                {props.task.food.spawnMax.toFixed(0)} m spawns
              </p>
            </div>
          </>
        )}

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
                    {frame ? `${fmtSteps(frame.step)} steps` : "—"}
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
