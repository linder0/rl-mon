"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useTheme } from "next-themes";
import Link from "next/link";
import { Loader2 } from "lucide-react";
import { ViewerApp, type TaskInfo, type Variant } from "@/lib/viewerApp";
import {
  RECOVERY_ENVS,
  aggregateRuns,
  defaultVariant,
  findEnvByLabel,
  findRecoveryEntry,
  sortRunsByEval,
} from "@/lib/catalog";
import type { ChartRun } from "@/lib/chartDefs";
import { useCatalog, useRunStats } from "@/lib/hooks";
import { usePersistentState } from "@/lib/persist";
import type { Theme } from "@/lib/theme";
import type { Backend } from "@/components/scene/webgpuRenderer";
import type { LoopStats, ParityResult } from "@/lib/loop";
import type { RunAgg } from "@/lib/types";
import { ControlPanel } from "@/components/ControlPanel";
import { StatsPanel } from "@/components/StatsPanel";
import { DialsPanel } from "@/components/DialsPanel";
import { ScenePanel } from "@/components/ScenePanel";
import { NetPanel } from "@/components/net/NetPanel";
import type { PanelView } from "@/components/PanelViewToggle";
import { readViewerInit, useLearningMode, useSceneColors } from "./hooks";

/** The project viewer: a full-screen 3D rollout of one of the project's
 * iterations (training runs), switchable via the Iteration dropdown. Opens on
 * `initialRun` when given (the /p/[env]/[run] deep link), else the best run. */
export default function IterationView({
  label,
  initialRun = null,
}: {
  label: string;
  initialRun?: string | null;
}) {
  const sceneRef = useRef<HTMLCanvasElement>(null);
  const actorRef = useRef<HTMLCanvasElement>(null);
  const criticRef = useRef<HTMLCanvasElement>(null);
  const appRef = useRef<ViewerApp | null>(null);

  const { resolvedTheme, setTheme } = useTheme();
  const theme: Theme = resolvedTheme === "light" ? "light" : "dark";

  const { index, error: catalogError } = useCatalog();

  const [runs, setRuns] = useState<RunAgg[]>([]);
  const [agg, setAgg] = useState<RunAgg | null>(null);
  const [variant, setVariant] = useState<Variant>("best");
  const [notFound, setNotFound] = useState(false);
  const [liveStats, setLiveStats] = useState<LoopStats | null>(null);
  const [parity, setParity] = useState<ParityResult | null>(null);
  const [value, setValue] = useState<number | null>(null);
  const [backend, setBackend] = useState<Backend | null>(null);

  const [loadingText, setLoadingText] = useState<string | null>("Loading…");
  const [error, setError] = useState<string | null>(null);

  const [playing, setPlaying] = useState(true);
  const [recoveryAvailable, setRecoveryAvailable] = useState(false);
  const [recoveryOn, setRecoveryOn] = useState(false);
  const [task, setTask] = useState<TaskInfo | null>(null);
  // Food spawn distance override (meters); null = the trained range.
  const [foodMax, setFoodMax] = useState<number | null>(null);
  const [speed, setSpeed] = usePersistentState("speed", 1);
  const [follow, setFollow] = usePersistentState("follow", true);
  const [statsCollapsed, setStatsCollapsed] = usePersistentState("statsCollapsed", false);
  const [netCollapsed, setNetCollapsed] = usePersistentState("netCollapsed", false);
  const [panelView, setPanelView] = usePersistentState<PanelView>("panelView", "stats");

  const scene3d = useSceneColors(appRef, theme);
  const learn = useLearningMode(appRef);

  // Training stats for every iteration (batched overlay charts + the selected
  // run's config), loaded through the catalog cache — the sim controller
  // doesn't touch stats at all.
  const statsByRun = useRunStats(runs);
  const stats = (agg && statsByRun.get(agg.runName)) ?? null;
  const overlayRuns: ChartRun[] = runs
    .filter((r) => statsByRun.has(r.runName))
    .map((r) => ({ name: r.runName, color: r.color, stats: statsByRun.get(r.runName)! }));

  useEffect(() => {
    if (catalogError) {
      setError(catalogError);
      setLoadingText(null);
    }
  }, [catalogError]);

  // Boot the sim once the catalog resolves this project. index/label/initialRun
  // are stable for the lifetime of the page, so this runs to completion once;
  // switching iterations afterwards goes through onSelectRun (no re-boot).
  useEffect(() => {
    if (!index) return;
    const scene = sceneRef.current;
    const actor = actorRef.current;
    const critic = criticRef.current;
    if (!scene || !actor || !critic) return;

    const env = findEnvByLabel(index, label);
    const envRuns = env ? sortRunsByEval(aggregateRuns(env)) : [];
    if (!env || envRuns.length === 0) {
      setNotFound(true);
      setLoadingText(null);
      return;
    }
    // Open on the deep-linked run when given (fall back to the best run).
    const found = envRuns.find((r) => r.runName === initialRun) ?? envRuns[0];
    setRuns(envRuns);
    setAgg(found);
    setVariant(defaultVariant(found));

    const recovery = RECOVERY_ENVS.has(env.env_id) ? findRecoveryEntry(index) : null;
    setRecoveryAvailable(recovery != null);

    const initialTheme: Theme = document.documentElement.classList.contains("dark")
      ? "dark"
      : "light";

    let valueLast = 0;
    const app = new ViewerApp(
      scene,
      actor,
      critic,
      {
        onLoading: (t) => setLoadingText(t),
        onError: (m) => {
          setError(m);
          setLoadingText(null);
        },
        onVariant: (v) => setVariant(v),
        onLiveStats: (s) => setLiveStats(s),
        onParity: (r) => setParity(r),
        onValue: (v) => {
          const now = performance.now();
          if (now - valueLast < 160) return;
          valueLast = now;
          setValue(v);
        },
        onBackend: (b) => setBackend(b),
        onSceneColors: scene3d.applyFromApp,
        onLearning: learn.onLearningState,
        onTask: (t) => setTask(t),
      },
      initialTheme,
      readViewerInit(),
    );
    appRef.current = app;
    void app.start(found, recovery);

    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement | null)?.tagName;
      if (tag === "SELECT" || tag === "INPUT") return;
      if (e.code === "Space") {
        e.preventDefault();
        setPlaying((p) => {
          const np = !p;
          app.setPlaying(np);
          return np;
        });
      } else if (e.code === "KeyR") {
        app.reset();
      }
    };
    window.addEventListener("keydown", onKey);

    return () => {
      window.removeEventListener("keydown", onKey);
      app.dispose();
      appRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [index, label, initialRun]);

  // Push theme changes to the 3D scene (CSS handles the chrome). setTheme applies
  // that theme's saved color override (or its defaults) and reports the effective
  // colors back via onSceneColors, which re-seeds the Scene panel's inputs.
  useEffect(() => {
    appRef.current?.setTheme(theme);
  }, [theme]);

  const togglePlay = useCallback(
    () =>
      setPlaying((p) => {
        const np = !p;
        appRef.current?.setPlaying(np);
        return np;
      }),
    [],
  );
  const toggleNet = useCallback(
    () =>
      setNetCollapsed((c) => {
        const nc = !c;
        appRef.current?.setNetCollapsed(nc);
        return nc;
      }),
    [],
  );
  const onReset = useCallback(() => appRef.current?.reset(), []);
  const onToggleRecovery = useCallback((on: boolean) => {
    setRecoveryOn(on);
    appRef.current?.setRecovery(on);
  }, []);
  const onKnockOver = useCallback(() => appRef.current?.knockOver(), []);
  const onFoodMax = useCallback((v: number) => {
    setFoodMax(v);
    appRef.current?.setFoodSpawnMax(v);
  }, []);
  const onSpeed = useCallback((v: number) => {
    setSpeed(v);
    appRef.current?.setSpeed(v);
  }, []);
  const onFollow = useCallback((v: boolean) => {
    setFollow(v);
    appRef.current?.setFollow(v);
  }, []);
  const onToggleStats = useCallback(() => setStatsCollapsed((c) => !c), []);
  const onToggleTheme = useCallback(
    () => setTheme(document.documentElement.classList.contains("dark") ? "light" : "dark"),
    [setTheme],
  );
  const onVariant = useCallback((v: Variant) => appRef.current?.setVariant(v), []);
  const onSelectRun = useCallback(
    (name: string) => {
      const next = runs.find((r) => r.runName === name);
      if (!next || next.runName === agg?.runName) return;
      setAgg(next);
      setVariant(defaultVariant(next));
      learn.stopEvolve();
      appRef.current?.setRun(next);
      // Keep the URL shareable without remounting the page (a route change
      // would tear down the WebGPU context).
      window.history.replaceState(null, "", `/p/${label}/${encodeURIComponent(name)}`);
    },
    [runs, agg, label, learn.stopEvolve],
  );
  const onSetThemeExplicit = useCallback((t: "light" | "dark") => setTheme(t), [setTheme]);

  if (notFound) {
    return (
      <div className="flex h-screen w-screen flex-col items-center justify-center gap-3 bg-background">
        <div className="text-sm text-muted-foreground">
          No project named “{label}” (or it has no exported runs).
        </div>
        <Link href="/" className="text-sm font-medium text-primary hover:underline">
          Back to projects
        </Link>
      </div>
    );
  }

  const overlayHidden = loadingText === null && !error;
  const summary = agg ? { run: agg, variant } : null;

  return (
    <div className="relative h-screen w-screen overflow-hidden">
      <canvas id="scene" ref={sceneRef} />

      {/* Left column: control pinned top, network viz pinned bottom-left. The
          net panel's mt-auto keeps it at the bottom when there's room and
          collapses to stack directly below the controls otherwise, so the two
          never overlap. */}
      <div className="pointer-events-none absolute inset-y-3 left-3 z-(--z-panel) flex flex-col gap-3">
        <ControlPanel
          label={label}
          runNames={runs.map((r) => r.runName)}
          currentRun={agg?.runName ?? ""}
          onSelectRun={onSelectRun}
          playing={playing}
          onTogglePlay={togglePlay}
          onReset={onReset}
          speed={speed}
          onSpeed={onSpeed}
          follow={follow}
          onFollow={onFollow}
          liveStats={liveStats}
          parity={parity}
          backend={backend}
          theme={theme}
          onToggleTheme={onToggleTheme}
          learning={learn.learning}
          evolvePlaying={learn.evolvePlaying}
          onToggleLearning={learn.onToggleLearning}
          onToggleEvolve={learn.onToggleEvolve}
          onScrub={learn.onScrub}
          recoveryAvailable={recoveryAvailable}
          recoveryOn={recoveryOn}
          onToggleRecovery={onToggleRecovery}
          onKnockOver={onKnockOver}
          task={task}
          foodMax={foodMax ?? task?.food?.spawnMax ?? 8}
          onFoodMax={onFoodMax}
        />
        <div className="mt-auto">
          <NetPanel
            actorRef={actorRef}
            criticRef={criticRef}
            collapsed={netCollapsed}
            onToggle={toggleNet}
            value={value}
          />
        </div>
      </div>

      {/* Right column: stats or the live dials HUD (swap via the header toggle). */}
      <div className="pointer-events-none absolute inset-y-3 right-3 z-(--z-panel) flex items-start">
        {panelView === "scene" ? (
          <ScenePanel
            cfg={scene3d.sceneCfg}
            theme={theme}
            collapsed={statsCollapsed}
            view={panelView}
            onSetView={setPanelView}
            onToggleCollapsed={onToggleStats}
            onSetTheme={onSetThemeExplicit}
            onBg={scene3d.onBg}
            onGround={scene3d.onGround}
            onGrid={scene3d.onGrid}
            onGridOn={scene3d.onGridOn}
            onShowForces={scene3d.onShowForces}
            onForceViz={scene3d.onForceViz}
            onAgent={scene3d.onAgent}
            onBloom={scene3d.onBloom}
          />
        ) : panelView === "dials" ? (
          <DialsPanel
            liveStats={liveStats}
            value={value}
            speed={speed}
            summary={summary}
            collapsed={statsCollapsed}
            view={panelView}
            onSetView={setPanelView}
            onToggleCollapsed={onToggleStats}
          />
        ) : (
          <StatsPanel
            agg={agg}
            variant={variant}
            stats={stats}
            overlayRuns={overlayRuns}
            collapsed={statsCollapsed}
            theme={theme}
            view={panelView}
            onSetView={setPanelView}
            onToggleCollapsed={onToggleStats}
            onVariant={onVariant}
          />
        )}
      </div>

      {!overlayHidden && (
        <div
          className={`absolute inset-0 z-(--z-overlay) flex flex-col items-center justify-center gap-4 bg-background transition-opacity ${
            error ? "" : "duration-500"
          }`}
        >
          {error ? (
            <div
              className="max-w-lg rounded-lg border border-destructive/40 bg-destructive/10 p-5 text-sm leading-relaxed text-destructive [&_code]:rounded [&_code]:bg-black/30 [&_code]:px-1.5 [&_code]:py-0.5"
              dangerouslySetInnerHTML={{ __html: error }}
            />
          ) : (
            <>
              <Loader2 className="size-8 animate-spin text-primary" />
              <div className="text-sm text-muted-foreground">{loadingText}</div>
            </>
          )}
        </div>
      )}
    </div>
  );
}
