"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useTheme } from "next-themes";
import { Loader2 } from "lucide-react";
import {
  ViewerApp,
  type LearningState,
  type LiveSel,
  type Variant,
  type SceneColors,
} from "@/lib/viewerApp";
import { usePersistentState, readPersisted } from "@/lib/persist";
import { SCENE_THEMES, type Theme } from "@/lib/theme";
import type { Backend } from "@/components/scene/webgpuRenderer";
import type { ChartPayload } from "@/lib/chartTypes";
import type { LoopStats, ParityResult } from "@/lib/loop";
import type { EnvGroup, RunAgg, StatsData } from "@/lib/types";
import { ControlPanel } from "@/components/ControlPanel";
import { StatsPanel } from "@/components/StatsPanel";
import { DialsPanel } from "@/components/DialsPanel";
import { ScenePanel, type SceneCfg } from "@/components/ScenePanel";
import { NetPanel } from "@/components/net/NetPanel";
import type { PanelView } from "@/components/PanelViewToggle";

// Seed the Scene panel from the dark theme's scene colors so there's one source
// of truth for the defaults (grid maps to the primary grid line, agent to the
// robot accent). gridOn/bloom are display prefs with no theme equivalent.
const DEFAULT_SCENE: SceneCfg = {
  bg: SCENE_THEMES.dark.bg,
  ground: SCENE_THEMES.dark.ground,
  grid: SCENE_THEMES.dark.grid1,
  agent: SCENE_THEMES.dark.robot,
  gridOn: true,
  bloom: 0.5,
};

type ColorsByTheme = Partial<Record<Theme, SceneColors>>;
const COLORS_KEY = "sceneColors";

/** Reads persisted UI prefs so they can seed the ViewerApp on startup. */
function persistedInit() {
  const scene = readPersisted<SceneCfg>("sceneCfg", DEFAULT_SCENE);
  return {
    speed: readPersisted("speed", 1),
    follow: readPersisted("follow", true),
    netCollapsed: readPersisted("netCollapsed", false),
    gridOn: scene.gridOn,
    bloom: scene.bloom,
    colorOverrides: readPersisted<ColorsByTheme>(COLORS_KEY, {}),
  };
}

export default function Viewer() {
  const sceneRef = useRef<HTMLCanvasElement>(null);
  const actorRef = useRef<HTMLCanvasElement>(null);
  const criticRef = useRef<HTMLCanvasElement>(null);
  const appRef = useRef<ViewerApp | null>(null);

  const { resolvedTheme, setTheme } = useTheme();
  const theme: Theme = resolvedTheme === "light" ? "light" : "dark";

  const [envs, setEnvs] = useState<EnvGroup[]>([]);
  const [currentEnv, setCurrentEnv] = useState("");
  const [runsForEnv, setRunsForEnv] = useState<RunAgg[]>([]);
  const [live, setLive] = useState<LiveSel>({ runName: "", variant: "best" });
  const [compared, setCompared] = useState<string[]>([]);
  const [summary, setSummary] = useState<{ run: RunAgg; variant: Variant } | null>(null);
  const [config, setConfig] = useState<StatsData | null>(null);
  const [charts, setCharts] = useState<ChartPayload | null>(null);
  const [liveStats, setLiveStats] = useState<LoopStats | null>(null);
  const [parity, setParity] = useState<ParityResult | null>(null);
  const [value, setValue] = useState<number | null>(null);
  const [backend, setBackend] = useState<Backend | null>(null);

  const [loadingText, setLoadingText] = useState<string | null>("Loading…");
  const [error, setError] = useState<string | null>(null);

  const [playing, setPlaying] = useState(true);
  const [learning, setLearning] = useState<LearningState | null>(null);
  const [evolvePlaying, setEvolvePlaying] = useState(false);
  const [recoveryAvailable, setRecoveryAvailable] = useState(false);
  const [recoveryOn, setRecoveryOn] = useState(false);
  const [speed, setSpeed] = usePersistentState("speed", 1);
  const [follow, setFollow] = usePersistentState("follow", true);
  const [statsCollapsed, setStatsCollapsed] = usePersistentState("statsCollapsed", false);
  const [netCollapsed, setNetCollapsed] = usePersistentState("netCollapsed", false);
  const [panelView, setPanelView] = usePersistentState<PanelView>("panelView", "stats");
  // gridOn/bloom are theme-independent and persisted here; the colors in sceneCfg
  // are display state kept in sync with the renderer (whose effective colors are
  // pushed via onSceneColors). Custom colors persist per theme in colorsByTheme.
  const [sceneCfg, setSceneCfg] = usePersistentState<SceneCfg>("sceneCfg", DEFAULT_SCENE);
  const [, setColorsByTheme] = usePersistentState<ColorsByTheme>(COLORS_KEY, {});

  useEffect(() => {
    const scene = sceneRef.current;
    const actor = actorRef.current;
    const critic = criticRef.current;
    if (!scene || !actor || !critic) return;

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
        onEnvs: (list, cur) => {
          setEnvs(list);
          setCurrentEnv(cur);
        },
        onRunTable: (runs, liveSel, comparedList) => {
          setRunsForEnv(runs);
          setLive(liveSel);
          setCompared(comparedList);
        },
        onSummary: (run, variant) => setSummary({ run, variant }),
        onConfig: (stats) => setConfig(stats),
        onCharts: (payload) => setCharts(payload),
        onLiveStats: (s) => setLiveStats(s),
        onParity: (r) => setParity(r),
        onValue: (v) => {
          const now = performance.now();
          if (now - valueLast < 160) return;
          valueLast = now;
          setValue(v);
        },
        onBackend: (b) => setBackend(b),
        onSceneColors: (c) =>
          setSceneCfg((s) => ({ ...s, bg: c.bg, ground: c.ground, grid: c.grid, agent: c.agent })),
        onLearning: (state) => {
          setLearning(state);
          if (!state.active) setEvolvePlaying(false);
        },
        onRecoveryAvailable: (available) => {
          setRecoveryAvailable(available);
          if (!available) setRecoveryOn(false);
        },
      },
      initialTheme,
      persistedInit(),
    );
    appRef.current = app;
    void app.start();

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
  }, []);

  // Push theme changes to the 3D scene (CSS handles the chrome). setTheme applies
  // that theme's saved color override (or its defaults) and reports the effective
  // colors back via onSceneColors, which re-seeds the Scene panel's inputs.
  useEffect(() => {
    appRef.current?.setTheme(theme);
  }, [theme]);

  // Auto-evolve: while engaged, advance one checkpoint every EVOLVE_MS so the
  // agent visibly improves on its own. Re-runs on each frame change (driven by
  // the app's onLearning callback), stopping at the fully-trained checkpoint.
  useEffect(() => {
    if (!learning?.active || !evolvePlaying) return;
    if (learning.index >= learning.frames.length - 1) {
      setEvolvePlaying(false);
      return;
    }
    const EVOLVE_MS = 2200;
    const t = setTimeout(() => appRef.current?.showFrame(learning.index + 1), EVOLVE_MS);
    return () => clearTimeout(t);
  }, [learning?.active, evolvePlaying, learning?.index, learning?.frames.length]);

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
  const onToggleLearning = useCallback((on: boolean) => {
    if (!on) setEvolvePlaying(false);
    void appRef.current?.setLearning(on);
  }, []);
  const onScrub = useCallback((i: number) => {
    setEvolvePlaying(false);
    void appRef.current?.showFrame(i);
  }, []);
  const onToggleEvolve = useCallback(() => {
    const starting = !evolvePlaying;
    // Restarting from the end replays the whole arc from the untrained policy.
    if (starting && learning && learning.index >= learning.frames.length - 1) {
      void appRef.current?.showFrame(0);
    }
    setEvolvePlaying(starting);
  }, [evolvePlaying, learning]);
  const onSelectEnv = useCallback((id: string) => {
    setCurrentEnv(id);
    appRef.current?.selectEnvId(id);
  }, []);
  const onToggleRecovery = useCallback((on: boolean) => {
    setRecoveryOn(on);
    appRef.current?.setRecovery(on);
  }, []);
  const onKnockOver = useCallback(() => appRef.current?.knockOver(), []);
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
  const onPickRun = useCallback((name: string) => appRef.current?.pickRun(name), []);
  const onVariant = useCallback(
    (name: string, v: Variant) => appRef.current?.setVariant(name, v),
    [],
  );
  const onToggleCompare = useCallback((name: string) => appRef.current?.toggleCompare(name), []);

  // Scene panel handlers: apply live to the renderer + mirror into local state.
  // After each color edit, snapshot the renderer's effective colors and save them
  // as this theme's override so they're restored on reload and theme switches.
  const persistColors = useCallback(() => {
    const app = appRef.current;
    if (!app) return;
    const colors = app.sceneColors();
    app.setColorOverride(theme, colors);
    setColorsByTheme((m) => ({ ...m, [theme]: colors }));
  }, [theme, setColorsByTheme]);

  const onSceneBg = useCallback(
    (hex: number) => {
      appRef.current?.setBackgroundColor(hex);
      setSceneCfg((s) => ({ ...s, bg: hex }));
      persistColors();
    },
    [persistColors, setSceneCfg],
  );
  const onSceneGround = useCallback(
    (hex: number) => {
      appRef.current?.setGroundColor(hex);
      setSceneCfg((s) => ({ ...s, ground: hex }));
      persistColors();
    },
    [persistColors, setSceneCfg],
  );
  const onSceneGrid = useCallback(
    (hex: number) => {
      appRef.current?.setGridColor(hex);
      setSceneCfg((s) => ({ ...s, grid: hex }));
      persistColors();
    },
    [persistColors, setSceneCfg],
  );
  const onSceneGridOn = useCallback(
    (on: boolean) => {
      appRef.current?.setGridVisible(on);
      setSceneCfg((s) => ({ ...s, gridOn: on }));
    },
    [setSceneCfg],
  );
  const onSceneAgent = useCallback(
    (hex: number) => {
      appRef.current?.setAgentColor(hex);
      setSceneCfg((s) => ({ ...s, agent: hex }));
      persistColors();
    },
    [persistColors, setSceneCfg],
  );
  const onSceneBloom = useCallback((v: number) => {
    appRef.current?.setBloomStrength(v);
    setSceneCfg((s) => ({ ...s, bloom: v }));
  }, []);
  const onSetThemeExplicit = useCallback((t: "light" | "dark") => setTheme(t), [setTheme]);

  const overlayHidden = loadingText === null && !error;

  return (
    <div className="relative h-screen w-screen overflow-hidden">
      <canvas id="scene" ref={sceneRef} />

      {/* Left column: control pinned top, network viz pinned bottom-left. The
          net panel's mt-auto keeps it at the bottom when there's room and
          collapses to stack directly below the controls otherwise, so the two
          never overlap. */}
      <div className="pointer-events-none absolute inset-y-3 left-3 z-(--z-panel) flex flex-col gap-3">
        <ControlPanel
          envs={envs}
          currentEnv={currentEnv}
          onSelectEnv={onSelectEnv}
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
          learning={learning}
          evolvePlaying={evolvePlaying}
          onToggleLearning={onToggleLearning}
          onToggleEvolve={onToggleEvolve}
          onScrub={onScrub}
          recoveryAvailable={recoveryAvailable}
          recoveryOn={recoveryOn}
          onToggleRecovery={onToggleRecovery}
          onKnockOver={onKnockOver}
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
            cfg={sceneCfg}
            theme={theme}
            collapsed={statsCollapsed}
            view={panelView}
            onSetView={setPanelView}
            onToggleCollapsed={onToggleStats}
            onSetTheme={onSetThemeExplicit}
            onBg={onSceneBg}
            onGround={onSceneGround}
            onGrid={onSceneGrid}
            onGridOn={onSceneGridOn}
            onAgent={onSceneAgent}
            onBloom={onSceneBloom}
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
            runs={runsForEnv}
            live={live}
            compared={compared}
            summary={summary}
            config={config}
            charts={charts}
            collapsed={statsCollapsed}
            theme={theme}
            view={panelView}
            onSetView={setPanelView}
            onToggleCollapsed={onToggleStats}
            onPickRun={onPickRun}
            onVariant={onVariant}
            onToggleCompare={onToggleCompare}
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
