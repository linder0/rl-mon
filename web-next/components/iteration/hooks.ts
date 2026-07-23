"use client";

/** UI-state hooks for the iteration viewer, extracted so IterationView stays a
 * wiring shell: scene-color persistence and the learning-mode scrubber. Both
 * talk to the ViewerApp through the ref that IterationView owns. */

import { useCallback, useEffect, useMemo, useState, type RefObject } from "react";
import type { LearningState, SceneColors, ViewerApp, ViewerInit } from "@/lib/viewerApp";
import { usePersistentState, readPersisted } from "@/lib/persist";
import { SCENE_THEMES, type Theme } from "@/lib/theme";
import { DEFAULT_FORCE_VIZ, type ForceVizCfg } from "@/components/scene/webgpuRenderer";
import type { SceneCfg } from "@/components/ScenePanel";

// Seed the Scene panel from the dark theme's scene colors so there's one source
// of truth for the defaults (grid maps to the primary grid line, agent to the
// robot accent). gridOn/bloom are display prefs with no theme equivalent.
const DEFAULT_SCENE: SceneCfg = {
  bg: SCENE_THEMES.dark.bg,
  ground: SCENE_THEMES.dark.ground,
  grid: SCENE_THEMES.dark.grid1,
  agent: SCENE_THEMES.dark.robot,
  gridOn: true,
  showForces: false,
  forceViz: DEFAULT_FORCE_VIZ,
  bloom: 0.5,
};

type ColorsByTheme = Partial<Record<Theme, SceneColors>>;
const COLORS_KEY = "sceneColors";

/** Fill in any keys missing from a persisted scene config (older saves predate
 * showForces/forceViz), so consumers always see a complete, valid config. */
function normalizeScene(cfg: SceneCfg): SceneCfg {
  return { ...DEFAULT_SCENE, ...cfg, forceViz: { ...DEFAULT_FORCE_VIZ, ...cfg.forceViz } };
}

/** Reads persisted UI prefs so they can seed the ViewerApp on startup. */
export function readViewerInit(): ViewerInit {
  const scene = normalizeScene(readPersisted<SceneCfg>("sceneCfg", DEFAULT_SCENE));
  return {
    speed: readPersisted("speed", 1),
    follow: readPersisted("follow", true),
    netCollapsed: readPersisted("netCollapsed", false),
    gridOn: scene.gridOn,
    showForces: scene.showForces,
    forceViz: scene.forceViz,
    bloom: scene.bloom,
    colorOverrides: readPersisted<ColorsByTheme>(COLORS_KEY, {}),
  };
}

/**
 * Scene panel state: colors/grid/bloom applied live to the renderer, mirrored
 * into persistent display state, and (for colors) snapshotted as the active
 * theme's override so they're restored on reload and theme switches.
 * `applyFromApp` is the ViewerApp onSceneColors callback — the app reports its
 * effective colors after init and theme changes, re-seeding the panel inputs.
 */
export function useSceneColors(appRef: RefObject<ViewerApp | null>, theme: Theme) {
  // gridOn/bloom are theme-independent and persisted here; the colors in
  // sceneCfg are display state kept in sync with the renderer. Custom colors
  // persist per theme in colorsByTheme.
  const [rawSceneCfg, setSceneCfg] = usePersistentState<SceneCfg>("sceneCfg", DEFAULT_SCENE);
  const [, setColorsByTheme] = usePersistentState<ColorsByTheme>(COLORS_KEY, {});
  // Backfill any keys older saves lack, with a stable identity so the memoized
  // ScenePanel doesn't re-render on unrelated parent updates.
  const sceneCfg = useMemo(() => normalizeScene(rawSceneCfg), [rawSceneCfg]);

  const applyFromApp = useCallback(
    (c: SceneColors) =>
      setSceneCfg((s) => ({ ...s, bg: c.bg, ground: c.ground, grid: c.grid, agent: c.agent })),
    [setSceneCfg],
  );

  const persistColors = useCallback(() => {
    const app = appRef.current;
    if (!app) return;
    const colors = app.sceneColors();
    app.setColorOverride(theme, colors);
    setColorsByTheme((m) => ({ ...m, [theme]: colors }));
  }, [appRef, theme, setColorsByTheme]);

  /** One handler per color channel: push to the renderer, mirror to state,
   * persist the override. */
  const colorHandler = useCallback(
    (key: "bg" | "ground" | "grid" | "agent", apply: (app: ViewerApp, hex: number) => void) =>
      (hex: number) => {
        const app = appRef.current;
        if (app) apply(app, hex);
        setSceneCfg((s) => ({ ...s, [key]: hex }));
        persistColors();
      },
    [appRef, persistColors, setSceneCfg],
  );

  const onBg = colorHandler("bg", (app, hex) => app.setBackgroundColor(hex));
  const onGround = colorHandler("ground", (app, hex) => app.setGroundColor(hex));
  const onGrid = colorHandler("grid", (app, hex) => app.setGridColor(hex));
  const onAgent = colorHandler("agent", (app, hex) => app.setAgentColor(hex));
  const onGridOn = useCallback(
    (on: boolean) => {
      appRef.current?.setGridVisible(on);
      setSceneCfg((s) => ({ ...s, gridOn: on }));
    },
    [appRef, setSceneCfg],
  );
  const onShowForces = useCallback(
    (on: boolean) => {
      appRef.current?.setShowForces(on);
      setSceneCfg((s) => ({ ...s, showForces: on }));
    },
    [appRef, setSceneCfg],
  );
  /** Patch the force-viz detail options (component toggles, colors, scale),
   * pushing the merged config to the renderer and persisting it. */
  const onForceViz = useCallback(
    (patch: Partial<ForceVizCfg>) => {
      setSceneCfg((s) => {
        const next: ForceVizCfg = { ...DEFAULT_FORCE_VIZ, ...s.forceViz, ...patch };
        appRef.current?.setForceViz(next);
        return { ...s, forceViz: next };
      });
    },
    [appRef, setSceneCfg],
  );
  const onBloom = useCallback(
    (v: number) => {
      appRef.current?.setBloomStrength(v);
      setSceneCfg((s) => ({ ...s, bloom: v }));
    },
    [appRef, setSceneCfg],
  );

  return {
    sceneCfg,
    applyFromApp,
    onBg,
    onGround,
    onGrid,
    onGridOn,
    onShowForces,
    onForceViz,
    onAgent,
    onBloom,
  };
}

/**
 * Learning-mode scrubber state: the checkpoint timeline pushed by the app plus
 * the "watch it evolve" auto-advance. `onLearningState` is the ViewerApp
 * onLearning callback; `stopEvolve` is for external interruptions (run switch).
 */
export function useLearningMode(appRef: RefObject<ViewerApp | null>) {
  const [learning, setLearning] = useState<LearningState | null>(null);
  const [evolvePlaying, setEvolvePlaying] = useState(false);

  const onLearningState = useCallback((state: LearningState) => {
    setLearning(state);
    if (!state.active) setEvolvePlaying(false);
  }, []);

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
  }, [appRef, learning?.active, evolvePlaying, learning?.index, learning?.frames.length]);

  const onToggleLearning = useCallback(
    (on: boolean) => {
      if (!on) setEvolvePlaying(false);
      void appRef.current?.setLearning(on);
    },
    [appRef],
  );
  const onScrub = useCallback(
    (i: number) => {
      setEvolvePlaying(false);
      void appRef.current?.showFrame(i);
    },
    [appRef],
  );
  const onToggleEvolve = useCallback(() => {
    const starting = !evolvePlaying;
    // Restarting from the end replays the whole arc from the untrained policy.
    if (starting && learning && learning.index >= learning.frames.length - 1) {
      void appRef.current?.showFrame(0);
    }
    setEvolvePlaying(starting);
  }, [appRef, evolvePlaying, learning]);
  const stopEvolve = useCallback(() => setEvolvePlaying(false), []);

  return { learning, evolvePlaying, onLearningState, onToggleLearning, onScrub, onToggleEvolve, stopEvolve };
}
