import { useEffect, useState } from "react";

/** localStorage namespace so persisted UI prefs don't collide with other keys. */
const PREFIX = "rlmon:";

function read<T>(key: string, fallback: T): T {
  if (typeof window === "undefined") return fallback;
  try {
    const raw = window.localStorage.getItem(PREFIX + key);
    if (raw == null) return fallback;
    return JSON.parse(raw) as T;
  } catch {
    return fallback;
  }
}

function write<T>(key: string, value: T): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(PREFIX + key, JSON.stringify(value));
  } catch {
    // Storage can be full or blocked (private mode); persistence is best-effort.
  }
}

/** Drop-in replacement for useState that mirrors the value into localStorage so
 * it survives reloads. Reads synchronously on first render — safe here because
 * the viewer is loaded client-only (ssr:false), so there's no hydration seam. */
export function usePersistentState<T>(
  key: string,
  initial: T,
): [T, React.Dispatch<React.SetStateAction<T>>] {
  const [state, setState] = useState<T>(() => read(key, initial));

  useEffect(() => {
    write(key, state);
  }, [key, state]);

  return [state, setState];
}

/** One-shot read of a persisted value, for wiring initial settings into
 * non-React code (e.g. the ViewerApp constructor). */
export function readPersisted<T>(key: string, fallback: T): T {
  return read(key, fallback);
}
