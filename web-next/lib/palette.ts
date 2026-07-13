/**
 * Canonical accent palette. Single source of truth for the hex colors used by
 * run swatches, chart lines, and diagnostic series.
 *
 * These MIRROR the `--chart-*` oklch tokens in `globals.css` — the two must be
 * kept in sync by hand. CSS authors colors in oklch (for the DOM chrome), while
 * uPlot canvases and the three.js scene need concrete hex, so the same palette
 * is expressed in both places. Order matches --chart-1..5 for the first five.
 */
export const ACCENT = {
  blue: "#6ea8fe", // --chart-1 (brand)
  purple: "#b78bff", // --chart-2
  green: "#59d499", // --chart-3
  orange: "#ffb454", // --chart-4
  red: "#ff6b6b", // --chart-5
  cyan: "#4dd0e1",
  pink: "#f06292",
  lime: "#aed581",
} as const;

/** Ordered palette for assigning a distinct color to each run in a comparison. */
export const RUN_PALETTE: readonly string[] = [
  ACCENT.blue,
  ACCENT.purple,
  ACCENT.green,
  ACCENT.orange,
  ACCENT.red,
  ACCENT.cyan,
  ACCENT.pink,
  ACCENT.lime,
];

/** Fixed colors for the training-diagnostic curves (stable across runs). */
export const SERIES_COLORS = {
  reward: ACCENT.purple,
  approxKl: ACCENT.orange,
  explainedVar: ACCENT.green,
  entropy: ACCENT.cyan,
} as const;
