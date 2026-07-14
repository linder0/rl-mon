/** Shared number formatting for run summaries and controls. (uPlot's axis/
 * tooltip formatters live with the chart — different precision rules.) */

/** Locale-formatted number, em-dash for missing values. */
export function fmt(v: number | null | undefined, digits = 0): string {
  if (v === null || v === undefined) return "—";
  return v.toLocaleString(undefined, { maximumFractionDigits: digits });
}

/** Compact step counts: 3_002_368 → "3.0M", 40_960 → "41k". */
export function fmtSteps(n: number | null | undefined): string {
  if (n === null || n === undefined) return "—";
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(n % 1_000_000 ? 1 : 0)}M`;
  if (n >= 1_000) return `${Math.round(n / 1_000)}k`;
  return String(n);
}
