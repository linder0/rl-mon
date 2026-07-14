"use client";

import { useEffect, useMemo, useRef } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";
import type { Series } from "@/lib/chartDefs";

function si(v: number): string {
  const a = Math.abs(v);
  if (a >= 1e9) return `${(v / 1e9).toFixed(1)}B`;
  if (a >= 1e6) return `${(v / 1e6).toFixed(1)}M`;
  if (a >= 1e3) return `${(v / 1e3).toFixed(0)}k`;
  if (a > 0 && a < 1) return v.toFixed(2);
  return `${Math.round(v)}`;
}

/** Precise-ish value for the hover tooltip (keeps small numbers readable). */
function fmtVal(v: number): string {
  const a = Math.abs(v);
  if (a >= 1e4) return si(v);
  if (a === 0) return "0";
  if (a < 1) return v.toFixed(3);
  if (a < 100) return v.toFixed(2);
  return v.toFixed(0);
}

function hexToRgba(hex: string, alpha: number): string {
  const h = hex.replace("#", "");
  const r = parseInt(h.slice(0, 2), 16);
  const g = parseInt(h.slice(2, 4), 16);
  const b = parseInt(h.slice(4, 6), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

const AXIS_FONT = "9px Inter, system-ui, sans-serif";

/** Read a CSS custom property off the document root (falls back to a dark-theme
 * default) so the on-canvas uPlot axes match the current UI theme. */
function cssVar(name: string, fallback: string): string {
  if (typeof window === "undefined") return fallback;
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

/** A uPlot line chart that renders one or more overlaid series, each with its
 * own (possibly differing) x axis merged onto a shared timeline, plus an
 * optional shaded +/- band on any series. A custom React legend matches the
 * app's styling. Re-renders on `theme` so the axis colors follow the UI. */
export function UPlotChart({
  series,
  height = 140,
  theme = "dark",
}: {
  series: Series[];
  height?: number;
  theme?: string;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const active = useMemo(() => series.filter((s) => s.x.length > 0), [series]);

  useEffect(() => {
    const container = ref.current;
    if (!container) return;
    container.innerHTML = "";
    if (active.length === 0) return;

    const grid = cssVar("--grid-line", "rgba(255,255,255,0.07)");
    const tickLabel = cssVar("--tick", "rgba(255,255,255,0.45)");
    const isDark = theme !== "light";
    const tipBg = isDark ? "rgba(17,21,29,0.96)" : "rgba(255,255,255,0.98)";
    const tipFg = isDark ? "#e6e9ef" : "#1a1f29";
    const tipBorder = isDark ? "rgba(255,255,255,0.12)" : "rgba(0,0,0,0.12)";

    // Merge every series' x values onto one sorted, unique timeline; align each
    // series' y (and band) to it, leaving null gaps where a run has no sample.
    const xset = new Set<number>();
    for (const s of active) for (const v of s.x) xset.add(v);
    const xs = [...xset].sort((a, b) => a - b);
    const alignTo = (sx: number[], sy: number[]): (number | null)[] => {
      const map = new Map<number, number>();
      for (let i = 0; i < sx.length; i++) map.set(sx[i], sy[i]);
      return xs.map((x) => (map.has(x) ? map.get(x)! : null));
    };

    const data: (number | null)[][] = [xs];
    const uSeries: uPlot.Series[] = [{}];
    const bands: uPlot.Band[] = [];
    // Track which data columns are real (visible) lines, for the hover tooltip.
    const lineCols: { di: number; label: string; color: string }[] = [];

    for (const s of active) {
      lineCols.push({ di: data.length, label: s.name, color: s.color });
      data.push(alignTo(s.x, s.y));
      uSeries.push({
        label: s.name,
        stroke: s.color,
        width: 1.8,
        points: { show: false },
      });
      if (s.band) {
        const hiIdx = data.length;
        data.push(alignTo(s.x, s.band.hi));
        uSeries.push({ stroke: "transparent", width: 0, points: { show: false } });
        const loIdx = data.length;
        data.push(alignTo(s.x, s.band.lo));
        uSeries.push({ stroke: "transparent", width: 0, points: { show: false } });
        bands.push({ series: [hiIdx, loIdx], fill: hexToRgba(s.color, 0.14) });
      }
    }

    // Floating tooltip that reads out the step + each series value on hover.
    const tip = document.createElement("div");
    tip.style.cssText = [
      "position:absolute", "pointer-events:none", "z-index:var(--z-tooltip)",
      "padding:5px 7px", "border-radius:6px", "display:none",
      "font:10px Inter,system-ui,sans-serif", "line-height:1.35",
      "white-space:nowrap", `background:${tipBg}`, `color:${tipFg}`,
      `border:1px solid ${tipBorder}`, "box-shadow:0 3px 12px rgba(0,0,0,0.28)",
    ].join(";");

    const showMultiple = lineCols.length > 1;
    const tooltipPlugin: uPlot.Plugin = {
      hooks: {
        init: (u) => u.over.appendChild(tip),
        setCursor: (u) => {
          const idx = u.cursor.idx;
          const left = u.cursor.left ?? -10;
          if (idx == null || left < 0) {
            tip.style.display = "none";
            return;
          }
          const xv = u.data[0][idx] as number;
          let anchorY = Infinity;
          const rows: string[] = [`<div style="opacity:0.6">step ${si(xv)}</div>`];
          for (const c of lineCols) {
            const v = u.data[c.di][idx] as number | null;
            if (v == null) continue;
            const y = u.valToPos(v, "y");
            if (y < anchorY) anchorY = y;
            const swatch = `<span style="display:inline-block;width:8px;height:2px;background:${c.color};vertical-align:middle;margin-right:5px"></span>`;
            const name = showMultiple ? `${c.label}: ` : "";
            rows.push(
              `<div>${swatch}${name}<b style="font-weight:600">${fmtVal(v)}</b></div>`,
            );
          }
          if (rows.length === 1) {
            tip.style.display = "none";
            return;
          }
          tip.innerHTML = rows.join("");
          tip.style.display = "block";
          const w = u.over.clientWidth;
          const dx = left < w * 0.5 ? "8px" : "calc(-100% - 8px)";
          tip.style.left = `${left}px`;
          tip.style.top = `${Number.isFinite(anchorY) ? anchorY : 8}px`;
          tip.style.transform = `translate(${dx}, -50%)`;
        },
      },
    };

    const opts: uPlot.Options = {
      width: container.clientWidth || 300,
      height,
      legend: { show: false },
      cursor: {
        y: false,
        drag: { x: false, y: false },
        points: { show: true, size: 6 },
      },
      plugins: [tooltipPlugin],
      scales: { x: { time: false } },
      padding: [8, 8, 0, 0],
      axes: [
        {
          stroke: tickLabel,
          grid: { stroke: grid, width: 1 },
          ticks: { show: false },
          font: AXIS_FONT,
          size: 24,
          values: (_u, vals) => vals.map(si),
        },
        {
          stroke: tickLabel,
          grid: { stroke: grid, width: 1 },
          ticks: { show: false },
          font: AXIS_FONT,
          size: 38,
          values: (_u, vals) => vals.map(si),
        },
      ],
      series: uSeries,
      bands,
    };

    const plot = new uPlot(opts, data as uPlot.AlignedData, container);
    const ro = new ResizeObserver(() => {
      const w = container.clientWidth;
      if (w > 0) plot.setSize({ width: w, height });
    });
    ro.observe(container);

    return () => {
      ro.disconnect();
      plot.destroy();
    };
  }, [active, height, theme]);

  if (active.length === 0) {
    return (
      <div
        className="flex items-center justify-center rounded-md bg-muted/40 text-xs text-muted-foreground"
        style={{ height }}
      >
        no data
      </div>
    );
  }

  const showLegend = active.length > 1 || !!active[0].name;
  return (
    <>
      <div ref={ref} className="overflow-hidden rounded-md bg-muted/30" />
      {showLegend && (
        <div className="mt-1 flex flex-wrap gap-x-3 gap-y-0.5 px-1 text-micro text-muted-foreground">
          {active.map((s) => (
            <span key={s.name} className="inline-flex items-center gap-1">
              <i
                className="inline-block h-[3px] w-2.5 rounded-full"
                style={{ background: s.color }}
              />
              {s.name}
            </span>
          ))}
        </div>
      )}
    </>
  );
}
