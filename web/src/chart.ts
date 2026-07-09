// Tiny dependency-free SVG line chart with optional shaded band and legend.

const SVGNS = "http://www.w3.org/2000/svg";

export interface Series {
  name: string;
  color: string;
  x: number[];
  y: number[];
  /** Optional symmetric band (e.g. +/- std) drawn behind the line. */
  band?: { lo: number[]; hi: number[] };
}

export interface ChartOptions {
  series: Series[];
  height?: number;
  yLabel?: string;
  xLabel?: string;
  /** Format for x tick labels; defaults to compact SI (e.g. 1.2M). */
  xFormat?: (v: number) => string;
  yFormat?: (v: number) => string;
}

const PAD = { top: 10, right: 12, bottom: 26, left: 46 };

function si(v: number): string {
  const a = Math.abs(v);
  if (a >= 1e9) return `${(v / 1e9).toFixed(1)}B`;
  if (a >= 1e6) return `${(v / 1e6).toFixed(1)}M`;
  if (a >= 1e3) return `${(v / 1e3).toFixed(0)}k`;
  return `${Math.round(v)}`;
}

function el(tag: string, attrs: Record<string, string | number>): SVGElement {
  const node = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, String(v));
  return node;
}

function niceExtent(min: number, max: number): [number, number] {
  if (min === max) return [min - 1, max + 1];
  const pad = (max - min) * 0.06;
  return [min - pad, max + pad];
}

/** Render (or re-render) a chart into `container`. Clears prior contents. */
export function renderChart(container: HTMLElement, opts: ChartOptions): void {
  container.innerHTML = "";
  const width = container.clientWidth || 300;
  const height = opts.height ?? 150;
  const xf = opts.xFormat ?? si;
  const yf = opts.yFormat ?? si;

  const active = opts.series.filter((s) => s.x.length > 0);
  if (active.length === 0) {
    const empty = document.createElement("div");
    empty.className = "chart-empty";
    empty.textContent = "no data";
    container.appendChild(empty);
    return;
  }

  let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
  for (const s of active) {
    for (const v of s.x) { xMin = Math.min(xMin, v); xMax = Math.max(xMax, v); }
    for (const v of s.y) { yMin = Math.min(yMin, v); yMax = Math.max(yMax, v); }
    if (s.band) {
      for (const v of s.band.lo) yMin = Math.min(yMin, v);
      for (const v of s.band.hi) yMax = Math.max(yMax, v);
    }
  }
  [yMin, yMax] = niceExtent(yMin, yMax);

  const iw = width - PAD.left - PAD.right;
  const ih = height - PAD.top - PAD.bottom;
  const sx = (v: number) => PAD.left + (xMax === xMin ? 0 : (v - xMin) / (xMax - xMin)) * iw;
  const sy = (v: number) => PAD.top + (1 - (yMax === yMin ? 0.5 : (v - yMin) / (yMax - yMin))) * ih;

  const svg = el("svg", {
    viewBox: `0 0 ${width} ${height}`,
    width: "100%",
    height,
    preserveAspectRatio: "none",
  });

  // Horizontal gridlines + y ticks.
  const yticks = 3;
  for (let i = 0; i <= yticks; i++) {
    const yv = yMin + (i / yticks) * (yMax - yMin);
    const y = sy(yv);
    svg.appendChild(el("line", {
      x1: PAD.left, y1: y, x2: width - PAD.right, y2: y,
      stroke: "rgba(255,255,255,0.07)", "stroke-width": 1,
    }));
    const t = el("text", {
      x: PAD.left - 6, y: y + 3, "text-anchor": "end",
      "font-size": 9, fill: "rgba(255,255,255,0.45)",
    });
    t.textContent = yf(yv);
    svg.appendChild(t);
  }

  // X ticks (first / mid / last).
  for (const xv of [xMin, (xMin + xMax) / 2, xMax]) {
    const t = el("text", {
      x: sx(xv), y: height - 8, "text-anchor": "middle",
      "font-size": 9, fill: "rgba(255,255,255,0.45)",
    });
    t.textContent = xf(xv);
    svg.appendChild(t);
  }

  for (const s of active) {
    if (s.band) {
      const pts: string[] = [];
      for (let i = 0; i < s.x.length; i++) pts.push(`${sx(s.x[i])},${sy(s.band.hi[i])}`);
      for (let i = s.x.length - 1; i >= 0; i--) pts.push(`${sx(s.x[i])},${sy(s.band.lo[i])}`);
      svg.appendChild(el("polygon", { points: pts.join(" "), fill: s.color, "fill-opacity": 0.14, stroke: "none" }));
    }
    const line = s.x.map((xv, i) => `${sx(xv)},${sy(s.y[i])}`).join(" ");
    svg.appendChild(el("polyline", {
      points: line, fill: "none", stroke: s.color,
      "stroke-width": 1.8, "stroke-linejoin": "round", "stroke-linecap": "round",
    }));
  }

  container.appendChild(svg);

  if (active.length > 1 || active[0].name) {
    const legend = document.createElement("div");
    legend.className = "chart-legend";
    for (const s of active) {
      const item = document.createElement("span");
      item.innerHTML = `<i style="background:${s.color}"></i>${s.name}`;
      legend.appendChild(item);
    }
    container.appendChild(legend);
  }
}
