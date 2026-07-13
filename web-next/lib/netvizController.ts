import type { PolicyNet } from "./net";
import type { Theme } from "./theme";

const PAD_X = 30;
const PAD_TOP = 10;
const PAD_BOTTOM = 18; // room for column labels
const MAX_NODE_R = 7;
const MAX_NODES_PER_COL = 64; // subsample bigger layers so it stays legible + fast
const EDGE_THRESHOLD = 0.06; // skip near-silent connections
const HOVER_RADIUS = 14; // px within which a node is "hovered"

export interface NetVizLabels {
  obs?: string[];
  act?: string[];
  /** Overrides the last column's label when it is a single scalar (critic). */
  output?: string;
}

/** Draws one network (actor or critic) and lights up neurons + the connections
 * carrying signal, updated every control step from externally-supplied
 * activations. Column 0 is the (normalized) observation, the last column is the
 * output; columns between are hidden layers. Layers larger than
 * MAX_NODES_PER_COL are shown as an evenly-spaced sample (labels report the true
 * count). Hovering a neuron shows its name and current value. */
export class NetVizController {
  private readonly canvas: HTMLCanvasElement;
  private readonly ctx: CanvasRenderingContext2D;
  private readonly tip: HTMLDivElement;
  private dpr = 1;

  private net: PolicyNet | null = null;
  private obs: Float32Array | null = null; // owned copy of the latest observation
  private acts: Float32Array[] | null = null;
  private labels: NetVizLabels = {};
  private maxW: number[] = []; // per-layer max |weight|, for edge normalization

  /** Called once per rendered frame with the scalar output (used by the critic
   * to surface its value estimate). */
  onValue: ((v: number) => void) | null = null;

  private cssW = 0;
  private cssH = 0;
  private xs: number[] = []; // column x positions
  private ys: Float32Array[] = []; // displayed-node y positions per column
  private idx: Int32Array[] = []; // displayed-node -> real neuron index per column
  private nodeR = 3;
  private colLabels: string[] = [];

  private dirty = false;
  private collapsed = false;
  private running = false;
  private disposed = false;
  private theme: Theme = "dark";

  private readonly onHoverBound: (e: MouseEvent) => void;
  private readonly onLeaveBound: () => void;
  private readonly resizeObserver: ResizeObserver;

  constructor(canvas: HTMLCanvasElement, theme: Theme = "dark") {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d")!;
    this.theme = theme;

    this.tip = document.createElement("div");
    this.tip.className = "net-tip";
    this.tip.style.display = "none";
    canvas.parentElement?.appendChild(this.tip);

    this.onHoverBound = (e) => this.onHover(e);
    this.onLeaveBound = () => (this.tip.style.display = "none");
    canvas.addEventListener("mousemove", this.onHoverBound);
    canvas.addEventListener("mouseleave", this.onLeaveBound);
    // Re-layout whenever the canvas's own box changes size (panel resize, flex
    // reflow, show/hide) so the drawing always fits its column exactly.
    this.resizeObserver = new ResizeObserver(() => this.layout());
    this.resizeObserver.observe(canvas);
  }

  setNet(net: PolicyNet | null, labels: NetVizLabels = {}): void {
    this.net = net;
    this.labels = labels;
    this.acts = null;
    // Weights are constant for a given net, so normalize edges against a
    // precomputed per-layer max instead of rescanning every frame.
    this.maxW = net
      ? net.layers.map((l) => {
          let m = 1e-6;
          for (const row of l.w) for (const x of row) m = Math.max(m, Math.abs(x));
          return m;
        })
      : [];
    this.layout();
    this.dirty = true;
    this.startLoop();
  }

  setCollapsed(collapsed: boolean): void {
    this.collapsed = collapsed;
    if (!collapsed) {
      this.layout();
      this.dirty = true;
    }
  }

  /** Re-theme the diagram (resting-neuron tone + label color follow the UI so
   * the network doesn't render dark-on-light). Triggers a redraw. */
  setTheme(theme: Theme): void {
    if (theme === this.theme) return;
    this.theme = theme;
    this.dirty = true;
    this.startLoop();
  }

  /** Resting-neuron / edge blend target: dark in dark mode, light in light mode
   * so idle neurons recede into the card and active ones pop. */
  private neutral(): number {
    return this.theme === "light" ? 202 : 46;
  }

  private mutedText(): string {
    return this.theme === "light" ? "rgba(90,100,120,0.9)" : "rgba(139,147,167,0.9)";
  }

  /** Supply the latest raw observation. Copied because callers reuse the buffer
   * across steps; the forward pass runs at most once per frame in the draw tick,
   * so feeding this several times per frame (physics catch-up) is nearly free. */
  setObs(obs: Float32Array): void {
    if (!this.obs || this.obs.length !== obs.length) this.obs = new Float32Array(obs.length);
    this.obs.set(obs);
    this.dirty = true;
  }

  dispose(): void {
    this.disposed = true;
    this.resizeObserver.disconnect();
    this.canvas.removeEventListener("mousemove", this.onHoverBound);
    this.canvas.removeEventListener("mouseleave", this.onLeaveBound);
    this.tip.remove();
  }

  private startLoop(): void {
    if (this.running) return;
    this.running = true;
    const tick = (): void => {
      if (this.disposed) return;
      if (this.dirty && !this.collapsed) {
        this.dirty = false;
        if (this.net && this.obs) {
          this.acts = this.net.forward(this.obs);
          const last = this.acts[this.acts.length - 1];
          if (this.onValue && last.length === 1) this.onValue(last[0]);
        }
        this.draw();
      }
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }

  private labelFor(col: number, realIdx: number): string {
    const last = this.net!.sizes.length - 1;
    if (col === 0) return this.labels.obs?.[realIdx] ?? `obs #${realIdx}`;
    if (col === last) {
      if (this.labels.output && this.net!.sizes[last] === 1) return this.labels.output;
      return this.labels.act?.[realIdx] ?? `out #${realIdx}`;
    }
    return `h${col} #${realIdx}`;
  }

  private layout(): void {
    // Size to the canvas's own CSS box (it fills its column via width/height
    // 100%), so the diagram never overflows the panel frame.
    const cssW = this.canvas.clientWidth;
    const cssH = this.canvas.clientHeight;
    if (cssW === 0 || cssH === 0) return;
    this.cssW = cssW;
    this.cssH = cssH;

    this.dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.canvas.width = Math.round(cssW * this.dpr);
    this.canvas.height = Math.round(cssH * this.dpr);
    this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);

    if (!this.net) return;
    const sizes = this.net.sizes;
    const nCols = sizes.length;

    this.xs = sizes.map((_, i) =>
      nCols === 1 ? cssW / 2 : PAD_X + (i * (cssW - 2 * PAD_X)) / (nCols - 1),
    );

    // Evenly-spaced sample of neuron indices for oversized columns.
    this.idx = sizes.map((count) => {
      const shown = Math.min(count, MAX_NODES_PER_COL);
      const arr = new Int32Array(shown);
      for (let i = 0; i < shown; i++) {
        arr[i] = shown === count ? i : Math.round((i * (count - 1)) / (shown - 1));
      }
      return arr;
    });

    const top = PAD_TOP;
    const bottom = cssH - PAD_BOTTOM;
    this.ys = this.idx.map((ids) => {
      const shown = ids.length;
      const ys = new Float32Array(shown);
      for (let i = 0; i < shown; i++) {
        ys[i] = shown === 1 ? (top + bottom) / 2 : top + (i * (bottom - top)) / (shown - 1);
      }
      return ys;
    });

    const maxShown = Math.max(...this.idx.map((a) => a.length));
    const vSpacing = (bottom - top) / Math.max(maxShown - 1, 1);
    const hSpacing = nCols > 1 ? (cssW - 2 * PAD_X) / (nCols - 1) : cssW;
    this.nodeR = Math.max(1.3, Math.min(MAX_NODE_R, vSpacing * 0.42, hSpacing * 0.22));

    this.colLabels = sizes.map((count, i) => {
      const name = i === 0 ? "obs" : i === nCols - 1 ? "out" : `h${i}`;
      return `${name}·${count}`;
    });
    this.dirty = true;
  }

  private draw(): void {
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.cssW, this.cssH);
    const net = this.net;
    if (!net) {
      // No network for this run (e.g. the critic wasn't exported): say so
      // instead of leaving a mysterious blank panel.
      if (this.cssW > 0 && this.cssH > 0) {
        ctx.fillStyle = this.mutedText();
        ctx.font = "11px Inter, system-ui, sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText("network not exported", this.cssW / 2, this.cssH / 2);
      }
      return;
    }
    // Layout may not have run yet (e.g. the canvas reported 0 size on the first
    // paint and layout() bailed early). Skip drawing until the per-column arrays
    // match the current network; the ResizeObserver re-runs layout once sized.
    if (this.xs.length !== net.sizes.length || this.idx.length !== net.sizes.length) return;
    const acts = this.acts;

    const scale = net.sizes.map((count, c) => {
      if (!acts) return 1;
      let m = 1e-6;
      const a = acts[c];
      for (let i = 0; i < count; i++) m = Math.max(m, Math.abs(a[i]));
      return m;
    });
    const val = (c: number, i: number): number => (acts ? acts[c][i] / scale[c] : 0);

    const neutral = this.neutral();

    // Connections: brightness ~ |sourceActivation * normalizedWeight|.
    ctx.lineWidth = 1;
    for (let li = 0; li < net.layers.length; li++) {
      const layer = net.layers[li];
      const maxW = this.maxW[li];
      const srcIds = this.idx[li];
      const dstIds = this.idx[li + 1];
      const srcX = this.xs[li];
      const dstX = this.xs[li + 1];
      const srcYs = this.ys[li];
      const dstYs = this.ys[li + 1];
      for (let dj = 0; dj < dstIds.length; dj++) {
        const j = dstIds[dj];
        const row = layer.w[j];
        const dy = dstYs[dj];
        for (let dk = 0; dk < srcIds.length; dk++) {
          const k = srcIds[dk];
          const signal = val(li, k) * (row[k] / maxW);
          const intensity = Math.abs(signal);
          if (intensity < EDGE_THRESHOLD) continue;
          ctx.strokeStyle = edgeColor(signal, intensity);
          ctx.beginPath();
          ctx.moveTo(srcX, srcYs[dk]);
          ctx.lineTo(dstX, dy);
          ctx.stroke();
        }
      }
    }

    // Neurons — drawn on top, with a glow that grows with activation.
    for (let c = 0; c < net.sizes.length; c++) {
      const ids = this.idx[c];
      const ys = this.ys[c];
      const x = this.xs[c];
      for (let di = 0; di < ids.length; di++) {
        const v = val(c, ids[di]);
        const a = Math.min(Math.abs(v), 1);
        const color = nodeColor(v, neutral);
        ctx.shadowBlur = 10 * a;
        ctx.shadowColor = color;
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(x, ys[di], this.nodeR * (1 + 0.5 * a), 0, Math.PI * 2);
        ctx.fill();
      }
    }
    ctx.shadowBlur = 0;

    // Column labels. On narrow canvases the columns sit close together, so the
    // labels would overlap into an unreadable strip and the end ones would clip
    // off the edges. Clamp each label inside the canvas and draw greedily,
    // skipping any that would collide — endpoints (obs/out) get priority since
    // they're the most meaningful.
    ctx.fillStyle = this.mutedText();
    ctx.font = "10px Inter, system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "alphabetic";
    const y = this.cssH - 5;
    const n = this.colLabels.length;
    const widths = this.colLabels.map((t) => ctx.measureText(t).width);
    const GAP = 5;
    const placed: { lo: number; hi: number }[] = [];
    const tryDraw = (c: number): void => {
      const half = widths[c] / 2;
      const cx = Math.max(half + 1, Math.min(this.cssW - half - 1, this.xs[c]));
      const lo = cx - half;
      const hi = cx + half;
      if (!placed.every((p) => hi + GAP <= p.lo || lo - GAP >= p.hi)) return;
      placed.push({ lo, hi });
      ctx.fillText(this.colLabels[c], cx, y);
    };
    tryDraw(0);
    if (n > 1) tryDraw(n - 1);
    for (let c = 1; c < n - 1; c++) tryDraw(c);
  }

  private onHover(e: MouseEvent): void {
    if (!this.net || this.collapsed) return;
    const rect = this.canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    let best = { d2: HOVER_RADIUS * HOVER_RADIUS, col: -1, realIdx: -1 };
    for (let c = 0; c < this.xs.length; c++) {
      const dx = mx - this.xs[c];
      if (Math.abs(dx) > HOVER_RADIUS) continue;
      const ys = this.ys[c];
      const ids = this.idx[c];
      for (let di = 0; di < ys.length; di++) {
        const dy = my - ys[di];
        const d2 = dx * dx + dy * dy;
        if (d2 < best.d2) best = { d2, col: c, realIdx: ids[di] };
      }
    }

    if (best.col < 0) {
      this.tip.style.display = "none";
      return;
    }
    const raw = this.acts ? this.acts[best.col][best.realIdx] : 0;
    this.tip.innerHTML =
      `<b>${this.labelFor(best.col, best.realIdx)}</b><span>${raw.toFixed(3)}</span>`;
    this.tip.style.display = "flex";
    this.tip.style.left = `${mx + 12}px`;
    this.tip.style.top = `${my + 12}px`;
  }
}

// Diverging palette: warm = positive activation/signal, cool = negative.
const WARM = [255, 150, 70];
const COOL = [110, 168, 254];

/** Node fill: blends from the theme's neutral tone toward the signed color by
 * |value| (neutral is dark in dark mode, light in light mode). */
function nodeColor(t: number, neutral: number): string {
  const a = Math.min(Math.abs(t), 1);
  const [tr, tg, tb] = t >= 0 ? WARM : COOL;
  const r = Math.round(neutral + (tr - neutral) * a);
  const g = Math.round(neutral + (tg - neutral) * a);
  const b = Math.round(neutral + (tb - neutral) * a);
  return `rgb(${r},${g},${b})`;
}

/** Edge stroke: signed color at an opacity that grows with signal strength. */
function edgeColor(signal: number, intensity: number): string {
  const [r, g, b] = signal >= 0 ? WARM : COOL;
  const alpha = 0.03 + 0.32 * Math.min(intensity, 1);
  return `rgba(${r},${g},${b},${alpha})`;
}
