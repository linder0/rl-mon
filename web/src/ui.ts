import type { EnvGroup, RunAgg, StatsData } from "./types";
import type { LoopStats, ParityResult } from "./loop";
import { renderChart, type Series } from "./chart";

export interface ChartPayload {
  evalOverlay: Series[];
  epLenOverlay: Series[];
  train: Series[];
  approxKl: Series[];
  explainedVar: Series[];
  entropy: Series[];
}

export interface UIHandlers {
  onSelectEnv: (envId: string) => void;
  onPickRun: (runName: string) => void;
  onVariant: (runName: string, variant: "final" | "best") => void;
  onToggleCompare: (runName: string) => void;
  onTogglePlay: (playing: boolean) => void;
  onReset: () => void;
  onSpeed: (speed: number) => void;
  onFollow: (follow: boolean) => void;
  onToggleNet: (collapsed: boolean) => void;
}

const $ = <T extends HTMLElement>(id: string): T =>
  document.getElementById(id) as T;

function fmt(v: number | null | undefined, digits = 0): string {
  if (v === null || v === undefined) return "—";
  return v.toLocaleString(undefined, { maximumFractionDigits: digits });
}

export class UI {
  private playing = true;
  private handlers!: UIHandlers;

  readonly env = $<HTMLSelectElement>("env");
  private readonly playpause = $<HTMLButtonElement>("playpause");
  private readonly reset = $<HTMLButtonElement>("reset");
  private readonly speed = $<HTMLInputElement>("speed");
  private readonly speedVal = $<HTMLElement>("speedval");
  private readonly follow = $<HTMLInputElement>("follow");
  private readonly stats = $<HTMLElement>("stats");
  private readonly parity = $<HTMLElement>("parity");
  private readonly loading = $<HTMLElement>("loading");
  private readonly loadingText = $<HTMLElement>("loading-text");

  private readonly statsPanel = $<HTMLElement>("statspanel");
  private readonly statsToggle = $<HTMLButtonElement>("stats-toggle");
  private readonly netPanel = $<HTMLElement>("netpanel");
  private readonly netToggle = $<HTMLButtonElement>("net-toggle");
  private readonly netValue = $<HTMLElement>("netvalue");
  private readonly spTitle = $<HTMLElement>("sp-title");
  private readonly runTable = $<HTMLElement>("run-table");
  private readonly cards = $<HTMLElement>("summary-cards");
  private readonly configTable = $<HTMLElement>("config-table");

  private readonly chartEls: Record<keyof ChartPayload, HTMLElement> = {
    evalOverlay: $("chart-eval"),
    epLenOverlay: $("chart-eplen"),
    train: $("chart-train"),
    approxKl: $("chart-kl"),
    explainedVar: $("chart-ev"),
    entropy: $("chart-entropy"),
  };
  private readonly chartHeights: Record<keyof ChartPayload, number> = {
    evalOverlay: 160, epLenOverlay: 150, train: 140,
    approxKl: 120, explainedVar: 120, entropy: 120,
  };
  private lastPayload: ChartPayload | null = null;

  bind(handlers: UIHandlers): void {
    this.handlers = handlers;
    this.env.addEventListener("change", () => handlers.onSelectEnv(this.env.value));

    this.playpause.addEventListener("click", () => {
      this.playing = !this.playing;
      this.playpause.textContent = this.playing ? "Pause" : "Play";
      handlers.onTogglePlay(this.playing);
    });

    this.reset.addEventListener("click", () => handlers.onReset());

    this.speed.addEventListener("input", () => {
      const v = parseFloat(this.speed.value);
      this.speedVal.textContent = `${v.toFixed(1)}×`;
      handlers.onSpeed(v);
    });

    this.follow.addEventListener("change", () => handlers.onFollow(this.follow.checked));

    this.statsToggle.addEventListener("click", () => {
      const hidden = this.statsPanel.classList.toggle("collapsed");
      this.statsToggle.textContent = hidden ? "Show" : "Hide";
    });

    this.netToggle.addEventListener("click", () => {
      const collapsed = this.netPanel.classList.toggle("collapsed");
      this.netToggle.textContent = collapsed ? "Show" : "Hide";
      handlers.onToggleNet(collapsed);
    });

    document.addEventListener("keydown", (e) => {
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === "SELECT" || tag === "INPUT") return;
      if (e.code === "Space") { e.preventDefault(); this.playpause.click(); }
      else if (e.code === "KeyR") this.reset.click();
    });

    window.addEventListener("resize", () => this.redrawCharts());
  }

  setEnvs(envs: EnvGroup[], currentEnvId: string): void {
    this.env.innerHTML = "";
    for (const e of envs) {
      const opt = document.createElement("option");
      opt.value = e.env_id;
      opt.textContent = e.env_id;
      if (e.env_id === currentEnvId) opt.selected = true;
      this.env.appendChild(opt);
    }
  }

  /** Render the run table for the current env. */
  renderRunTable(
    runs: RunAgg[],
    live: { runName: string; variant: "final" | "best" },
    compared: Set<string>,
  ): void {
    this.runTable.innerHTML = "";
    const header = document.createElement("div");
    header.className = "run-row head";
    header.innerHTML =
      `<span></span><span>run</span><span>best</span><span>final</span><span>variant</span><span></span>`;
    this.runTable.appendChild(header);

    for (const run of runs) {
      const row = document.createElement("div");
      row.className = "run-row" + (run.runName === live.runName ? " live" : "");

      const swatch = `<span class="rr-swatch" style="background:${run.color}"></span>`;
      const name = `<button class="rr-name" title="${run.runName}">${run.runName}<small>seed ${run.seed ?? "—"}</small></button>`;
      const best = `<span class="rr-metric">${fmt(run.summary.best_eval_mean, 0)}</span>`;
      const fin = `<span class="rr-metric">${fmt(run.summary.final_eval_mean, 0)}</span>`;

      const hasFinal = !!run.variants.final;
      const hasBest = !!run.variants.best;
      const isLive = run.runName === live.runName;
      const seg =
        `<span class="rr-variant seg">` +
        `<button data-v="final" ${!hasFinal ? "disabled" : ""} class="${isLive && live.variant === "final" ? "on" : ""}">F</button>` +
        `<button data-v="best" ${!hasBest ? "disabled" : ""} class="${isLive && live.variant === "best" ? "on" : ""}">B</button>` +
        `</span>`;
      const checked = compared.has(run.runName) || isLive;
      const cmp = `<input type="checkbox" class="rr-compare" ${checked ? "checked" : ""} ${isLive ? "disabled" : ""} title="overlay eval curve" style="accent-color:${run.color}"/>`;

      row.innerHTML = swatch + name + best + fin + seg + cmp;

      row.querySelector(".rr-name")!.addEventListener("click", () =>
        this.handlers.onPickRun(run.runName));
      row.querySelectorAll<HTMLButtonElement>(".rr-variant button").forEach((b) => {
        if (b.disabled) return;
        b.addEventListener("click", (ev) => {
          ev.stopPropagation();
          this.handlers.onVariant(run.runName, b.dataset.v as "final" | "best");
        });
      });
      const box = row.querySelector<HTMLInputElement>(".rr-compare")!;
      box.addEventListener("change", (ev) => {
        ev.stopPropagation();
        this.handlers.onToggleCompare(run.runName);
      });

      this.runTable.appendChild(row);
    }
  }

  renderSummary(run: RunAgg, variant: string): void {
    this.spTitle.textContent = `${run.runName} · ${variant}`;
    const s = run.summary;
    const card = (label: string, value: string, accent = false) =>
      `<div class="card${accent ? " accent" : ""}"><div class="card-v">${value}</div><div class="card-k">${label}</div></div>`;
    this.cards.innerHTML =
      card("best eval", fmt(s.best_eval_mean, 0), true) +
      card("final eval", fmt(s.final_eval_mean, 0)) +
      card("ep length", fmt(s.eval_ep_len_final, 0)) +
      card("timesteps", fmt(s.timesteps, 0)) +
      card("seed", s.seed === null || s.seed === undefined ? "—" : String(s.seed));
  }

  renderCharts(payload: ChartPayload): void {
    this.lastPayload = payload;
    (Object.keys(this.chartEls) as (keyof ChartPayload)[]).forEach((key) => {
      renderChart(this.chartEls[key], {
        series: payload[key],
        height: this.chartHeights[key],
      });
    });
  }

  renderConfig(stats: StatsData): void {
    const hp = (stats.config.hyperparameters ?? {}) as Record<string, unknown>;
    const args = (stats.config.args ?? {}) as Record<string, unknown>;
    const netArch = (hp.policy_kwargs as { net_arch?: unknown })?.net_arch;
    const rows: [string, string][] = [
      ["env", stats.config.env_id ?? "—"],
      ["algorithm", "PPO"],
      ["learning rate", String(hp.learning_rate ?? "—")],
      ["n_steps", String(hp.n_steps ?? "—")],
      ["batch size", String(hp.batch_size ?? "—")],
      ["n_epochs", String(hp.n_epochs ?? "—")],
      ["gamma", String(hp.gamma ?? "—")],
      ["gae_lambda", String(hp.gae_lambda ?? "—")],
      ["clip range", String(hp.clip_range ?? "—")],
      ["ent_coef", String(hp.ent_coef ?? "—")],
      ["net arch", Array.isArray(netArch) ? `[${netArch.join(", ")}]` : "—"],
      ["n_envs", String(args.n_envs ?? "—")],
      ["device", stats.config.device ?? "—"],
    ];
    this.configTable.innerHTML = rows
      .map(([k, v]) => `<div class="cfg-k">${k}</div><div class="cfg-v">${v}</div>`)
      .join("");
  }

  private redrawCharts(): void {
    if (this.lastPayload) this.renderCharts(this.lastPayload);
  }

  renderLiveStats(s: LoopStats): void {
    this.stats.innerHTML = `
      <span class="k">Episode</span><span class="v">${s.episode}</span>
      <span class="k">Step</span><span class="v">${s.step}</span>
      <span class="k">Distance</span><span class="v">${s.distance.toFixed(2)} m</span>
      <span class="k">State</span><span class="v" style="color:${
        s.healthy ? "var(--good)" : "var(--bad)"
      }">${s.healthy ? "healthy" : "fallen"}</span>
      <span class="k">FPS</span><span class="v">${s.fps.toFixed(0)}</span>`;
  }

  setParity(result: ParityResult): void {
    const ok = result.maxActionError < 1e-3;
    this.parity.className = `parity ${ok ? "ok" : "bad"}`;
    this.parity.textContent = ok
      ? `parity ok — matches Python (Δ ${result.maxActionError.toExponential(1)})`
      : `parity warning — Δ ${result.maxActionError.toExponential(1)}`;
  }

  setValue(v: number): void {
    this.netValue.textContent = v.toFixed(2);
  }

  setLoading(text: string | null): void {
    if (text === null) this.loading.classList.add("hidden");
    else {
      this.loading.classList.remove("hidden");
      this.loadingText.textContent = text;
    }
  }

  showError(message: string): void {
    this.loading.classList.remove("hidden");
    this.loading.innerHTML = `<div class="error-box">${message}</div>`;
  }
}
