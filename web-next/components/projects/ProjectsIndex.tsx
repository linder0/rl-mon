"use client";

import Link from "next/link";
import { Loader2 } from "lucide-react";
import { SiteHeader } from "@/components/SiteHeader";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { aggregateRuns, sortRunsByEval } from "@/lib/catalog";
import { useCatalog } from "@/lib/hooks";
import type { EnvGroup } from "@/lib/types";
import { fmt, fmtSteps } from "@/lib/format";

function Metric({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return (
    <div>
      <div className={`text-lg font-semibold tabular-nums ${accent ? "text-primary" : ""}`}>
        {value}
      </div>
      <div className="text-nano uppercase tracking-wide text-muted-foreground">{label}</div>
    </div>
  );
}

function ProjectCard({ env }: { env: EnvGroup }) {
  const runs = sortRunsByEval(aggregateRuns(env));
  const best = runs[0];

  return (
    <Link href={`/p/${env.label}`} className="group">
      <Card className="h-full gap-3 p-5 transition-colors group-hover:border-primary/50">
        <div className="flex items-start justify-between gap-2">
          <div>
            <div className="text-base font-semibold tracking-tight group-hover:text-primary">
              {env.label}
            </div>
            <div className="text-label text-muted-foreground">{env.env_id}</div>
          </div>
          <Badge variant="secondary" className="shrink-0">
            {runs.length} {runs.length === 1 ? "iteration" : "iterations"}
          </Badge>
        </div>

        <div className="grid grid-cols-3 gap-2">
          <Metric label="best eval" value={fmt(best?.summary.best_eval_mean)} accent />
          <Metric label="timesteps" value={fmtSteps(best?.summary.timesteps)} />
          <Metric label="ep len" value={fmt(best?.summary.eval_ep_len_final)} />
        </div>

        {best && (
          <div className="truncate text-micro text-muted-foreground">
            best: {best.runName}
          </div>
        )}
      </Card>
    </Link>
  );
}

export function ProjectsIndex() {
  const { index, error } = useCatalog();

  return (
    <div className="min-h-screen bg-background">
      <SiteHeader />
      <main className="mx-auto max-w-6xl px-4 py-8">
        <div className="mb-6">
          <h1 className="text-xl font-semibold tracking-tight">Projects</h1>
          <p className="text-sm text-muted-foreground">
            One project per environment — open one to watch it live and compare its
            training iterations.
          </p>
        </div>

        {error ? (
          <div className="rounded-lg border border-destructive/40 bg-destructive/10 p-5 text-sm text-destructive">
            {error}
          </div>
        ) : !index ? (
          <div className="flex items-center gap-3 text-sm text-muted-foreground">
            <Loader2 className="size-4 animate-spin" /> Loading catalog…
          </div>
        ) : (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {index.envs.map((env) => (
              <ProjectCard key={env.env_id} env={env} />
            ))}
          </div>
        )}
      </main>
    </div>
  );
}
