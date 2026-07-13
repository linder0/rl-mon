"use client";

import { Button } from "@/components/ui/button";

export type PanelView = "stats" | "dials" | "scene";

const VIEWS: { id: PanelView; label: string }[] = [
  { id: "stats", label: "Stats" },
  { id: "dials", label: "Dials" },
  { id: "scene", label: "UI" },
];

/** Segmented Stats | Dials | UI switch shared by the right-panel views. */
export function PanelViewToggle({
  view,
  onSetView,
}: {
  view: PanelView;
  onSetView: (v: PanelView) => void;
}) {
  return (
    <div className="inline-flex rounded-md bg-muted/60 p-0.5">
      {VIEWS.map((v) => {
        const active = view === v.id;
        return (
          <Button
            key={v.id}
            size="sm"
            variant={active ? "default" : "ghost"}
            className={
              active
                ? "h-6 px-2 text-label"
                : "h-6 px-2 text-label text-muted-foreground hover:text-foreground"
            }
            onClick={() => onSetView(v.id)}
          >
            {v.label}
          </Button>
        );
      })}
    </div>
  );
}
