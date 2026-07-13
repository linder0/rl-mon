"use client";

import { memo, type RefObject } from "react";
import { Panel } from "@/components/ui/panel";
import { Button } from "@/components/ui/button";

interface NetPanelProps {
  actorRef: RefObject<HTMLCanvasElement | null>;
  criticRef: RefObject<HTMLCanvasElement | null>;
  collapsed: boolean;
  onToggle: () => void;
  value: number | null;
}

function NetPanelImpl(props: NetPanelProps) {
  return (
    <Panel size="xl">
      <div className="flex flex-col gap-2 p-3">
        <div className="flex items-center justify-between gap-2">
          <div className="text-label font-medium">
            Policy network <span className="text-muted-foreground">live activations</span>
          </div>
          <Button
            variant="ghost"
            size="sm"
            className="h-6 px-2 text-xs"
            onClick={props.onToggle}
          >
            {props.collapsed ? "Show" : "Hide"}
          </Button>
        </div>

        {/* Kept mounted (hidden via `hidden` when collapsed) so the
            NetVizController instances always own a live canvas. */}
        <div className={props.collapsed ? "hidden" : "flex gap-3"}>
          <div className="flex min-w-0 flex-1 flex-col gap-1">
            <div className="text-label font-medium">
              Actor <span className="text-muted-foreground">→ action</span>
            </div>
            <div className="relative h-40">
              <canvas ref={props.actorRef} className="block size-full" />
            </div>
          </div>
          <div className="flex min-w-0 flex-1 flex-col gap-1">
            <div className="flex items-baseline gap-1.5 text-label font-medium">
              Critic <span className="text-muted-foreground">→ value</span>
              <span className="ml-auto tabular-nums text-primary">
                {props.value === null ? "—" : props.value.toFixed(2)}
              </span>
            </div>
            <div className="relative h-40">
              <canvas ref={props.criticRef} className="block size-full" />
            </div>
          </div>
        </div>
      </div>
    </Panel>
  );
}

export const NetPanel = memo(NetPanelImpl);
