"use client";

import dynamic from "next/dynamic";
import { Loader2 } from "lucide-react";

// The viewer touches WebGPU, WASM (MuJoCo + ONNX) and window on load, so it must
// only ever run in the browser. Loading it via next/dynamic with ssr:false keeps
// it out of the server bundle entirely.
const Viewer = dynamic(() => import("@/components/Viewer"), {
  ssr: false,
  loading: () => (
    <div className="flex h-screen w-screen flex-col items-center justify-center gap-4 bg-background">
      <Loader2 className="size-8 animate-spin text-primary" />
      <div className="text-sm text-muted-foreground">Loading viewer…</div>
    </div>
  ),
});

export function ClientApp() {
  return <Viewer />;
}
