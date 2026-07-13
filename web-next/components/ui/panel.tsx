import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"

import { Card } from "@/components/ui/card"
import { cn } from "@/lib/utils"

/**
 * Floating glass panel — the shared chrome for every overlay panel in the
 * viewer (control, stats, dials, scene, net). Owns the "card over the 3D scene"
 * identity so the class string lives in one place instead of being copy-pasted.
 *
 * Presentational only: collapse/visibility/persistence stays with each panel.
 */
const panelVariants = cva(
  "pointer-events-auto max-w-[calc(100vw-1.5rem)] gap-0 border-0 bg-(--panel-bg) py-0 shadow-(--panel-shadow) backdrop-blur-xl",
  {
    variants: {
      size: {
        sm: "w-60",
        md: "w-[300px]",
        lg: "w-[340px]",
        xl: "w-[460px]",
      },
    },
    defaultVariants: {
      size: "md",
    },
  }
)

function Panel({
  className,
  size,
  ...props
}: Omit<React.ComponentProps<typeof Card>, "size"> &
  VariantProps<typeof panelVariants>) {
  return (
    <Card
      data-slot="panel"
      className={cn(panelVariants({ size }), className)}
      {...props}
    />
  )
}

/**
 * Standard panel header row: a left region (title / view toggle / logotype) and
 * a right region (actions such as Hide / theme toggle), justified apart.
 */
function PanelHeader({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="panel-header"
      className={cn("flex items-center justify-between gap-2 p-3", className)}
      {...props}
    />
  )
}

Panel.Header = PanelHeader

export { Panel, PanelHeader, panelVariants }
