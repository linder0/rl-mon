import * as React from "react"

import { cn } from "@/lib/utils"

/** Small uppercase, letter-spaced section heading used inside overlay panels
 * (Runs, Outcome, Config, …). Muted by default so it recedes behind content. */
function SectionLabel({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="section-label"
      className={cn(
        "text-micro font-semibold tracking-wider text-muted-foreground uppercase",
        className
      )}
      {...props}
    />
  )
}

export { SectionLabel }
