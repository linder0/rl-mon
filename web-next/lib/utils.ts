import { clsx, type ClassValue } from "clsx"
import { extendTailwindMerge } from "tailwind-merge"

// Register the custom font-size utilities (text-label / text-micro / text-nano
// from globals.css) so tailwind-merge treats them as font sizes. Otherwise it
// misclassifies them as text-color and drops real color classes like
// text-primary-foreground when both are present on one element.
const twMerge = extendTailwindMerge({
  extend: {
    classGroups: {
      "font-size": [{ text: ["label", "micro", "nano"] }],
    },
  },
})

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}
