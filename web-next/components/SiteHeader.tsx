"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { useTheme } from "next-themes";
import { Moon, Sun } from "lucide-react";
import { Button } from "@/components/ui/button";

/** Sticky header for the projects index: logotype and the theme toggle.
 * (The viewer pages have their own breadcrumb inside the control panel.) */
export function SiteHeader() {
  const { resolvedTheme, setTheme } = useTheme();
  // Avoid a hydration mismatch: the theme is unknown until mounted.
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);

  return (
    <header className="sticky top-0 z-10 border-b border-border/60 bg-background/80 backdrop-blur-xl">
      <div className="mx-auto flex h-12 max-w-6xl items-center justify-between gap-3 px-4">
        <Link href="/" className="flex items-baseline gap-1.5 text-sm hover:text-primary">
          <span className="font-semibold tracking-tight">RL MuJoCo</span>
          <span className="text-micro font-medium uppercase tracking-widest text-muted-foreground">
            projects
          </span>
        </Link>
        <Button
          variant="ghost"
          size="icon"
          className="size-8 shrink-0"
          aria-label="Toggle color theme"
          onClick={() => setTheme(resolvedTheme === "light" ? "dark" : "light")}
        >
          {mounted && resolvedTheme === "light" ? (
            <Moon className="size-4" />
          ) : (
            <Sun className="size-4" />
          )}
        </Button>
      </div>
    </header>
  );
}
