import { IterationClient } from "@/components/iteration/IterationClient";

/** A project opens straight into the viewer, on its best iteration. */
export default async function ProjectPage({
  params,
}: {
  params: Promise<{ env: string }>;
}) {
  const { env } = await params;
  return <IterationClient label={decodeURIComponent(env)} />;
}
