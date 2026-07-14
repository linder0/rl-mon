import { IterationClient } from "@/components/iteration/IterationClient";

/** Deep link to a specific iteration; the viewer opens on it directly. */
export default async function IterationPage({
  params,
}: {
  params: Promise<{ env: string; run: string }>;
}) {
  const { env, run } = await params;
  return (
    <IterationClient
      label={decodeURIComponent(env)}
      initialRun={decodeURIComponent(run)}
    />
  );
}
