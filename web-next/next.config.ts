import type { NextConfig } from "next";
import path from "path";

const nextConfig: NextConfig = {
  // The viewer owns a single WebGPU context + WASM sim tied to a canvas; Strict
  // Mode's double mount/unmount in dev would init it twice on the same canvas.
  reactStrictMode: false,

  // This app has its own lockfile; pin tracing here so Next doesn't infer the
  // parent directory as the workspace root.
  outputFileTracingRoot: path.join(__dirname),

  // mujoco-js ships a single-file wasm glue and onnxruntime-web loads its own
  // wasm at runtime (from a CDN, single-threaded). Both bundles reference Node
  // builtins for their non-browser code paths; stub them out so the browser
  // build resolves cleanly.
  webpack: (config) => {
    config.resolve = config.resolve ?? {};
    config.resolve.fallback = {
      ...(config.resolve.fallback ?? {}),
      fs: false,
      path: false,
      crypto: false,
      module: false,
      worker_threads: false,
      perf_hooks: false,
      os: false,
      url: false,
      vm: false,
    };
    return config;
  },

  // The viewer runs single-threaded ONNX (no SharedArrayBuffer), so no
  // cross-origin isolation is required today. If we later switch to
  // multi-threaded or the WebGPU execution provider, uncomment these headers.
  //
  // async headers() {
  //   return [
  //     {
  //       source: "/(.*)",
  //       headers: [
  //         { key: "Cross-Origin-Opener-Policy", value: "same-origin" },
  //         { key: "Cross-Origin-Embedder-Policy", value: "require-corp" },
  //       ],
  //     },
  //   ];
  // },
};

export default nextConfig;
