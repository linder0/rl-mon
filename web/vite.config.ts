import { defineConfig } from "vite";

export default defineConfig({
  server: { port: 5173, open: false },
  // mujoco-js ships an 11 MB single-file wasm module and onnxruntime-web loads
  // its own wasm at runtime; keep esbuild's dependency pre-bundler away from both.
  optimizeDeps: {
    exclude: ["mujoco-js", "onnxruntime-web"],
  },
  build: {
    target: "es2022",
    chunkSizeWarningLimit: 12000,
  },
});
