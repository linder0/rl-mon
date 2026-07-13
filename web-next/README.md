# RL MuJoCo — Web Viewer (Next.js)

Run your trained MuJoCo policies **live in the browser**: MuJoCo physics
compiled to WebAssembly, the policy exported to ONNX, rendered with three.js
(WebGPU, with automatic WebGL fallback). No server logic at runtime — all
simulation and inference happen client-side.

This is the **primary viewer**. The original Vite app in [`../web`](../web) is
legacy.

```
┌──────────────── browser (per frame) ─────────────────┐
│  MuJoCo WASM ──qpos,qvel──▶ build obs ──▶ ONNX policy │
│       ▲                                        │      │
│       └──── set ctrl, step frame_skip ◀── action      │
│  data.xpos / data.xquat ──▶ three.js WebGPU ──▶ GPU   │
└───────────────────────────────────────────────────────┘
```

## 1. Export trained policies (from the repo root)

This reads runs under `runs/<env>/` and writes browser assets into
`web-next/public/` (and mirrors them to the legacy `web/public/`). Every run is
exported individually (final + best), so you can compare seeds/configs side by
side:

```bash
# from the repo root, with the project venv active
python export_onnx.py --all-envs             # every run of every env (recommended)
python export_onnx.py --env Walker2d-v5 --all  # every Walker run with a model
python export_onnx.py --env Walker2d-v5       # just the latest Walker run
python export_onnx.py --run runs/hopper/lambda_a10 --variant final
```

Each exported run+variant produces (id = `<env>__<run_name>__<final|best>`):

- `public/policies/<id>.onnx` — the policy MLP with VecNormalize normalization
  baked in (input = raw gym observation, output = deterministic action).
- `public/policies/<id>.json` — the exact observation/stepping/health spec plus
  a deterministic "parity trace" used to verify correctness in the browser.
- `public/policies/<id>.stats.json` — config/hyperparameters, training curve
  (from `progress.csv`), and eval curve (from `evaluations.npz`) for the dashboard.
- `public/models/<model>.xml` — the MuJoCo model (physics + geometry).
- `public/policies/index.json` — the grouped catalog the app's Run picker reads.

## 2. Run the viewer

```bash
cd web-next
npm install
npm run dev        # http://localhost:3000
```

Controls: play/pause (Space), reset (R), speed slider, follow-camera toggle, and
a grouped Run picker (by environment) to switch between every exported run and
its final/best variant — with run comparison overlays on the charts. The stats
panel shows summary cards (best/final eval reward, episode length, timesteps,
seed), the eval-reward curve (mean ± std) and training-reward curve (rendered
with uPlot), the run's config/hyperparameters, and a live `parity` line
confirming the in-browser policy matches the Python reference. A collapsible
"Policy network" panel visualizes live actor/critic activations every control
step.

## 3. Validate headlessly (optional)

Runs the same `mujoco-js` + `onnxruntime-web` code path as the app in Node and
checks policy parity, physics-rollout parity vs Python, and gait quality:

```bash
npm run validate -- walker2d    # or: node scripts/validate.mjs walker2d
npm run validate -- hopper
```

## How it works

- `app/` — the Next.js App Router shell. The whole viewer is loaded via
  `next/dynamic` with `ssr: false` (`components/ClientApp.tsx`) since it touches
  WebGPU, WASM, and `window` on load.
- `lib/viewerApp.ts` — the framework-agnostic core that owns the sim, policy,
  renderer, and loop, and reports state back to React through callbacks.
- `lib/mujocoSim.ts` — loads the WASM module, mounts the MJCF, and steps the
  simulation exactly like Gymnasium (same observation, `frame_skip`, and health
  checks, read from the exported metadata).
- `lib/policy.ts` — runs the ONNX policy via `onnxruntime-web`.
- `lib/loop.ts` — the real-time `obs → policy → step → render` loop with
  auto-reset on falling / time limit.
- `components/scene/webgpuRenderer.ts` — builds three.js meshes from the
  model's geoms and updates them each frame from the body world transforms
  (`data.xpos` / `data.xquat`); uses the `three/webgpu` renderer with a WebGL
  backend fallback. MuJoCo is Z-up, so everything hangs under a root rotated to
  three.js's Y-up.
- `lib/net.ts` + `lib/netvizController.ts` + `components/net/NetPanel.tsx` — a
  plain-JS MLP forward pass mirroring the exported graph, used to read every
  neuron's activation for the live network visualization.
- `components/charts/UPlotChart.tsx` — uPlot wrapper for the training/eval
  reward curves.

## Notes

- `onnxruntime-web` loads its wasm from a CDN pinned to the installed version
  and runs single-threaded, so no COOP/COEP headers are required (see
  `next.config.ts` for the headers to enable if that changes).
- `mujoco-js` ships the MuJoCo wasm inline (single file), so nothing extra needs
  to be served. Both packages reference Node builtins in their non-browser code
  paths; `next.config.ts` stubs those out for the browser build.
- React Strict Mode is disabled: the viewer owns a single WebGPU context + WASM
  sim tied to a canvas, and Strict Mode's dev double-mount would initialize it
  twice.
- The Walker2d/Hopper models use only primitive geoms (capsules, spheres, boxes)
  and MuJoCo's built-in procedural textures, so no external mesh assets are
  needed. Models that reference `.obj`/`.stl`/`.png` files would need those
  copied into `public/models/` too.
