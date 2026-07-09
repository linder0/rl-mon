# RL MuJoCo — Web Viewer

Run your trained MuJoCo policies **live in the browser**: MuJoCo physics compiled
to WebAssembly, the policy exported to ONNX, rendered with three.js/WebGL. No
server at runtime — it's a static site.

```
┌──────────────── browser (per frame) ─────────────────┐
│  MuJoCo WASM ──qpos,qvel──▶ build obs ──▶ ONNX policy │
│       ▲                                        │      │
│       └──── set ctrl, step frame_skip ◀── action      │
│  data.xpos / data.xquat ──▶ three.js renderer ──▶ GL  │
└───────────────────────────────────────────────────────┘
```

## 1. Export trained policies (from the repo root)

This reads runs under `runs/<env>/` and writes browser assets into
`web/public/`. Every run is exported individually (final + best), so you can
compare seeds/configs side by side:

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
cd web
npm install
npm run dev        # http://localhost:5173
```

Controls: play/pause (Space), reset (R), speed slider, follow-camera toggle, and
a grouped Run dropdown (by environment) to switch between every exported run and
its final/best variant. The left panel shows live rollout stats and a `parity`
line confirming the in-browser policy matches the Python reference. The right
panel is a per-run dashboard: summary cards (best/final eval reward, episode
length, timesteps, seed), the eval-reward curve (mean ± std) and training-reward
curve, and the run's config/hyperparameters.

## 3. Validate headlessly (optional)

Runs the same `mujoco-js` + `onnxruntime-web` code path as the app in Node and
checks policy parity, physics-rollout parity vs Python, and gait quality:

```bash
node scripts/validate.mjs walker2d
node scripts/validate.mjs hopper
```

## How it works

- `src/mujocoSim.ts` — loads the WASM module, mounts the MJCF, and steps the
  simulation exactly like Gymnasium (same observation, `frame_skip`, and health
  checks, read from the exported metadata).
- `src/policy.ts` — runs the ONNX policy via `onnxruntime-web`.
- `src/renderer.ts` — builds three.js meshes from the model's geoms and updates
  them each frame from the body world transforms (`data.xpos` / `data.xquat`);
  MuJoCo is Z-up so everything hangs under a root rotated to three.js's Y-up.
- `src/loop.ts` — the real-time `obs → policy → step → render` loop with
  auto-reset on falling / time limit.

## Notes

- `onnxruntime-web` loads its wasm from a CDN pinned to the installed version and
  runs single-threaded, so no COOP/COEP headers are required.
- `mujoco-js` ships the MuJoCo wasm inline (single file), so nothing extra needs
  to be served.
- The Walker2d/Hopper models use only primitive geoms (capsules, spheres, boxes)
  and MuJoCo's built-in procedural textures, so no external mesh assets are
  needed. Models that reference `.obj`/`.stl`/`.png` files would need those
  copied into `public/models/` too.
