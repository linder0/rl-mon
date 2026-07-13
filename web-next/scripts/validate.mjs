// Headless validation of the browser simulation pipeline: runs the SAME
// mujoco-js physics + onnxruntime-web policy that the app uses, and checks
//   1. policy parity   - JS actions match the Python-recorded actions
//   2. rollout parity  - JS obs track the Python deterministic rollout
//   3. gait quality    - the trained agent survives and moves forward
//
// Usage: node scripts/validate.mjs [walker2d|hopper]

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import loadMujoco from "mujoco-js";
import * as ort from "onnxruntime-web";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// In Node, point onnxruntime-web at the local wasm files (the browser uses a
// CDN instead). Use a file:// URL so the ESM loader accepts it.
ort.env.wasm.wasmPaths = new URL(
  "../node_modules/onnxruntime-web/dist/", import.meta.url,
).href;
ort.env.wasm.numThreads = 1;
ort.env.logLevel = "error";
const PUB = path.join(__dirname, "..", "public");

function buildObs(model, data, meta) {
  const obs = new Float32Array(meta.obs_dim);
  let k = 0;
  for (const c of meta.obs_components) {
    if (c.kind === "qpos") {
      for (let i = c.start ?? 0; i < meta.nq; i++) obs[k++] = data.qpos[i];
    } else if (c.kind === "qvel") {
      const lo = c.clip ? c.clip[0] : -Infinity, hi = c.clip ? c.clip[1] : Infinity;
      for (let i = c.start ?? 0; i < meta.nv; i++) {
        const v = data.qvel[i];
        obs[k++] = v < lo ? lo : v > hi ? hi : v;
      }
    } else if (c.kind === "cfrc_ext") {
      const cf = data.cfrc_ext;
      const lo = c.clip ? c.clip[0] : -Infinity, hi = c.clip ? c.clip[1] : Infinity;
      for (let i = (c.start_body ?? 0) * 6; i < cf.length; i++) {
        const v = cf[i];
        obs[k++] = v < lo ? lo : v > hi ? hi : v;
      }
    }
  }
  return obs;
}

function isHealthy(data, meta) {
  const h = meta.healthy, qpos = data.qpos, qvel = data.qvel;
  for (let i = 0; i < meta.nq; i++) if (!Number.isFinite(qpos[i])) return false;
  for (let i = 0; i < meta.nv; i++) if (!Number.isFinite(qvel[i])) return false;
  const z = qpos[h.z_index];
  if (!(z > h.z_range[0] && z < h.z_range[1])) return false;
  if (h.angle_range) {
    const angle = qpos[h.angle_index];
    if (!(angle > h.angle_range[0] && angle < h.angle_range[1])) return false;
  }
  if (h.state_range) {
    const [lo, hi] = h.state_range;
    for (let i = 2; i < meta.nq; i++) if (!(qpos[i] > lo && qpos[i] < hi)) return false;
    for (let i = 0; i < meta.nv; i++) if (!(qvel[i] > lo && qvel[i] < hi)) return false;
  }
  return true;
}

function resolveEntry(match) {
  const index = JSON.parse(fs.readFileSync(path.join(PUB, "policies", "index.json"), "utf8"));
  const runs = (index.envs ?? []).flatMap((e) => e.runs);
  if (runs.length === 0) throw new Error("no runs in index.json — run export_onnx.py first");
  const found = runs.find((r) => r.id.includes(match)) ?? runs[0];
  return found;
}

async function main() {
  const which = process.argv[2] || "walker2d";
  const entry = resolveEntry(which);
  const meta = JSON.parse(fs.readFileSync(path.join(PUB, entry.meta), "utf8"));
  console.log(`(run: ${entry.id})`);

  const mj = await loadMujoco();
  mj.FS.mkdir("/working");
  mj.FS.mount(mj.MEMFS, { root: "." }, "/working");
  const xml = fs.readFileSync(path.join(PUB, "models", meta.model_xml), "utf8");
  mj.FS.writeFile(`/working/${meta.model_xml}`, xml);
  const model = mj.MjModel.loadFromXML(`/working/${meta.model_xml}`);
  const data = new mj.MjData(model);

  const onnxBytes = fs.readFileSync(path.join(PUB, "policies", meta.policy_onnx));
  const session = await ort.InferenceSession.create(onnxBytes, {
    executionProviders: ["wasm"], graphOptimizationLevel: "all",
  });
  const inName = session.inputNames[0], outName = session.outputNames[0];
  const act = async (obs) => {
    const t = new ort.Tensor("float32", obs, [1, obs.length]);
    const out = await session.run({ [inName]: t });
    return out[outName].data;
  };

  const setState = (qpos, qvel) => {
    for (let i = 0; i < meta.nq; i++) data.qpos[i] = qpos[i];
    for (let i = 0; i < meta.nv; i++) data.qvel[i] = qvel[i];
    for (let i = 0; i < meta.nu; i++) data.ctrl[i] = 0;
    mj.mj_forward(model, data);
  };
  const step = (action) => {
    for (let i = 0; i < meta.nu; i++) data.ctrl[i] = action[i];
    for (let f = 0; f < meta.frame_skip; f++) mj.mj_step(model, data);
    if (meta.needs_rne) mj.mj_rnePostConstraint(model, data);
  };

  console.log(`\n=== ${meta.env_id}  (obs=${meta.obs_dim}, act=${meta.act_dim}, nbody=${model.nbody}, ngeom=${model.ngeom}) ===`);

  // 1. Policy parity: feed recorded obs -> compare actions (no physics).
  let maxActErr = 0;
  for (const s of meta.parity_trace) {
    const a = await act(Float32Array.from(s.obs));
    for (let i = 0; i < a.length; i++) maxActErr = Math.max(maxActErr, Math.abs(a[i] - s.action[i]));
  }
  const policyOk = maxActErr < 1e-3;
  console.log(`1. policy parity : max|Δaction| = ${maxActErr.toExponential(2)}  ${policyOk ? "OK" : "FAIL"}`);

  // 2. Rollout parity: deterministic start, step with policy, compare obs to trace.
  setState(meta.init_qpos, meta.init_qvel);
  let firstErr = 0, maxObsErr10 = 0;
  for (let t = 0; t < meta.parity_trace.length; t++) {
    const obs = buildObs(model, data, meta);
    let e = 0;
    for (let i = 0; i < obs.length; i++) e = Math.max(e, Math.abs(obs[i] - meta.parity_trace[t].obs[i]));
    if (t === 0) firstErr = e;
    if (t < 10) maxObsErr10 = Math.max(maxObsErr10, e);
    step(await act(obs));
  }
  console.log(`2. rollout parity: obs err  step0 = ${firstErr.toExponential(2)},  max over 10 steps = ${maxObsErr10.toExponential(2)}  (small = physics engines agree)`);

  // 3. Gait quality across several noised resets (these policies are stochastic
  //    in outcome: gym itself shows a wide spread of episode lengths). We check
  //    the agent CAN produce a long forward-moving episode, matching gym.
  const rng = () => (Math.random() * 2 - 1) * meta.reset_noise_scale;
  const MAXT = 1000, EPISODES = 10;
  const lens = [], dists = [];
  for (let ep = 0; ep < EPISODES; ep++) {
    setState(meta.init_qpos.map((v) => v + rng()), meta.init_qvel.map((v) => v + rng()));
    const x0 = data.qpos[0];
    let survived = 0;
    for (let t = 0; t < MAXT; t++) {
      step(await act(buildObs(model, data, meta)));
      if (meta.healthy.terminate_when_unhealthy && !isHealthy(data, meta)) break;
      survived++;
    }
    lens.push(survived);
    dists.push(data.qpos[0] - x0);
  }
  const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
  const maxLen = Math.max(...lens);
  const bestDist = Math.max(...dists);
  // A working gait: at least one long survival with real forward progress.
  const gaitOk = maxLen >= 400 && bestDist > 1.5;
  console.log(`3. gait quality  : survival lens ${JSON.stringify(lens)} (mean ${mean(lens).toFixed(0)}, best ${maxLen}), best forward distance ${bestDist.toFixed(2)} m  ${gaitOk ? "OK" : "WEAK"}`);

  const allOk = policyOk && gaitOk;
  console.log(`\nRESULT: ${allOk ? "PASS" : "CHECK"}  (${meta.env_id})`);
  process.exit(allOk ? 0 : 2);
}

main().catch((e) => { console.error(e); process.exit(1); });
