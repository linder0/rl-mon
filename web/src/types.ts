/** Metadata + stats emitted by export_onnx.py for each (run, variant). Mirrors
 * the JSON written under web/public/policies/. */

/** One piece of the observation vector, rebuilt from mjData in the browser.
 *  - qpos:     data.qpos[start:]
 *  - qvel:     data.qvel[start:], optionally clipped
 *  - cfrc_ext: data.cfrc_ext[start_body:] flattened, clipped (contact forces) */
export interface ObsComponent {
  kind: "qpos" | "qvel" | "cfrc_ext";
  start?: number;
  start_body?: number;
  clip: [number, number] | null;
}

export interface HealthySpec {
  terminate_when_unhealthy: boolean;
  z_index: number;
  z_range: [number, number];
  angle_index: number;
  angle_range: [number, number] | null;
  state_range: [number, number] | null;
}

export interface ParityStep {
  obs: number[];
  action: number[];
}

export interface RunSummary {
  timesteps: number | null;
  train_ep_rew_final: number | null;
  final_eval_mean: number | null;
  best_eval_mean: number | null;
  eval_ep_len_final: number | null;
  seed: number | null;
}

/** One dense layer of the actor MLP (weights are [out][in]). Exported by
 * export_onnx.py so the browser can reproduce per-neuron activations. */
export interface PolicyNetLayer {
  w: number[][];
  b: number[];
  act: "tanh" | "relu" | "elu" | "leaky_relu" | "silu" | "gelu" | "linear";
  in: number;
  out: number;
}

export interface PolicyNetSpec {
  input_dim: number;
  layers: PolicyNetLayer[];
}

export interface EnvMeta {
  env_id: string;
  model_xml: string;
  obs_dim: number;
  act_dim: number;
  nq: number;
  nv: number;
  nu: number;
  frame_skip: number;
  timestep: number;
  dt: number;
  action_low: number[];
  action_high: number[];
  obs_components: ObsComponent[];
  needs_rne: boolean;
  obs_labels?: string[];
  action_labels?: string[];
  init_qpos: number[];
  init_qvel: number[];
  reset_noise_scale: number;
  healthy: HealthySpec;
  normalization: {
    mean: number[];
    var: number[];
    epsilon: number;
    clip_obs: number;
  };
  policy_onnx: string;
  policy_net?: PolicyNetSpec;
  value_net?: PolicyNetSpec;
  run: { id: string; label: string; run_name: string; variant: string };
  summary: RunSummary;
  parity_trace: ParityStep[];
}

export interface RunEntry {
  id: string;
  run_name: string;
  variant: string;
  onnx: string;
  meta: string;
  stats: string;
  model_xml: string;
  summary: RunSummary;
}

export interface EnvGroup {
  env_id: string;
  label: string;
  runs: RunEntry[];
}

export interface PolicyIndex {
  envs: EnvGroup[];
}

/** One training run within an env, aggregating its final/best variants (which
 * share the same training/eval curves and config, differing only in weights). */
export interface RunAgg {
  runName: string;
  envId: string;
  color: string;
  seed: number | null;
  summary: RunSummary;
  statsPath: string;
  variants: { final?: RunEntry; best?: RunEntry };
}

export interface StatsData {
  config: {
    env_id: string | null;
    command: string | null;
    created: string | null;
    device: string | null;
    hyperparameters: Record<string, unknown>;
    args: Record<string, unknown>;
    versions: Record<string, string>;
  };
  curves: {
    train: { t: number[]; reward: number[]; ep_len: number[] };
    eval: { t: number[]; mean: number[]; std: number[]; ep_len: number[] };
    diag: Record<string, { t: number[]; v: number[] }>;
  };
  summary: RunSummary;
}
