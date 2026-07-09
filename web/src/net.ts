import type { EnvMeta, PolicyNetLayer, PolicyNetSpec } from "./types";

/** Applies a scalar activation function by name. */
function activate(x: number, act: PolicyNetLayer["act"]): number {
  switch (act) {
    case "tanh":
      return Math.tanh(x);
    case "relu":
      return x > 0 ? x : 0;
    case "leaky_relu":
      return x > 0 ? x : 0.01 * x;
    case "elu":
      return x >= 0 ? x : Math.exp(x) - 1;
    case "silu":
      return x / (1 + Math.exp(-x));
    case "gelu":
      return 0.5 * x * (1 + Math.tanh(0.7978845608 * (x + 0.044715 * x * x * x)));
    default:
      return x;
  }
}

/** Reproduces an MLP forward pass (actor or critic) in the browser so we can
 * read every neuron's activation for the network visualization. Mirrors the
 * exported graph: normalize the raw observation (VecNormalize stats), then run
 * the dense layers in order. Kept independent of ONNX Runtime so it's cheap to
 * call every control step. */
export class PolicyNet {
  readonly layers: PolicyNetLayer[];
  /** Neuron counts per column: input obs, then each layer's output. */
  readonly sizes: number[];
  private readonly mean: Float32Array;
  private readonly invStd: Float32Array;
  private readonly clip: number;

  constructor(spec: PolicyNetSpec, meta: EnvMeta) {
    this.layers = spec.layers;
    this.sizes = [spec.input_dim, ...spec.layers.map((l) => l.out)];

    const norm = meta.normalization;
    const n = norm.mean.length;
    this.mean = Float32Array.from(norm.mean);
    this.invStd = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      this.invStd[i] = 1 / Math.sqrt(norm.var[i] + norm.epsilon);
    }
    this.clip = norm.clip_obs;
  }

  private static build(spec: PolicyNetSpec | undefined, meta: EnvMeta): PolicyNet | null {
    return spec && spec.layers.length ? new PolicyNet(spec, meta) : null;
  }

  /** The actor network (obs -> action mean). */
  static actor(meta: EnvMeta): PolicyNet | null {
    return PolicyNet.build(meta.policy_net, meta);
  }

  /** The critic network (obs -> scalar state value). */
  static critic(meta: EnvMeta): PolicyNet | null {
    return PolicyNet.build(meta.value_net, meta);
  }

  /** Returns activations for every column: index 0 is the normalized obs fed to
   * the network, then one array per layer output (post-activation). */
  forward(obs: Float32Array): Float32Array[] {
    const c = this.clip;
    let x = new Float32Array(obs.length);
    for (let i = 0; i < obs.length; i++) {
      const v = (obs[i] - this.mean[i]) * this.invStd[i];
      x[i] = v < -c ? -c : v > c ? c : v;
    }

    const acts: Float32Array[] = [x];
    for (const layer of this.layers) {
      const { w, b, act, out } = layer;
      const y = new Float32Array(out);
      for (let j = 0; j < out; j++) {
        const row = w[j];
        let sum = b[j];
        for (let k = 0; k < row.length; k++) sum += row[k] * x[k];
        y[j] = activate(sum, act);
      }
      acts.push(y);
      x = y;
    }
    return acts;
  }
}
