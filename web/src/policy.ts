import * as ort from "onnxruntime-web";

// Load the ONNX Runtime wasm from a CDN matching the installed version, and run
// single-threaded so no cross-origin isolation (COOP/COEP) headers are needed.
ort.env.wasm.wasmPaths =
  "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/";
ort.env.wasm.numThreads = 1;

// The wasm runtime is single-threaded; serialize ALL inference across every
// policy so overlapping calls (e.g. the live loop + a parity check during a
// model switch) can never run re-entrantly and corrupt the session.
let inferenceLock: Promise<unknown> = Promise.resolve();
function runExclusive<T>(fn: () => Promise<T>): Promise<T> {
  const result = inferenceLock.then(fn);
  inferenceLock = result.catch(() => {});
  return result;
}

/** Runs a trained policy exported to ONNX. Input is the RAW gym observation;
 * VecNormalize normalization is baked into the graph, so it returns the
 * deterministic action directly. */
export class Policy {
  private readonly session: ort.InferenceSession;
  private readonly inputName: string;
  private readonly outputName: string;
  readonly actDim: number;

  private constructor(
    session: ort.InferenceSession,
    inputName: string,
    outputName: string,
    actDim: number,
  ) {
    this.session = session;
    this.inputName = inputName;
    this.outputName = outputName;
    this.actDim = actDim;
  }

  static async create(onnxUrl: string, actDim: number): Promise<Policy> {
    const session = await ort.InferenceSession.create(onnxUrl, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    return new Policy(
      session,
      session.inputNames[0],
      session.outputNames[0],
      actDim,
    );
  }

  async act(obs: Float32Array): Promise<Float32Array> {
    // Copy the input: onnxruntime keeps a reference to the tensor buffer, and
    // the caller reuses its obs buffer across steps.
    const input = new ort.Tensor("float32", obs.slice(), [1, obs.length]);
    return runExclusive(async () => {
      const output = await this.session.run({ [this.inputName]: input });
      const data = output[this.outputName].data as Float32Array;
      return data.slice(0, this.actDim);
    });
  }
}
