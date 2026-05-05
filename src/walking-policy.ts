// walking-policy.ts — load + run TuragaLab/flybody's trained walking
// policy in the browser. The policy is a 5-layer MLP trained with
// distributed DMPO on imitation data from real fly tracking
// (Vaxenburg et al. 2025, Nature, "Whole-body physics simulation of
// fruit fly locomotion"). 4.8 MB of float32 weights, ~3 MB of which
// is the input layer (741 obs × 512 hidden).
//
// Architecture (after `tools/extract_walking_policy.py`):
//   obs[741] → Dense(512) → LayerNorm → tanh
//           → Dense(512) → tanh
//           → Dense(512) → tanh
//           → Dense(512) → tanh
//           → Dense(59)              ← action mean (deterministic)
//
// LayerNorm only on the first hidden layer (Acme's LayerNormMLP
// convention — keeps activations sane after the high-dim input
// projection).
//
// What this module DOES:
//   - Load the binary, parse header, slice weight buffers.
//   - Run forward pass on a 741-vec observation, return 59-vec action.
//
// What this module DOES NOT yet do:
//   - Build the observation vector. flybody's walk_imitation task
//     concatenates ~10 observable streams (gyro + accelerometer +
//     velocimeter + world_zaxis + joints_pos + joints_vel +
//     actuator_activation + appendages_pos + force + touch +
//     ref_displacement + ref_root_quat) in a specific order. Each
//     stream has a fixed length defined by the MJCF. To wire this up,
//     someone needs to either:
//       (a) Run flybody in Python, dump observation_spec ordering and
//           per-step values, port the construction to JS verbatim.
//       (b) Read dm_control source and reproduce the
//           Dict[str, ndarray] → flat-vector concatenation logic.
//   - Map 59 action outputs to specific flybody actuator indices. The
//     policy was trained on a subset of the 111 actuators (the 59
//     leg + thorax actuators relevant to walking). Mapping requires
//     reading the trained env's action_spec.

const MAGIC = "WGFLYWLK";
const HEADER_BYTES = 8 + 6 * 4;       // magic + 6 × u32

export interface WalkingPolicy {
  obsDim: number;       // 741
  hidden: number;       // 512
  actDim: number;       // 59
  nLayers: number;      // 4 (hidden layers)
  /** Run a single forward pass. obs.length must equal obsDim. */
  act(obs: Float32Array, out?: Float32Array): Float32Array;
}

export async function loadWalkingPolicy(
  url: string = (import.meta.env.VITE_WALKING_POLICY_URL ?? "/walking-policy.bin"),
): Promise<WalkingPolicy> {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`failed to fetch ${url}: ${resp.status}`);
  const buf = await resp.arrayBuffer();

  const dv = new DataView(buf);
  const magic = String.fromCharCode(...new Uint8Array(buf, 0, 8));
  if (magic !== MAGIC) throw new Error(`bad magic: got "${magic}", expected "${MAGIC}"`);
  const version = dv.getUint32(8, true);
  if (version !== 1) throw new Error(`unsupported policy version ${version}`);
  const obsDim = dv.getUint32(12, true);
  const hidden = dv.getUint32(16, true);
  const actDim = dv.getUint32(20, true);
  const nLayers = dv.getUint32(24, true);

  // Slice weights in declared order (see extract_walking_policy.py).
  let off = HEADER_BYTES;
  const slice = (n: number): Float32Array => {
    const view = new Float32Array(buf, off, n);
    off += n * 4;
    return view;
  };

  const lnScale = slice(hidden);                  // [hidden]
  const lnBias = slice(hidden);                   // [hidden]
  const denseW: Float32Array[] = [];
  const denseB: Float32Array[] = [];
  // Layer 1: obs → hidden
  denseW.push(slice(obsDim * hidden));            // [obsDim, hidden]
  denseB.push(slice(hidden));                     // [hidden]
  // Layers 2..nLayers: hidden → hidden
  for (let l = 1; l < nLayers; l++) {
    denseW.push(slice(hidden * hidden));
    denseB.push(slice(hidden));
  }
  // Output head: hidden → actDim
  const headW = slice(hidden * actDim);
  const headB = slice(actDim);

  if (off !== buf.byteLength) {
    console.warn(`walking-policy.bin: ${buf.byteLength - off} trailing bytes ignored`);
  }

  // Pre-allocate scratch buffers for the forward pass — avoids GC
  // churn at 60 fps if anyone ever runs this every frame.
  const h = new Float32Array(hidden);
  const h2 = new Float32Array(hidden);

  function dense(x: Float32Array, w: Float32Array, b: Float32Array, out: Float32Array, inDim: number, outDim: number) {
    // out[j] = b[j] + sum_i x[i] * w[i*outDim + j]
    for (let j = 0; j < outDim; j++) out[j] = b[j];
    for (let i = 0; i < inDim; i++) {
      const xi = x[i];
      if (xi === 0) continue;
      const row = i * outDim;
      for (let j = 0; j < outDim; j++) out[j] += xi * w[row + j];
    }
  }

  function layerNorm(x: Float32Array, scale: Float32Array, bias: Float32Array) {
    const n = x.length;
    let mean = 0;
    for (let i = 0; i < n; i++) mean += x[i];
    mean /= n;
    let varSum = 0;
    for (let i = 0; i < n; i++) {
      const d = x[i] - mean;
      varSum += d * d;
    }
    const inv = 1 / Math.sqrt(varSum / n + 1e-5);
    for (let i = 0; i < n; i++) x[i] = (x[i] - mean) * inv * scale[i] + bias[i];
  }

  function tanh(x: Float32Array) {
    for (let i = 0; i < x.length; i++) x[i] = Math.tanh(x[i]);
  }

  const action = new Float32Array(actDim);

  function act(obs: Float32Array, out: Float32Array = action): Float32Array {
    if (obs.length !== obsDim) {
      throw new Error(`obs has ${obs.length} dims, expected ${obsDim}`);
    }
    // Layer 1: obs → 512, LayerNorm, tanh
    dense(obs, denseW[0], denseB[0], h, obsDim, hidden);
    layerNorm(h, lnScale, lnBias);
    tanh(h);
    // Layers 2..nLayers: hidden → hidden, tanh
    let cur = h;
    let nxt = h2;
    for (let l = 1; l < nLayers; l++) {
      dense(cur, denseW[l], denseB[l], nxt, hidden, hidden);
      tanh(nxt);
      [cur, nxt] = [nxt, cur];
    }
    // Output head: hidden → actDim (no activation — action mean)
    dense(cur, headW, headB, out, hidden, actDim);
    return out;
  }

  return { obsDim, hidden, actDim, nLayers, act };
}
