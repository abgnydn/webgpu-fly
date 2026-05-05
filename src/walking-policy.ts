// walking-policy.ts — load + run TuragaLab/flybody's trained walking
// policy in the browser. The policy is a 5-layer MLP trained with
// distributed DMPO on imitation data from real fly tracking
// (Vaxenburg et al. 2025, Nature, "Whole-body physics simulation of
// fruit fly locomotion"). 4.8 MB of float32 weights, ~3 MB of which
// is the input layer (741 obs × 512 hidden).
//
// Architecture (Acme LayerNormMLP + MultivariateNormalDiagHead, mean
// only at inference):
//   obs[741] → Linear(512) → LayerNorm → tanh
//           → Linear(512) → ELU
//           → Linear(512) → ELU
//           → Linear(512) → ELU      (activate_final=True)
//           → Linear(59)             ← action mean (deterministic)
//
// LayerNorm only on the first hidden layer; subsequent layers use
// ELU (Acme default for the trailing snt.MLP block).
//
// What this module DOES:
//   - Load the binary, parse header, slice weight buffers.
//   - Run forward pass on a 741-vec observation, return 59-vec action.
//
// What this module DOES NOT yet do:
//   - Build the observation vector at runtime from mujoco_wasm state.
//     The schema below is decoded from the SavedModel's FunctionDef
//     (the 12 flatten/concat constants in the inference graph give
//     each observable's flat dim, and dm_control's flatten_observation
//     concatenates Dict by sorted key, so the order is alphabetical).
//   - Map 59 action outputs to flybody MJCF actuator indices. The
//     subset is the same 59 actuators that show up in
//     actuator_activation — they're all listed in the MJCF in the
//     <actuator/> block, in declaration order.

/** Order matters — observations are concatenated alphabetically, which
 * is dm_control's `flatten_observation_specs` convention applied to
 * the dict the walker exposes. Total: 741 floats. */
export const WALKING_OBS_LAYOUT: Array<{ name: string; dim: number; shape: readonly number[] }> = [
  { name: "accelerometer",       dim:   3, shape: [3] },
  { name: "actuator_activation", dim:  59, shape: [59] },
  { name: "appendages_pos",      dim:  21, shape: [7, 3] },     // 7 appendages × xyz, egocentric frame
  { name: "force",               dim:  18, shape: [6, 3] },     // 6 contact force sensors × xyz
  { name: "gyro",                dim:   3, shape: [3] },
  { name: "joints_pos",          dim:  85, shape: [85] },        // 85 actuated joints' qpos
  { name: "joints_vel",          dim:  85, shape: [85] },        // matching qvel
  { name: "ref_displacement",    dim: 195, shape: [65, 3] },     // 65-frame future reference ΔR
  { name: "ref_root_quat",       dim: 260, shape: [65, 4] },     // 65-frame future reference quat
  { name: "touch",               dim:   6, shape: [6] },         // 6 touch sensors (foot contact)
  { name: "velocimeter",         dim:   3, shape: [3] },
  { name: "world_zaxis",         dim:   3, shape: [3] },
];

/** Sums to 741 — the policy's input dimension. */
export const WALKING_OBS_TOTAL = WALKING_OBS_LAYOUT.reduce((s, o) => s + o.dim, 0);

/** Helper: returns the byte offset where `name` starts in a 741-vec
 * built from WALKING_OBS_LAYOUT. Useful for writing one observable at
 * a time without recomputing offsets. */
export function obsOffset(name: string): number {
  let off = 0;
  for (const o of WALKING_OBS_LAYOUT) {
    if (o.name === name) return off;
    off += o.dim;
  }
  throw new Error(`unknown observable: ${name}`);
}

const MAGIC = "WGFLYWLK";
const HEADER_BYTES = 8 + 6 * 4;       // magic + 6 × u32

/** Per-channel running mean/std from a flybody env rollout, used to
 * approximate the trained env's ObservationActionNorm wrapper. The
 * 286 dims cover the non-synthetic streams in alphabetical order,
 * skipping ref_displacement and ref_root_quat (those we synthesize
 * with a known scale at runtime, so they don't need normalizing). */
export interface ObsNormStats {
  mean: Float32Array;     // length 286
  std: Float32Array;      // length 286
}

export async function loadWalkingObsNorm(
  url: string = (import.meta.env.VITE_WALKING_OBS_NORM_URL ?? "/walking-obs-norm.bin"),
): Promise<ObsNormStats> {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`failed to fetch ${url}: ${resp.status}`);
  const buf = await resp.arrayBuffer();
  const dv = new DataView(buf);
  const n = dv.getUint32(0, true);
  // skip 4-byte reserved
  if (8 + 8 * n !== buf.byteLength) {
    throw new Error(`obs-norm size mismatch: header n=${n}, file ${buf.byteLength} bytes`);
  }
  const mean = new Float32Array(buf, 8, n);
  const std = new Float32Array(buf, 8 + 4 * n, n);
  return { mean, std };
}

export interface ActOptions {
  /** Apply per-call LayerNorm over the full 741-vec before the first
   * dense. Crude fallback when no rollout-derived obsNorm is loaded. */
  inputLayerNorm?: boolean;
  /** Per-channel (mean, std) from a flybody env rollout. Applied to
   * the non-synthetic streams of the 741-vec. Strongly preferred over
   * inputLayerNorm — preserves relative scales between channels. */
  obsNorm?: ObsNormStats;
}

/** The 286 normalized dims map back into the 741-vec at these byte
 * offsets (alphabetical order minus ref_displacement at 274..469 and
 * ref_root_quat at 469..729). Sums to 286. */
const OBS_NORM_RANGES: ReadonlyArray<readonly [number, number]> = [
  [0, 274],     // accelerometer..joints_vel (covers norm[0..274])
  [729, 741],   // touch + velocimeter + world_zaxis (covers norm[274..286])
];

export interface WalkingPolicy {
  obsDim: number;       // 741
  hidden: number;       // 512
  actDim: number;       // 59
  nLayers: number;      // 4 (hidden layers)
  /** Run a single forward pass. obs.length must equal obsDim. */
  act(obs: Float32Array, out?: Float32Array, opts?: ActOptions): Float32Array;
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

  function elu(x: Float32Array) {
    for (let i = 0; i < x.length; i++) {
      x[i] = x[i] > 0 ? x[i] : Math.expm1(x[i]);
    }
  }

  const action = new Float32Array(actDim);

  // Per-call obs scratch so we can normalize without mutating caller's array.
  const obsNorm = new Float32Array(obsDim);

  function act(obs: Float32Array, out: Float32Array = action, opts: ActOptions = {}): Float32Array {
    if (obs.length !== obsDim) {
      throw new Error(`obs has ${obs.length} dims, expected ${obsDim}`);
    }
    let input = obs;
    if (opts.obsNorm) {
      // Per-channel (mean, std) normalization from rollout. Approximates
      // the trained env's ObservationActionNorm. Synthetic ref_*
      // channels are passed through unchanged.
      obsNorm.set(obs);
      const m = opts.obsNorm.mean;
      const sd = opts.obsNorm.std;
      let normIdx = 0;
      for (const [lo, hi] of OBS_NORM_RANGES) {
        for (let i = lo; i < hi; i++) {
          obsNorm[i] = (obs[i] - m[normIdx]) / sd[normIdx];
          normIdx++;
        }
      }
      input = obsNorm;
    } else if (opts.inputLayerNorm) {
      // Crude fallback: per-call LayerNorm over the whole vector.
      let mean = 0;
      for (let i = 0; i < obsDim; i++) mean += obs[i];
      mean /= obsDim;
      let varSum = 0;
      for (let i = 0; i < obsDim; i++) {
        const d = obs[i] - mean;
        varSum += d * d;
      }
      const inv = 1 / Math.sqrt(varSum / obsDim + 1e-5);
      for (let i = 0; i < obsDim; i++) obsNorm[i] = (obs[i] - mean) * inv;
      input = obsNorm;
    }
    // Layer 1: input → 512, LayerNorm, tanh
    dense(input, denseW[0], denseB[0], h, obsDim, hidden);
    layerNorm(h, lnScale, lnBias);
    tanh(h);
    // Layers 2..nLayers: hidden → hidden, ELU (Acme LayerNormMLP default)
    let cur = h;
    let nxt = h2;
    for (let l = 1; l < nLayers; l++) {
      dense(cur, denseW[l], denseB[l], nxt, hidden, hidden);
      elu(nxt);
      [cur, nxt] = [nxt, cur];
    }
    // Output head: hidden → actDim (no activation — action mean)
    dense(cur, headW, headB, out, hidden, actDim);
    return out;
  }

  return { obsDim, hidden, actDim, nLayers, act };
}
