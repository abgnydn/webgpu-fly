// lif.wgsl — fused per-timestep LIF kernel for the FlyWire connectome.
//
// One thread per neuron. Each thread:
//   1. Sums presynaptic spikes from spikes_prev, weighted, into delta_in.
//   2. Advances two-state alpha synapse:
//        g_y[n+1] = g_y[n]*a_syn + g_x[n]
//        g_x[n+1] = g_x[n]*a_syn + delta_in
//      i_syn = g_y * w_syn. Spreads each spike's effect over ~tau_syn (5 ms).
//      This matches Shiu et al. 2024's alpha synapse; w_syn = 0.275 mV is
//      the literal paper value, no calibration fudge.
//   3. Adds external drive (Poisson stamped by host), then leaky integrate,
//      threshold, reset, atomic-OR spike bit into spikes_curr.
//
// Host ping-pongs spikes_prev / spikes_curr each timestep so the gather always
// reads stable last-step state.

struct Params {
  num_neurons      : u32,   // offset 0
  alpha            : f32,   // 4   exp(-dt / tau_m)
  v_thresh         : f32,   // 8
  v_reset          : f32,   // 12
  v_rest           : f32,   // 16
  refractory_steps : u32,   // 20
  ext_gain         : f32,   // 24
  step             : u32,   // 28
  w_syn            : f32,   // 32  Shiu 2024 free param
  a_syn            : f32,   // 36  exp(-dt / tau_syn), tau_syn ≈ 5 ms
  pad0             : f32,   // 40  pads struct to 48 B for uniform 16-byte stride
  pad1             : f32,   // 44
};

@group(0) @binding(0) var<uniform>             params      : Params;
@group(0) @binding(1) var<storage, read>       row_ptr     : array<u32>;
@group(0) @binding(2) var<storage, read>       col_idx     : array<u32>;
@group(0) @binding(3) var<storage, read>       weight      : array<f32>;
@group(0) @binding(4) var<storage, read>       spikes_prev : array<u32>;        // packed bitset, 1 = fired last step
@group(0) @binding(5) var<storage, read_write> spikes_curr : array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> vm          : array<f32>;
@group(0) @binding(7) var<storage, read_write> refrac      : array<u32>;
@group(0) @binding(8) var<storage, read>       ext_input   : array<f32>;
@group(0) @binding(9) var<storage, read_write> g_x         : array<f32>;
@group(0) @binding(10) var<storage, read_write> g_y        : array<f32>;

const WG_SIZE : u32 = 64u;

fn spike_bit(idx : u32) -> f32 {
  let word = spikes_prev[idx >> 5u];
  return f32((word >> (idx & 31u)) & 1u);
}

@compute @workgroup_size(WG_SIZE)
fn step_lif(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= params.num_neurons) { return; }

  // 1. Synaptic gather over incoming edges. weight[k] is the pre-signed
  // synapse count from build_csr.py — already multiplied by ±1 for E/I sign.
  let row_start = row_ptr[i];
  let row_end   = row_ptr[i + 1u];
  var delta_in : f32 = 0.0;
  for (var k = row_start; k < row_end; k = k + 1u) {
    let pre = col_idx[k];
    delta_in = delta_in + weight[k] * spike_bit(pre);
  }

  // 2. Alpha synapse update. y uses old x; then x absorbs new spikes.
  let gx_old = g_x[i];
  let gy_new = g_y[i] * params.a_syn + gx_old;
  let gx_new = gx_old * params.a_syn + delta_in;
  g_y[i] = gy_new;
  g_x[i] = gx_new;
  let i_syn = gy_new * params.w_syn;

  // 3. External drive
  let i_in = i_syn + ext_input[i] * params.ext_gain;

  // 4. Refractory
  let r = refrac[i];
  if (r > 0u) {
    refrac[i] = r - 1u;
    vm[i]     = params.v_reset;
    return;
  }

  // 5. Leaky integrate
  let v_old = vm[i];
  let v_new = params.v_rest + params.alpha * (v_old - params.v_rest) + i_in;
  vm[i] = v_new;

  // 6. Threshold + reset
  if (v_new >= params.v_thresh) {
    vm[i]     = params.v_reset;
    refrac[i] = params.refractory_steps;
    let word_i = i >> 5u;
    let bit    = 1u << (i & 31u);
    atomicOr(&spikes_curr[word_i], bit);
  }
}

// Host runs this dispatch first each timestep to zero the spike bitset.
@compute @workgroup_size(WG_SIZE)
fn clear_spikes(@builtin(global_invocation_id) gid : vec3<u32>) {
  let words = (params.num_neurons + 31u) >> 5u;
  let i = gid.x;
  if (i >= words) { return; }
  atomicStore(&spikes_curr[i], 0u);
}
