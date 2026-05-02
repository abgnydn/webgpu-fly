// lif.wgsl — fused per-timestep LIF kernel for the FlyWire connectome.
//
// One thread per neuron. Each thread:
//   1. Reads its row of incoming edges from the CSR (row_ptr, col_idx, weight).
//   2. Sums presynaptic spikes from spikes_prev, weighted.
//      Weights are pre-signed at build time (see tools/build_csr.py) so no
//      branching on E/I in the inner loop.
//   3. Adds external drive (Poisson stamped by host).
//   4. Integrates Vm with a leaky update: v ← v_rest + alpha*(v - v_rest) + i_in
//   5. Threshold + reset; sets spike bit in spikes_curr (atomic OR per word).
//
// Host ping-pongs spikes_prev / spikes_curr each timestep so the gather always
// reads stable last-step state.

struct Params {
  num_neurons      : u32,
  alpha            : f32,   // exp(-dt / tau_m)
  v_thresh         : f32,
  v_reset          : f32,
  v_rest           : f32,
  refractory_steps : u32,
  ext_gain         : f32,
  step             : u32,
  w_syn            : f32,   // Shiu 2024: 0.275 mV per synapse count
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
  // synapse count from build_csr.py; multiplied here by w_syn (mV per
  // synapse count, Shiu et al. 2024) so summed input is in mV.
  let row_start = row_ptr[i];
  let row_end   = row_ptr[i + 1u];
  var i_syn : f32 = 0.0;
  for (var k = row_start; k < row_end; k = k + 1u) {
    let pre = col_idx[k];
    i_syn = i_syn + weight[k] * spike_bit(pre);
  }
  i_syn = i_syn * params.w_syn;

  // 2. External drive
  let i_in = i_syn + ext_input[i] * params.ext_gain;

  // 3. Refractory
  let r = refrac[i];
  if (r > 0u) {
    refrac[i] = r - 1u;
    vm[i]     = params.v_reset;
    return;
  }

  // 4. Leaky integrate
  let v_old = vm[i];
  let v_new = params.v_rest + params.alpha * (v_old - params.v_rest) + i_in;
  vm[i] = v_new;

  // 5. Threshold + reset
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
