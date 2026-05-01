// accumulate.wgsl — adds one timestep's spike bits into a per-neuron counter.
// Used by FlySim.captureRollingRate to keep all per-step work on the GPU.

struct Params {
  num_neurons      : u32,
  alpha            : f32,
  v_thresh         : f32,
  v_reset          : f32,
  v_rest           : f32,
  refractory_steps : u32,
  ext_gain         : f32,
  step             : u32,
};

@group(0) @binding(0) var<uniform>             params : Params;
@group(0) @binding(1) var<storage, read>       spikes : array<u32>;       // packed bitset
@group(0) @binding(2) var<storage, read_write> accum  : array<u32>;       // per-neuron spike count

const WG_SIZE : u32 = 64u;

@compute @workgroup_size(WG_SIZE)
fn accumulate_spikes(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= params.num_neurons) { return; }
  let word = spikes[i >> 5u];
  let bit  = (word >> (i & 31u)) & 1u;
  accum[i] = accum[i] + bit;
}

@compute @workgroup_size(WG_SIZE)
fn clear_accum(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= params.num_neurons) { return; }
  accum[i] = 0u;
}
