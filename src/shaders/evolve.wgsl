// evolve.wgsl — parallel CPG-policy evaluation for ARS evolution.
//
// One thread per candidate policy. Each thread runs a short forward
// simulation of a stripped hexapod gait — six legs in tripod
// alternation, simplified ground contact, point-mass body — and writes
// a scalar reward into rewards[i].
//
// The host (src/evolution.ts) wraps this with: sample population →
// dispatch this kernel → readback rewards → top-k mean → resample.
// 50 generations × N=128 candidates × 200 simulation steps is well
// under 10 ms total on M2 Pro.

const POLICY_DIM : u32 = 8u;
const SIM_STEPS  : u32 = 200u;
const DT         : f32 = 0.005;     // 5 ms per step → 1 s rollout

struct EvolveParams {
  population_size : u32,
  policy_dim      : u32,
  sim_steps       : u32,
  pad             : u32,
};

@group(0) @binding(0) var<uniform> ep : EvolveParams;
@group(0) @binding(1) var<storage, read>       params  : array<f32>;
@group(0) @binding(2) var<storage, read_write> rewards : array<f32>;

const PI : f32 = 3.14159265;

// Tripod: legs T1L, T2R, T3L share phase A; T1R, T2L, T3R share phase B = A+π.
// Encoded as a sign array indexing into the 6-leg loop.
fn tripod_offset(leg : u32) -> f32 {
  // legs 0,3,4 → group A (phase 0); legs 1,2,5 → group B (phase π)
  if (leg == 0u || leg == 3u || leg == 4u) {
    return 0.0;
  }
  return PI;
}

@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) gid : vec3<u32>) {
  let cid = gid.x;
  if (cid >= ep.population_size) { return; }

  let base = cid * POLICY_DIM;
  // Decode policy. Squashes raw params to physical ranges so the
  // optimizer can explore in unbounded space; sigmoid-like mappings.
  let log_freq    = params[base + 0u];
  let coxa_amp    = clamp(params[base + 1u], 0.0, 1.5);
  let femur_amp   = clamp(params[base + 2u], 0.0, 1.5);
  let tibia_amp   = clamp(params[base + 3u], 0.0, 1.5);
  let swing_ratio = clamp(params[base + 4u], 0.1, 0.9);
  let lift_offset = clamp(params[base + 5u], 0.0, 1.0);
  let stride_gain = clamp(params[base + 6u], 0.0, 2.0);
  let drag_coeff  = clamp(params[base + 7u], 0.0, 1.0);

  let freq = 5.0 * exp(clamp(log_freq, -1.5, 1.5));   // 1 .. 22 Hz

  var body_x : f32 = 0.0;
  var body_v : f32 = 0.0;
  var energy : f32 = 0.0;
  var contact_quality : f32 = 0.0;
  var lifts_total     : f32 = 0.0;

  for (var t : u32 = 0u; t < ep.sim_steps; t = t + 1u) {
    let phase = f32(t) * DT * freq * 2.0 * PI;

    var thrust : f32 = 0.0;
    for (var leg : u32 = 0u; leg < 6u; leg = leg + 1u) {
      let p = phase + tripod_offset(leg);
      let sp = sin(p);
      // swing if sin > swing_threshold; else stance. swing_ratio sets
      // how much of the cycle is spent in swing (smaller = longer stance).
      let swing_thresh = cos(swing_ratio * PI);
      let in_swing = sp > swing_thresh;

      // Lift uses an offset-shifted phase so it leads/lags coxa.
      let lift_p = p + lift_offset * PI;
      let lift = max(0.0, sin(lift_p)) * femur_amp;

      if (in_swing) {
        // Swing: leg lifts. Track lift quality — without it, the leg
        // drags through the floor on the way back instead of swinging,
        // which biomechanically means no thrust on the next stance.
        lifts_total = lifts_total + lift;
        energy = energy + femur_amp * femur_amp * 0.05;
      } else {
        // Stance: coxa retracts, body pushes off. Thrust ∝ amplitude
        // × cos of phase × tibia push × stride. Crucially: GATED by
        // femur lift quality. A leg that didn't lift during the prior
        // swing is dragging — it still applies force, but in the wrong
        // direction. min(1, femur_amp * 3) clamps the gate so any
        // meaningful lift unlocks full thrust.
        let lift_gate = min(1.0, femur_amp * 3.0);
        let push = coxa_amp * (-sp) * tibia_amp * stride_gain * lift_gate;
        thrust = thrust + push;
        contact_quality = contact_quality + 1.0;
        energy = energy + (coxa_amp * coxa_amp + tibia_amp * tibia_amp) * 0.05;
      }
    }
    // Body update: integrate body_v under thrust − drag.
    let drag = drag_coeff * body_v * abs(body_v);
    body_v = body_v + (thrust - drag) * DT;
    body_x = body_x + body_v * DT;
  }

  // Reward shape: forward distance, scaled energy penalty, bonus for
  // good stance contact (proxy for not falling because feet were on
  // the ground enough). No upright bonus needed in this simplified
  // 1-D rollout.
  let total = f32(ep.sim_steps);
  let stance_frac = contact_quality / (total * 6.0);
  let reward = body_x
             - 0.002 * energy
             + 0.5 * stance_frac
             - 0.05 * abs(stance_frac - 0.5);
  rewards[cid] = reward;
}
