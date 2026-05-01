// evolution.ts — placeholder for browser-side ARS / evolution strategy.
//
// Idea: parallel-evolve a CPG parameter vector against forward-locomotion
// reward in a simplified hexapod sim, all on WebGPU. Best parameters
// drive flybody's real leg actuators in mujoco_wasm — sim-to-real but
// for browsers.
//
// Architecture sketch (not yet implemented):
//
//   1. Policy = ~30-50 dim parameter vector (per-leg phase offsets,
//      coxa/femur/tibia amplitudes, frequency, swing-stance ratio).
//   2. WebGPU compute pipeline runs N=512+ candidate policies in parallel:
//      - Each thread group simulates one fly for K=200 timesteps using
//        a stripped-down hexapod dynamics (capsule legs, freejoint body,
//        heightfield contact).
//      - Reward = forward distance over K timesteps + upright bonus.
//   3. Reduction: sort by reward, average the top-k% (CMA-ES / OpenAI ES
//      style) to update the population mean.
//   4. Resample: next generation = mean + σ × N(0, 1) per dim.
//   5. After ~50 generations (~30 sec wall on M2 Pro), best parameters
//      are written to flybody's actual position actuators in the live
//      mujoco_wasm room. Sim-to-real transfer.
//
// Why we don't have it yet:
//   - WGSL compute pipeline for hexapod dynamics is non-trivial (~half day)
//   - Reward shaping for stable walking takes iteration
//   - The kinematic-boost workaround in physics.driveLegs already
//     produces visible walking, so this is polish, not unblock.
//
// Re-enable when you want to ship "evolved by your tab" as a demo
// feature. Until then this file exists to document the intent.

export const TODO_EVOLUTION = "see header comment for ARS plan";
