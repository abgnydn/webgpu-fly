// simParams.ts — pure host-side LIF simulation parameters, no GPU/DOM deps.
// Kept separate from sim.ts so unit tests can import it without pulling in
// shader ?raw imports.

export interface SimParams {
  dtMs: number;          // timestep in milliseconds, e.g. 1.0 for 1 kHz sim
  tauMs: number;         // membrane time constant, e.g. 20 ms
  vThresh: number;       // mV
  vReset: number;        // mV
  vRest: number;         // mV
  refractoryMs: number;  // ms (rounded to integer steps)
  extGain: number;       // multiplier on ext_input buffer
  wSyn: number;          // mV per synapse — Shiu et al 2024 free parameter (0.275 mV)
}

// LIF state constants are from Shiu et al. 2024 (Nature 632:210–217;
// github.com/philshiu/Drosophila_brain_model/model.py):
//   v_0 / v_rst = -52 mV   v_th = -45 mV   t_mbr = 20 ms   t_rfc = 2.2 ms
//   tau_syn = 5 ms (alpha synapse)
//
// The kernel runs a real two-state alpha synapse, A_SYN = exp(-1/5).
// w_syn is tuned EMPIRICALLY (not from peak-matching) to land KC at the
// canonical 5-15% on Mixed sensory. Alpha synapse integrates each spike
// over ~5ms so the cascade amplifies non-linearly vs old single-step
// direct injection — peak-matching gives way too hot a brain (73% KC).
// 0.005 keeps the dataset's natural cascade strength visible without
// runaway.
export const DEFAULT_PARAMS: SimParams = {
  dtMs: 1.0,
  tauMs: 20.0,
  vThresh: -45.0,
  vReset: -52.0,
  vRest: -52.0,
  refractoryMs: 2.2,
  extGain: 2.0,
  wSyn: 0.005,
};

/** Fails fast if the host timestep does not match the compiled shader.
 *
 * lif.wgsl hard-codes A_SYN = exp(-dt/tau_syn) for dt = 1 ms. writeParams()
 * rescales alpha and refractory steps for any dt, but A_SYN stays stale, so
 * a different dt silently corrupts synaptic decay. Guard until the shader
 * makes A_SYN a dynamic uniform field.
 */
export function assertValidDt(params: SimParams): void {
  if (params.dtMs !== 1) {
    throw new Error(
      `lif.wgsl A_SYN is compiled for dt=1ms (tau_syn=5ms); ` +
      `params.dtMs must be 1.0, got ${params.dtMs}`
    );
  }
}
