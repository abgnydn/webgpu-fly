// evolutionParams.ts — pure host-side gait-policy parameters, no GPU/DOM deps.
// Kept separate from evolution.ts so unit tests can import it without pulling in
// shader ?raw imports.
//
// POLICY_DIM must stay in sync with src/shaders/evolve.wgsl:13.

export const POLICY_DIM = 8;

export interface EvolvedGait {
  freq: number;        // Hz (decoded from log_freq)
  coxaAmp: number;
  femurAmp: number;
  tibiaAmp: number;
  swingRatio: number;
  liftOffset: number;
  strideGain: number;
  dragCoeff: number;
}

export function decode(p: Float32Array): EvolvedGait {
  const clamp = (x: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, x));
  return {
    freq: 5.0 * Math.exp(clamp(p[0], -1.5, 1.5)),
    coxaAmp: clamp(p[1], 0, 1.5),
    femurAmp: clamp(p[2], 0, 1.5),
    tibiaAmp: clamp(p[3], 0, 1.5),
    swingRatio: clamp(p[4], 0.1, 0.9),
    liftOffset: clamp(p[5], 0, 1),
    strideGain: clamp(p[6], 0, 2),
    dragCoeff: clamp(p[7], 0, 1),
  };
}
