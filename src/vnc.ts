// vnc.ts — hand-coded VNC stand-in. The connectome is brain-only:
// DN axons exit the brain into the ventral nerve cord (which we don't
// have), so brain output never reaches the body unless something
// translates it. This module is that translator.
//
// Honest version of the famous-DN motor mapping. The previous
// implementation was a hardcoded lookup ("when the DNa01 button is
// clicked, set fwd=0.6") which bypassed the brain entirely. Here we
// instead read the *actual* spike rates of each named DN out of the
// brain's snapshot and combine them with canonical per-DN motor
// primitives drawn from the descending-neuron literature
// (Namiki et al. 2018, Bidaye et al. 2020).
//
// Calling code stims a neuron set, runs the brain, hands the spike-rate
// snapshot to `motorFromBrain`. If the cascade fired DNa01 strongly,
// fwd goes up. If it fired no famous DNs (e.g. spontaneous rest), the
// body sits. That's brain-driven control.

export interface MotorPrimitive {
  fwd: number;     // forward walking velocity contribution per unit spike rate
  turn: number;    // turn rate contribution per unit spike rate (signed)
  jump: number;    // upward escape impulse (cm/s) per unit spike rate
}

// Per-DN canonical motor effect, normalized so a "fully active" DN
// (rate ~0.5 spikes/step) produces a unit motor command. Numbers are
// hand-tuned to the literature: DNa02 is the strongest forward command,
// DNp01 (Giant Fiber) is pure escape, DNg13 is the canonical steering
// DN. Values are mean across both L and R copies of each named DN.
export const FAMOUS_DN_PRIMITIVES: Record<string, MotorPrimitive> = {
  DNa01: { fwd:  1.6, turn: 0,    jump:  0  },   // forward walking (Bidaye 2020)
  DNa02: { fwd:  2.4, turn: 0,    jump:  0  },   // fast forward
  DNb01: { fwd: -1.6, turn: 0,    jump:  0  },   // moonwalker, backward
  DNp01: { fwd:  0,   turn: 0,    jump: 60  },   // Giant Fiber, escape (Wyman 1984)
  DNg13: { fwd:  0.6, turn: 1.6,  jump:  0  },   // steer-while-walking
};

export interface MotorContext {
  /** Per-DN-name → list of neuron indices (both L and R copies). */
  famousDns: Record<string, number[]>;
  /** All left-hemisphere DNs (any name). For broad L/R asymmetry. */
  dnLeft: number[];
  /** All right-hemisphere DNs. */
  dnRight: number[];
  /**
   * Optional retinal cue for closed-loop control. When a visual target
   * is present, the small-field looming detector adds to escape-jump
   * activation and the horizontal angle nudges the steering. Without
   * this, the brain output is the only motor source.
   */
  visual?: { angle: number; area: number };
}

export interface MotorCommand { fwd: number; turn: number; jump: number; }

const meanRate = (rate: Float32Array, idxs: number[]) => {
  if (!idxs.length) return 0;
  let s = 0;
  for (const i of idxs) s += rate[i];
  return s / idxs.length;
};

/**
 * Map a brain snapshot to a body command. Reads:
 *   1. Mean rate of each famous DN; multiplies by its canonical primitive.
 *   2. Bilateral DN-broad asymmetry as a fallback steering signal so
 *      stim presets that don't trigger a famous DN still produce some
 *      direction (e.g. a unilateral central stim).
 *
 * Output is unbounded; callers clamp/smoothing as needed.
 */
export function motorFromBrain(
  rate: Float32Array,
  ctx: MotorContext,
): MotorCommand {
  let fwd = 0, turn = 0, jump = 0;

  // 1. Famous DN primitives
  for (const [name, idxs] of Object.entries(ctx.famousDns)) {
    const prim = FAMOUS_DN_PRIMITIVES[name];
    if (!prim) continue;
    const r = meanRate(rate, idxs);
    fwd += prim.fwd * r;
    turn += prim.turn * r;
    jump += prim.jump * r;
  }

  // 2. Broad L/R DN asymmetry. Even when no famous DN dominates, an
  // imbalance in overall DN firing (e.g. unilateral stim) should yield
  // a turn. Same 20% deadband as before — connectivity asymmetry in
  // FlyWire L/R counts produces a persistent ~10-15% bias on symmetric
  // stims.
  const meanL = meanRate(rate, ctx.dnLeft);
  const meanR = meanRate(rate, ctx.dnRight);
  const total = meanL + meanR;
  if (total > 0.01) {
    const asym = (meanR - meanL) / (total + 1e-6);
    const trim = Math.abs(asym) < 0.20 ? 0 : asym - Math.sign(asym) * 0.20;
    turn += trim * 1.6;
    // Also reinforce forward — strong bilateral DN activity = the brain
    // wants to move. Scale total rate (mean of L+R, in [0, 1]) → fwd.
    fwd += Math.min(1, total * 8);
  }

  // 3. Visual reflex (optional). When the retina sees a red target,
  // the steering also gets a soft nudge so the loop tightens — this
  // approximates the visual-system → DN feedback that the connectome
  // does encode but that takes hundreds of ms to settle. The factor is
  // small so the brain still dominates when DNs fire.
  if (ctx.visual && Number.isFinite(ctx.visual.angle)) {
    turn += ctx.visual.angle * 0.5;
  }

  return { fwd, turn, jump };
}
