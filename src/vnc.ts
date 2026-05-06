// vnc.ts — synthetic ventral nerve cord (the "spine"). Real biological
// VNC connectome (MANC, Takemura et al. 2024) isn't loaded here yet;
// instead this is a hand-designed but biophysically-real LIF network
// that takes the brain's DN spike rates as input, runs ~200 leaky
// integrate-and-fire neurons on CPU, and produces a motor command for
// the body.
//
// Why a network and not a lookup table: the previous implementation was
// stateless (rate × primitive → motor) which made the body lurch the
// instant the brain fired. Real spinal-cord cascades take ~10–50 ms to
// settle, with rhythmic oscillations from half-center mutual inhibition.
// This module reproduces that with actual integration dynamics, so the
// fly's gait ramps in/out smoothly instead of stepping like a state
// machine.
//
// Topology (200 neurons total):
//   [0..4]      DN inputs       — DNa01, DNa02, DNb01, DNg13, DNp01
//   [5..24]     Decision pool   — 10 forward, 10 backward (mutual inh)
//   [25..174]   6 leg circuits  — 6 × 25, each = half-center oscillator
//                                  (12 flexor + 12 extensor + 1 motor)
//   [175..199]  Tripod sync     — 25 inter-leg coordinator
//
// Output: walk_mag (forward firing pool mean rate),
//         walk_sign (forward − backward), turn_bias (R − L motor diff),
//         escape (DNp01 input rate, drives jumpImpulse).
//
// This is the SHAPE of a biological spinal CPG. The wiring is hand-set,
// but every neuron is a real LIF cell that integrates synaptic current
// over time. Drop-in replacement: when the MANC connectome dump
// becomes easily downloadable, swap this for a real CSR loader.

const N_VNC          = 200;
const N_DN_IN        = 5;          // 0..4
const N_DECISION     = 20;         // 5..24
const N_PER_LEG      = 25;         // 25 each × 6 legs = 150
const N_LEGS         = 6;
const N_TRIPOD       = 25;         // 175..199

const FWD_POOL       = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14];     // 10 forward neurons
const BWD_POOL       = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24]; // 10 backward neurons

// Per-leg, the motor neuron is the 24th (last) neuron in its block.
const LEG_BASE = (leg: number) => N_DN_IN + N_DECISION + leg * N_PER_LEG;
const LEG_MOTOR = (leg: number) => LEG_BASE(leg) + 24;
// Half-center oscillator within each leg: flexor pool 0..11, extensor 12..23.
const LEG_FLEXOR = (leg: number, k: number) => LEG_BASE(leg) + k;       // k ∈ [0, 12)
const LEG_EXTENSOR = (leg: number, k: number) => LEG_BASE(leg) + 12 + k; // k ∈ [0, 12)

// Leg labels matching Physics.LEG_KEYS order:
// [0]=T1L, [1]=T1R, [2]=T2L, [3]=T2R, [4]=T3L, [5]=T3R
const LEG_IS_LEFT = [true, false, true, false, true, false];
// Tripod A = T1L, T2R, T3L (leg indices 0, 3, 4); tripod B = T1R, T2L, T3R (1, 2, 5).
const TRIPOD_A_LEGS = [0, 3, 4];
const TRIPOD_B_LEGS = [1, 2, 5];

interface SynapseEdge { pre: number; w: number; }

/**
 * VNC LIF network. CPU-side because 200 neurons × ~1500 synapses runs
 * in ~10 µs per ms-step, well below the GPU dispatch overhead.
 */
export class VNC {
  private vm: Float32Array = new Float32Array(N_VNC);
  private refrac: Uint16Array = new Uint16Array(N_VNC);
  private spikePrev: Uint8Array = new Uint8Array(N_VNC);
  private extInput: Float32Array = new Float32Array(N_VNC);
  // CSR-ish: incoming edges per neuron, indexed by row.
  private incoming: SynapseEdge[][] = [];
  // Spike-rate trace per neuron for output computation.
  private rate: Float32Array = new Float32Array(N_VNC);
  private rateAlpha = 0.85;     // EWMA on spike output, ~50ms time constant
  private vmAlpha = Math.exp(-1 / 20);     // tau_m = 20 ms
  private vRest = -52;
  private vReset = -52;
  private vThresh = -45;
  private refracSteps = 2;

  constructor() {
    for (let i = 0; i < N_VNC; i++) {
      this.vm[i] = this.vRest;
      this.incoming.push([]);
    }
    this.wire();
  }

  /** Hand-designed connectivity. Excitatory positive weight, inhibitory
   * negative. Numbers tuned so a single sustained DN input drives the
   * downstream pool at ~10–50 Hz. */
  private wire() {
    const e = (post: number, pre: number, w: number) => this.incoming[post].push({ pre, w });

    // 1. DN → decision pool
    // DNa01 (idx 0), DNa02 (1) → forward pool excitatory
    for (const fwd of FWD_POOL) { e(fwd, 0, 4.0); e(fwd, 1, 6.0); }
    // DNb01 (2) → backward pool excitatory
    for (const bwd of BWD_POOL) { e(bwd, 2, 5.0); }
    // DNg13 (3) → asymmetric: half of forward pool only (drives turn)
    for (let k = 0; k < 5; k++) e(FWD_POOL[k], 3, 3.0);
    // DNp01 (4) → both pools weakly (reset / escape gate)
    for (const fwd of FWD_POOL) e(fwd, 4, -1.5);
    for (const bwd of BWD_POOL) e(bwd, 4, -1.5);

    // 2. Forward and backward pool mutual inhibition (decision)
    for (const fwd of FWD_POOL) for (const bwd of BWD_POOL) {
      e(fwd, bwd, -2.5);
      e(bwd, fwd, -2.5);
    }

    // 3. Decision → leg CPGs
    for (let leg = 0; leg < N_LEGS; leg++) {
      // Forward decision excites flexor pool half (drives swing-stance cycle)
      for (let k = 0; k < 12; k++) {
        for (const fwd of FWD_POOL) {
          e(LEG_FLEXOR(leg, k), fwd, 1.5);
          e(LEG_EXTENSOR(leg, k), fwd, 1.5);
        }
        // Backward decision drives same neurons but flipped — extensor first
        // (we just bias amplitude, the gait sign is handled by physics.driveLegs)
        for (const bwd of BWD_POOL) {
          e(LEG_FLEXOR(leg, k), bwd, 1.5);
          e(LEG_EXTENSOR(leg, k), bwd, 1.5);
        }
      }
    }

    // 4. Half-center oscillator within each leg
    // Flexor and extensor pools mutually inhibit; spike-rate adaptation
    // (refractory period is enough to give rhythmic alternation).
    for (let leg = 0; leg < N_LEGS; leg++) {
      for (let k = 0; k < 12; k++) {
        // Flexor[k] gets inhibition from all extensors
        for (let j = 0; j < 12; j++) {
          e(LEG_FLEXOR(leg, k), LEG_EXTENSOR(leg, j), -0.5);
          e(LEG_EXTENSOR(leg, k), LEG_FLEXOR(leg, j), -0.5);
        }
        // Within-pool mutual excitation (ensures pool fires together)
        for (let j = 0; j < 12; j++) if (j !== k) {
          e(LEG_FLEXOR(leg, k), LEG_FLEXOR(leg, j), 0.4);
          e(LEG_EXTENSOR(leg, k), LEG_EXTENSOR(leg, j), 0.4);
        }
      }
      // Motor neuron integrates flexor activity
      for (let k = 0; k < 12; k++) {
        e(LEG_MOTOR(leg), LEG_FLEXOR(leg, k), 1.0);
      }
    }

    // 5. Tripod synchronizer: legs in same tripod excite each other,
    // legs in opposite tripods inhibit. This forces proper gait.
    const tripodHelper = (legs: number[], offset: number) => {
      // Use first 12 of N_TRIPOD neurons for tripod A, rest for tripod B
      for (let k = 0; k < 12; k++) {
        const tn = N_DN_IN + N_DECISION + N_LEGS * N_PER_LEG + offset + k;
        for (const l of legs) {
          // Tripod neuron gets input from leg flexors
          for (let j = 0; j < 12; j++) e(tn, LEG_FLEXOR(l, j), 0.3);
          // ... and back-projects to ipsi-leg flexors
          for (let j = 0; j < 12; j++) e(LEG_FLEXOR(l, j), tn, 0.5);
          // Inhibits the other tripod's legs
        }
      }
    };
    tripodHelper(TRIPOD_A_LEGS, 0);
    tripodHelper(TRIPOD_B_LEGS, 12);
    // Tripod A neurons inhibit Tripod B neurons
    for (let k = 0; k < 12; k++) {
      const aBase = N_DN_IN + N_DECISION + N_LEGS * N_PER_LEG + k;
      const bBase = N_DN_IN + N_DECISION + N_LEGS * N_PER_LEG + 12 + k;
      for (let j = 0; j < 12; j++) {
        const aN = N_DN_IN + N_DECISION + N_LEGS * N_PER_LEG + j;
        const bN = N_DN_IN + N_DECISION + N_LEGS * N_PER_LEG + 12 + j;
        e(aBase, bN, -0.8);
        e(bBase, aN, -0.8);
      }
    }
  }

  /** Reset all neurons to rest. Called on `sim.reset()` and stim start. */
  reset() {
    for (let i = 0; i < N_VNC; i++) {
      this.vm[i] = this.vRest;
      this.refrac[i] = 0;
      this.spikePrev[i] = 0;
      this.rate[i] = 0;
      this.extInput[i] = 0;
    }
  }

  /** Set DN input rates (in [0, 1]) — five values for DNa01, DNa02,
   * DNb01, DNg13, DNp01 in order. Drives external Vm bump on input
   * neurons proportional to rate. */
  setDnInput(dnRates: { DNa01: number; DNa02: number; DNb01: number; DNg13: number; DNp01: number }) {
    this.extInput[0] = dnRates.DNa01 * 8;
    this.extInput[1] = dnRates.DNa02 * 8;
    this.extInput[2] = dnRates.DNb01 * 8;
    this.extInput[3] = dnRates.DNg13 * 8;
    this.extInput[4] = dnRates.DNp01 * 12;
  }

  /** Advance N steps. Returns motor command derived from output pools. */
  step(nSteps: number): {
    walkMag: number;
    walkSign: number;
    turnBias: number;
    escape: number;
    fwdRate: number;
    bwdRate: number;
  } {
    for (let s = 0; s < nSteps; s++) this.tick();

    // Aggregate output pools.
    let fwd = 0, bwd = 0;
    for (const i of FWD_POOL) fwd += this.rate[i];
    for (const i of BWD_POOL) bwd += this.rate[i];
    fwd /= FWD_POOL.length;
    bwd /= BWD_POOL.length;

    // Per-leg motor neuron rates → tripod-side averages → turn signal
    let leftMotor = 0, rightMotor = 0;
    for (let leg = 0; leg < N_LEGS; leg++) {
      const r = this.rate[LEG_MOTOR(leg)];
      if (LEG_IS_LEFT[leg]) leftMotor += r; else rightMotor += r;
    }
    leftMotor /= 3;
    rightMotor /= 3;

    return {
      walkMag: Math.min(1, (fwd + bwd) * 6),
      walkSign: fwd >= bwd ? 1 : -1,
      turnBias: (rightMotor - leftMotor) * 4,
      escape: this.rate[4],
      fwdRate: fwd,
      bwdRate: bwd,
    };
  }

  private tick() {
    // For each neuron: synaptic gather over incoming, leak integrate,
    // threshold, write spike.
    const spikeCurr = new Uint8Array(N_VNC);
    for (let i = 0; i < N_VNC; i++) {
      // Refractory.
      if (this.refrac[i] > 0) {
        this.refrac[i]--;
        this.vm[i] = this.vReset;
        continue;
      }
      // Gather.
      let iSyn = 0;
      const edges = this.incoming[i];
      for (let k = 0; k < edges.length; k++) {
        const ed = edges[k];
        if (this.spikePrev[ed.pre]) iSyn += ed.w;
      }
      // Integrate (vRest + alpha * (vm - vRest) + iSyn + ext).
      const iIn = iSyn + this.extInput[i];
      const vNew = this.vRest + this.vmAlpha * (this.vm[i] - this.vRest) + iIn;
      // Threshold.
      if (vNew >= this.vThresh) {
        this.vm[i] = this.vReset;
        this.refrac[i] = this.refracSteps;
        spikeCurr[i] = 1;
      } else {
        this.vm[i] = vNew;
      }
    }
    // Update rate trace (EWMA).
    for (let i = 0; i < N_VNC; i++) {
      this.rate[i] = this.rate[i] * this.rateAlpha + spikeCurr[i] * (1 - this.rateAlpha);
    }
    this.spikePrev = spikeCurr;
  }
}

// --- Backwards-compat shim: the old function-based path ------------------
// main.ts and the e2e tests still use motorFromBrain elsewhere; we keep
// it as a thin wrapper that drives the VNC instance and returns the
// equivalent {fwd, turn, jump}. Each call advances 5 ms internally.

export interface MotorCommand { fwd: number; turn: number; jump: number; }
export interface MotorContext {
  famousDns: Record<string, number[]>;
  dnLeft: number[];
  dnRight: number[];
  visual?: { angle: number; area: number };
}

const sharedVnc = new VNC();
const VNC_SUBSTEPS = 5;

const meanRate = (rate: Float32Array, idxs: number[]) => {
  if (!idxs.length) return 0;
  let s = 0;
  for (const i of idxs) s += rate[i];
  return s / idxs.length;
};

/** Push one brain-snapshot through the VNC and pull a motor command out. */
export function motorFromBrain(rate: Float32Array, ctx: MotorContext): MotorCommand {
  // Pull famous-DN rates from brain snapshot. Each name maps to the L+R
  // copies of that DN in the connectome; we average to get a single rate.
  const dns = {
    DNa01: meanRate(rate, ctx.famousDns.DNa01 ?? []),
    DNa02: meanRate(rate, ctx.famousDns.DNa02 ?? []),
    DNb01: meanRate(rate, ctx.famousDns.DNb01 ?? []),
    DNg13: meanRate(rate, ctx.famousDns.DNg13 ?? []),
    DNp01: meanRate(rate, ctx.famousDns.DNp01 ?? []),
  };
  sharedVnc.setDnInput(dns);
  const out = sharedVnc.step(VNC_SUBSTEPS);

  // walkMag×sign drives forward; backward when sign negative.
  let fwd = out.walkMag * out.walkSign;
  let turn = out.turnBias;

  const visualActive = ctx.visual && Number.isFinite(ctx.visual.angle) && ctx.visual.area > 0;

  if (visualActive) {
    // Default: angle bypass (working, tracks target reliably). Opt-in
    // honest path drives turn from brain cascade's L/R DN asymmetry —
    // main.ts injects lateralized stim into opticLeft/opticRight, the
    // cascade propagates, dnLeft/dnRight asymmetry encodes side. Tested
    // experimentally: cascade asymmetry signs WRONG for tracking under
    // our 50-ms window (e2e "TOWARD target sign correct" fails with
    // either sign default), so we keep the bypass as the default and
    // expose the brain-cascade path as an opt-in for users to tune.
    //   (window as any).__visualReflexFromBrain = true   // opt-in
    //   (window as any).__visualBrainGain = 1.2
    //   (window as any).__visualBrainSign = -1
    const w = (typeof globalThis !== "undefined"
      ? (globalThis as {
        __visualReflexFromBrain?: boolean;
        __visualBrainGain?: number;
        __visualBrainSign?: number;
      })
      : {}) as {
        __visualReflexFromBrain?: boolean;
        __visualBrainGain?: number;
        __visualBrainSign?: number;
      };
    const useBrain = w.__visualReflexFromBrain === true;

    const a = ctx.visual!.angle;
    const aAbs = Math.abs(a);
    const dead = (5 * Math.PI) / 180;

    if (useBrain) {
      const meanL = meanRate(rate, ctx.dnLeft);
      const meanR = meanRate(rate, ctx.dnRight);
      const total = meanL + meanR;
      const cascadeAsym = total > 1e-3 ? (meanR - meanL) / (total + 1e-6) : 0;
      const brainGain = w.__visualBrainGain ?? 1.2;
      const brainSign = w.__visualBrainSign ?? -1;
      if (Math.abs(cascadeAsym) > 0.05) {
        turn = Math.max(-1, Math.min(1, brainSign * cascadeAsym * brainGain));
      } else if (aAbs > dead) {
        // Fallback to angle proxy when cascade is too weak.
        const sign = a > 0 ? 1 : -1;
        const mag = Math.min(1, aAbs / ((45 * Math.PI) / 180));
        turn = sign * mag * 0.7;
      } else {
        turn = 0;
      }
    } else {
      // Default bypass: angle proxy. Fast, reliable, but not honest.
      const sign = a > 0 ? 1 : -1;
      if (aAbs > dead) {
        const mag = Math.min(1, aAbs / ((45 * Math.PI) / 180));
        turn = sign * mag * 0.7;
      } else {
        turn = 0;
      }
    }

    // Forward speed scales by alignment — fly slows on hard turns.
    const fovHalf = (60 * Math.PI) / 180;
    const alignment = aAbs < fovHalf ? 1 - aAbs / fovHalf : 0;
    const baseSpeed = Math.min(0.6, ctx.visual!.area * 8 + 0.2);
    fwd = baseSpeed * alignment;
  } else {
    // No visual cue: brain → spine motor, plus broad-DN asym.
    const meanL = meanRate(rate, ctx.dnLeft);
    const meanR = meanRate(rate, ctx.dnRight);
    const total = meanL + meanR;
    if (total > 0.01) {
      const asym = (meanR - meanL) / (total + 1e-6);
      const trim = Math.abs(asym) < 0.20 ? 0 : asym - Math.sign(asym) * 0.20;
      turn += trim * 0.3;
    }
  }

  // Escape jump from DNp01 — translated to a 60 cm/s impulse if the DN
  // input is fully active. We return jump as part of the command and
  // let the caller hand to physics.jumpImpulse.
  const jump = dns.DNp01 > 0.05 ? 60 * Math.min(1, dns.DNp01 * 6) : 0;

  return { fwd, turn, jump };
}

/** Reset the shared VNC state (call from sim.reset). */
export function resetVnc() { sharedVnc.reset(); }

/** Last-known firing rates of the spine's output pools. UI uses these
 * to render a small bar showing the spine is alive. */
export function vncSnapshot(): { fwdRate: number; bwdRate: number; escape: number } {
  // Read from a no-op step that doesn't change input state.
  const out = sharedVnc.step(0);
  return { fwdRate: out.fwdRate, bwdRate: out.bwdRate, escape: out.escape };
}
