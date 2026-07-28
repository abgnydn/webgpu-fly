# Limitations — what webgpu-fly cannot do, what is untested, what is approximate

This is the honest single-page companion to the README. Everything here is
real and most of it is also documented in commit messages. If a claim in the
README sounds impressive, this file is where the asterisks live.

The one-line summary: **webgpu-fly is the most *reachable* fly-brain
simulator, not the most *accurate* one.** It trades scientific completeness
for "a real connectome on a phone in 30 seconds." Treat it as a teaching,
demonstration, and intuition-building tool — not as a validated platform for
publishing fly-brain dynamics.

---

## 1. Performance — what "0.25 kHz" means and doesn't

- The brain LIF kernel runs at **~0.25 kHz of biological time** on an Apple
  M2 Pro (16 GB). That is **4× slower than real time** for a 1 kHz
  biological target. It is **not** a faster-than-reference claim.
- It is **memory-bandwidth-bound**, not compute-bound. The gather over CSR
  rows dominates; adding more GPU ALUs would not help.
- Honest cross-checks on the **same machine**: NEST 3.10 (the C++ scientific
  standard) hit **0.67 kHz**, a hand-written multicore Rust port hit
  **0.45 kHz**, this WebGPU version **0.25 kHz**. The original 1 kHz target
  from the project's first design note was **unreachable for all three** on
  M2 Pro. We report the measurements, not the target. See `tools/cpu-bench/`
  and `tools/nest_bench.py`; reproduce the WebGPU number with
  `npm run bench:brain`.
- These are single-machine numbers on one GPU. We have **not** benchmarked
  across NVIDIA / AMD / Intel adapters, or across browsers.

## 2. The model is LIF, and deliberately simple

- Neurons are **leaky integrate-and-fire** with a two-state alpha synapse.
  No Hodgkin-Huxley channels, no dendritic compartments, no spatial synapse
  positions, no neuromodulation dynamics.
- We use the **aggregated** proofread connection table (~15M unique (pre,
  post) pairs, with each pair's synapse count summed into its weight),
  **not** the ~54M raw synapses. v1 LIF does not use per-synapse spatial
  position. Dendritic compartment models would require switching to the much
  larger raw table.
- **Neurotransmitter → sign is a hard mapping**, baked into the weights at
  build time: acetylcholine → +1, GABA/glutamate → −1, and the modulatory
  transmitters (dopamine, serotonin, octopamine) plus any prediction below
  0.5 confidence → 0 (silent). This is a coarse approximation: glutamate is
  "mostly" inhibitory in fly via GluCl but not always, and modulation is
  simply out of scope for v1. Changing the confidence threshold requires
  rebuilding `brain.bin`.

## 3. Dynamics validation is qualitative, not quantitative

- We check the **shape** of the dynamics: no-input networks go silent (no
  runaway), Kenyon cells fire sparsely (~5–15%) under sensory drive,
  consistent with Shiu et al. 2024.
- We have **not** done a quantitative, cell-type-resolved firing-rate match
  against a published reference simulation across the whole brain. The
  sparsity check is the strongest dynamics claim we stand behind.

## 4. The brain → spine → body path has documented approximations

These are the "honest gaps" from the README, restated as limitations:

1. **Trained RL walker walks slower than native.** The published policy
   checkpoint shipped **without** its `ObservationActionNorm` running
   mean/std, so the policy expects raw observations. We feed raw obs (which
   matches native best) or, optionally, a rollout-derived norm; either keeps
   actions non-saturating, but per-simulated-second walking speed is roughly
   **half** of native flybody. Part of the remaining gap is sim-time vs
   wall-time budget — we run ~25% of real time at 60 fps.
2. **Closed-loop visual reflex is approximated.** When the target is visible,
   the default path can fall back to `turn ∝ retinal angle` because the
   brain's genuine optic→DN contralateral cascade takes longer than our 50 ms
   tick to develop. An opt-in "honest mode" replaces this with a
   brain-cascade-derived turn (dnLeft/dnRight asymmetry), which is slower and
   weaker but real. The shortcut is a usability default, not a biological
   claim.
3. **Optional kinematic assist on the body.** In CPG mode (RL policy off),
   a soft assist on the freejoint translation, scaled by motor command, makes
   the demo watchable. It is **off** in RL-policy mode and can be turned off
   everywhere via the "Honest mode" button (`Physics.kinematicAssistEnabled`).
   With it off, the body moves only via actuator → ground reaction.
4. **Reference walking trajectory.** The trained policy expects a reference
   trajectory; the default is a procedural straight line. Real fly mocap from
   the Vaxenburg deposit is wired in as opt-in (`__walkingRefFromMocap`).

## 5. Brain ↔ spine wiring is name-match, not synaptic

- The brain and ventral nerve cord are **separate** connectomes from
  **different animals** (FlyWire = female FAFB brain; MANC = male VNC). They
  are joined by **cell-type name match** — `DNa01` in the brain is treated as
  the same neuron as `DNa01` in the VNC. This is biologically motivated (the
  soma is in the brain, the axon in the cord) but it is a **name join, not a
  reconstructed synaptic bridge**. There is no electron-microscopy continuity
  across the boundary; none exists in public data yet.

## 6. Deployment / browser constraints

- **WebGPU is required.** Works on Chrome and Edge; Safari needs a recent
  version with WebGPU enabled; Firefox WebGPU is still maturing. No WebGL
  fallback — if WebGPU is absent, the app does not run.
- **First load streams ~300 MB** (brain + spine + body assets from R2),
  cached afterward via HTTP immutable cache + IndexedDB. On a cold cache and
  a slow connection this is a real wait.
- The deployment is **memory-hungry** in the tab during boot; low-RAM mobile
  devices may fail to allocate the connectome buffers.

## 7. What is NOT claimed

- Not a NEST / Brian2 / NEURON replacement for scientific simulation.
- Not a quantitatively validated whole-brain dynamics reference.
- Not biophysically detailed (no channels, compartments, or modulation).
- Not a synaptically continuous brain-to-body reconstruction.
- Not benchmarked beyond a single M2 Pro / Chromium configuration.

---

*Found something here that's worse than described, or a claim in the README
that this file doesn't cover? That's a valid issue — open one. Honest
negatives are first-class in this repo.*
