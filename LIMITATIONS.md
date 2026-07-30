# Limitations — what webgpu-fly cannot do, what is untested, what is approximate

This is the honest single-page companion to the README. Everything here is
real and most of it is also documented in commit messages. If a claim in the
README sounds impressive, this file is where the asterisks live.

The one-line summary: **webgpu-fly is the most *reachable* fly-brain
simulator, not the most *accurate* one.** It trades scientific completeness
for "a real connectome on a phone in 30 seconds." Treat it as a teaching,
demonstration, and intuition-building tool — not as a validated platform for
publishing fly-brain dynamics.

A second summary, for the body specifically: **the connectome does not walk
the fly.** Roughly 20.3M connectome edges reach the body as about one scalar
magnitude plus a turn bias per tick, and those two numbers scale a hand-written
`sin(t × 10 Hz)` tripod. §8 is the complete shortcut inventory, with measured
numbers for what the body does once the assist is off.

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

These are the "honest gaps" from the README, restated as limitations. §8 is
the complete list; these four are the ones with the longest history.

1. **Trained RL walker does not walk.** The published policy checkpoint
   shipped **without** its `ObservationActionNorm` running mean/std, so the
   policy expects raw observations. We feed raw obs (which matches native
   best) or, optionally, a rollout-derived norm; either keeps actions
   non-saturating (|action| max ≈ 7). But every browser speed number ever
   recorded for this path — including the "roughly half of native" figure
   this item used to carry — was measured with the kinematic assist active
   (item 3), which writes body velocity directly. With the assist off, the
   fly capsizes within ~1.5 s of the policy being enabled and stays on its
   back: uprightness −0.851 → −0.978, net travel 0.057–0.317 cm over ~2
   simulated seconds. The network is self-consistent and the actions are
   in-distribution; the defect is downstream of it — observation
   construction, plant, or initial pose. See §8.
2. **Closed-loop visual reflex is a hand-written angle law.** When the target
   is visible, the default path is `turn ∝ retinal angle`
   (`src/vnc.ts:369-378`), and forward speed is set from retinal area and
   alignment regardless of branch (`src/vnc.ts:380-384`). The brain's genuine
   optic→DN contralateral cascade takes longer than our 50 ms tick to
   develop. An opt-in "honest mode" switches to a brain-cascade-derived turn
   (dnLeft/dnRight asymmetry) — but that path falls back to the identical
   hand-written law whenever cascade asymmetry is below 0.05, and the code
   records the cascade's sign as empirically wrong for tracking under our
   window (`src/vnc.ts:322-326`). It is an opt-in to a path that does not
   currently work, not a working honest alternative.
3. **Kinematic assist on the body — on by default, and it is the
   locomotion.** A direct write to the freejoint sets translation and yaw
   velocity from the motor command (`qvel[0]`, `qvel[1]`, `qvel[5]`,
   re-asserted every substep, `src/physics.ts:596-604`). It is **not** off in
   RL-policy mode: `fwdCmd`/`turnCmd` are written only by `driveLegs`
   (`src/physics.ts:717-718`), the policy path skips `driveLegs`
   (`src/room.ts:633-647`), and nothing clears them — not `reset()`
   (`src/physics.ts:991-1010`) either — so the last CPG command keeps driving
   the body throughout "trained walking". The "Honest mode" button turns the
   assist off everywhere (`Physics.kinematicAssistEnabled`). With it off the
   body's translation comes only from actuator → ground reaction, but the
   pitch/roll damper (§8) still runs.
4. **Reference walking trajectory.** The trained policy expects a reference
   trajectory; the default is a procedural straight line. Real fly mocap from
   the Vaxenburg deposit is wired in as opt-in (`__walkingRefFromMocap`), but
   it is baked at 50 Hz (`tools/bake_walking_ref.py:46`) and replayed one
   frame per 2 ms control tick (`src/physics.ts:896`), a 10× rate error, and
   the 65-frame lookahead exceeds the 57-frame trajectory so the window wraps
   mid-observation. Measured: enabling it takes |action| max from ~7 to
   3097–3250 and the fly spins 944–1051° in ~1.8 simulated seconds. The
   opt-in currently makes the policy's input further out of distribution,
   not closer.

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

## 8. Full shortcut inventory — what actually moves the body

The "Honest mode" button flips exactly three flags (`src/main.ts:669-683`).
The table below has thirteen rows, and nine of them are behind no toggle at
all. This section is all of them — the ones the button covers and the ones it
does not — so the button is not the only place they are disclosed.

Nothing here is a claim about the neural simulation. The FlyWire and MANC
connectomes are real, both LIF networks genuinely run on the GPU, and the
stimulus→cascade dynamics are connectome-derived. What follows is about how
the *body* is driven, which is a different and much weaker story.

### The information bottleneck

Start here, because it frames every row of the table:

```
FlyWire   139,255 neurons / 15,091,983 edges
   ↓  name join: 7 DN types → 16 of 23,188 VNC neurons (0.069%)  main.ts:383-391
   ↓  L and R copies collapsed to one scalar each                main.ts:387-390
MANC       23,188 neurons /  5,243,574 edges
   ↓  369 leg motor neurons → 6 leg-group means                  main.ts:396-400
   ↓  → mancTotal, mancAsym                                      main.ts:452-453
   ↓  and the direction sign is discarded — it comes from the
      hand-wired 200-neuron synthetic spine                      main.ts:470
   ↓  driveLegs(walk, turn)                                      room.ts:640
   =  sin(t · 10 Hz), 18 of 48 leg actuators                     physics.ts:663-719
```

**Roughly 20.3 million connectome edges reach the body as about one scalar
magnitude plus a turn bias per tick, and those two numbers scale a
`sin(t × 10 Hz)` tripod.** In the "Track target" closed loop the connectome
contribution to locomotion is zero: MANC is skipped entirely
(`src/main.ts:457-465`) and both forward and turn are overwritten by retinal
geometry (`src/vnc.ts:369-384`).

### The inventory

| Shortcut | Substitutes for | Honest mode? | Where |
|---|---|---|---|
| **Kinematic assist** — writes freejoint `qvel[0]`, `qvel[1]`, `qvel[5]` from `fwdCmd`/`turnCmd`, re-asserted every substep before `mj_step` | ground reaction from leg contact | **yes** (`Physics.kinematicAssistEnabled`) | `src/physics.ts:576-604`, default on at `:554` |
| **Assist is live in RL-policy mode** — `fwdCmd`/`turnCmd` are written only by `driveLegs`, the policy path skips `driveLegs`, and nothing zeroes them (including `reset()`), so a stale CPG command keeps driving the body under the policy | — | only insofar as it turns the assist off globally; nothing else clears the commands | `src/physics.ts:717-718`, `src/room.ts:633-647`, `src/physics.ts:991-1010` |
| **Pitch/roll attitude damper** — `qvel[3] *= 0.85; qvel[4] *= 0.85` per substep, ×0.039 per 2 ms control tick | balance, and the body's ability to tip at all | **no** — it sits before and outside the assist guard | `src/physics.ts:592-595` |
| **Boot stimulus drives itself** — science mode auto-runs `STIMULI[0]` at load, saturating the spine to `fwdCmd = 0.99999` for the length of its window; the drive is put back to rest when that window ends, so the residual no longer survives to the first user click. `decayDrive()` is defined and never called (one grep hit, the definition) | a brain whose drive responds to what you click | **no** | `src/main.ts:1287-1291`, `src/main.ts:499` |
| **Tripod CPG is the source of leg timing** — `phase = data.time · 10 Hz`, hard-coded gait constants, 3 of 8 DOFs driven per leg | motor-neuron output setting stance/swing | **no** | `src/physics.ts:654-661`, `:663-719`, actuator cache `:213-220` |
| **Wing motion is hand-written** — 218 Hz analytic stroke, amplitude hard-capped at ×0.2 of flybody's canonical pattern because anything above ~0.25 launches the freejoint body | wing motor neurons (MANC's 66 are read for the readout only) | **no** | `src/physics.ts:494-518`, cap at `:504` |
| **`jumpImpulse` writes `qvel[2]` directly** | leg extension producing a takeoff | **no** | `src/physics.ts:979-981`, called from `src/main.ts:495` |
| **Adhesion clamped to 1.0** at init and whenever walk drive < 0.01 — a standing fly is glued to the floor | claw contact and friction holding a stationary fly | **no** | `src/physics.ts:221-227`, `:709-712` |
| **Visual-reflex angle bypass** — `turn ∝ retinal angle`, forward speed from retinal area | the brain's optic→DN contralateral cascade | **yes**, but the brain path falls back to the identical law when cascade asymmetry < 0.05, and the code records the cascade's sign as empirically **wrong** for tracking | `src/vnc.ts:369-378`; brain path `:352-368`; sign note `:322-326` |
| **Sweep-mode spine bypass** — target lost for 4+ ticks writes a scripted alternating scan turn straight to the body | search behaviour emerging from the brain | **no** | `src/main.ts:998-1005` |
| **Walking reference** — synthetic open-loop ramp by default; the mocap opt-in is baked at 50 Hz and replayed at 500 Hz, and the 65-frame lookahead exceeds the 57-frame clip | the policy's training reference clip | switches to mocap, which measures **worse** (§4.4) | `src/physics.ts:905-919`, `:896`; `tools/bake_walking_ref.py:46` |
| **"Evolve gait (WebGPU ARS)" does not evolve against MuJoCo** — the fitness is a 1-D point-mass rollout with analytic thrust and quadratic drag: no gravity, no ground contact, no body — and the winner is written into the live physics body | optimizing the gait against the actual simulated fly | **no** | `src/shaders/evolve.wgsl:4-6`, `:101-104`; applied at `src/main.ts:1054-1056` |
| **The speed readout displays the assist** — `bodySpeed` reads `qvel[0..1]`, the exact slots the assist writes immediately before `mj_step` | measured locomotion | **no** | `src/physics.ts:985-989`, rendered at `src/main.ts:739,746` |

One more, about evidence rather than physics: **the walking policy's forward
pass is not verified against the published SavedModel.**
`tools/verify_walking_policy.py` imports only `json`, `struct`, `pathlib` and
`numpy` — no TensorFlow. It reads `public/walking-policy.bin`, the file
`extract_walking_policy.py` wrote, and re-implements the same *assumed*
architecture. The passing fixture therefore shows TS ≡ a numpy
re-implementation of the same guess, not TS ≡ flybody's deployed policy. The
extractor's own comments record the guess as unresolved
(`tools/extract_walking_policy.py:43`, "layernorm bias (?) — actually
unsure").

### What the body does with the assist off (measured)

16 controlled browser runs, headed Chromium, one M-series Mac, ~2 simulated
seconds of sampling per window. "cm/sim s" is net displacement per *simulated*
second (the portable number; wall-clock distance is machine-dependent).
"Upright" is the body z-axis' world-z component: +1 upright, −1 upside down.
These runs were sampled while the boot stimulus still left its saturated drive
in place, so the two "boot residual" rows and the DNa01 delta below describe a
build whose idle forward command was 1.000; `src/main.ts:1287-1291` now returns
it to zero. The assist-on/assist-off contrast, which is what the table is for,
is unaffected — it is measured within each row.

| Controller | Assist | cm/sim s | Straightness | Total yaw | Upright |
|---|---|---|---|---|---|
| none (boot residual only) | ON | 0.820 / 0.815 | 0.87 / 0.86 | +99° / +100° | 0.98 → 1.00 |
| none (boot residual only) | OFF | 0.052 / 0.053 | 0.036 | −258° / −259° | 0.98 → 0.98 |
| hand-coded CPG (DNa01) | ON | 0.798 / 0.798 | 0.84 | +113° | 0.98 → 0.98 |
| hand-coded CPG (DNa01) | OFF | 0.071 / 0.074 | 0.048 / 0.049 | −244° / −245° | 0.99 → 0.98 |
| trained RL policy | ON | 0.878 / 0.670 | 0.93 / 0.69 | −68° / +152° | **+0.057 → −0.944** / 0.97 → 0.80 |
| trained RL policy | OFF | 0.029 / 0.163 | 0.53 / 0.48 | −108° / +138° | **−0.851 → −0.978** |

Reading it:

- **The assist is the locomotion, and it does not care what the controller is
  doing.** 0.798–0.878 cm/sim s with the assist on across three completely
  different controller states — no controller at all, the hand-coded CPG, and
  the trained RL policy. With it off, the same three give 0.029–0.163
  cm/sim s. A 4×–30× collapse, and the assist-on number is just the 1.0 cm/s
  command minus what yaw and MuJoCo take back.
- **Displacement is decoupled from body state.** In one assist-on run the
  fly's uprightness went from +0.057 to −0.944 — it turned over — and it still
  translated at 0.878 cm/sim s with straightness 0.93. A fly gliding smoothly
  forward on its back at the commanded speed.
- **Assist off is not a slow walk; the direction is wrong.** The CPG
  accumulates 3.43 cm of path length for 0.163–0.170 cm of net displacement,
  straightness 0.048, total yaw −245°. The fly pirouettes in place. The legs
  do move and do couple to the ground; the net effect is rotation and jitter.
- **The DNa01 button contributes ~nothing to forward motion.** A page nobody
  clicked travels 1.831 / 1.858 cm; after clicking DNa01, 1.838 / 1.838 cm — a
  0.4% difference. `fwdCmd` was already pinned at 1.000 by the boot residual
  these runs still carried; DNa01 moved only `turnCmd` (0.012 → 0.119). What
  the measurement shows about the button is that its forward axis is a no-op
  whenever anything has already saturated the drive.
- **Nothing is numerically unstable.** Zero non-finite `qpos` entries across
  all 16 runs; vertical drift within 0.16 cm of spawn everywhere. The
  simulation is healthy and simply produces no net thrust.
- On the CPG path, "Honest mode" is *exactly* equivalent to "assist off" — the
  honest-mode CPG runs reproduce the plain assist-off CPG runs to 3–4
  significant figures, because the other two flags are only read in code paths
  CPG mode never enters.

The pitch/roll damper is not in that table, because **the project has never
been run with it off.** It has been ×0.005 per CPG render frame for the entire
life of the codebase, in every mode including Honest mode, so no measurement
here — or in any commit message — describes a fly that could tip over. That
baseline is unmeasured.

---

*Found something here that's worse than described, or a claim in the README
that this file doesn't cover? That's a valid issue — open one. Honest
negatives are first-class in this repo.*
