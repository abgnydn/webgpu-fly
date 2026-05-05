# webgpu-fly

A real fly brain talking to a real fly spinal cord talking to a real fly body, live in your browser tab.

**Demo:** https://webgpu-fly.pages.dev — no install, just a URL. Tested in Chrome / Edge desktop with WebGPU enabled (default since 2023).

## What's actually inside

| Layer | Source | What runs |
|---|---|---|
| Brain | [FlyWire FAFB](https://flywire.ai/) connectome (Dorkenwald et al. 2024, Nature) | 139,255 neurons, 15M synaptic edges, two-state alpha-synapse LIF on WebGPU |
| Spine | [Janelia MANC](https://www.janelia.org/project-team/flyem/manc-connectome) connectome (Takemura et al. 2024) | 23,188 VNC neurons, 5.2M edges, second WebGPU LIF instance |
| Body | [TuragaLab/flybody](https://github.com/TuragaLab/flybody) MJCF (Vaxenburg et al. 2025, Nature) | 67 bodies, 111 actuators, real physics in mujoco_wasm |
| Eyes | offscreen render-to-texture from fly head pose | 64×16 retinal sample fed to brain optic neurons |
| Walker | trained RL policy from the Vaxenburg et al. 2025 [Figshare deposit](https://janelia.figshare.com/articles/dataset/MuJoCo_fruit_fly_body_model_datasets_supporting_Whole-body_simulation_of_realistic_fruit_fly_locomotion_with_deep_reinforcement_learning_/25309105) | 4-hidden-layer LayerNormMLP, 741-dim observation → 59 actions, pure-TS forward pass |

The brain's descending neurons feed the VNC by cell-type name match (DNa01 in brain ↔ DNa01 in VNC are the same neuron, brain side has the soma, VNC side has the axon). The VNC's motor neurons drive the leg actuators. Wire-by-wire from real biology, no learned shortcuts in the brain→spine path.

## What you can click

- **5 stim presets** (Visual flash / Olfactory hit / Mixed sensory / Spontaneous / DN buttons) — fires real neurons in the connectome, watch the cascade in the heatmap
- **5 famous DNs** (DNa01, DNa02, DNb01 moonwalker, DNp01 escape jump, DNg13 turning) with links to [FlyWire Codex](https://codex.flywire.ai)
- **Track target** — closed-loop visual servo: fly's eyes see a red sphere, brain decides where to walk
- **Evolve gait** — 128-policy CMA-ES on WebGPU, ~0.5 sec, winner drives the live body
- **Use RL policy** — the trained Vaxenburg 2025 walker; click and the body walks via 59 RL actions instead of the hand-coded CPG

Right-click anywhere in the room to drop the red target.

## How it compares

This is roughly the same idea as:
- [EON Systems](https://eon.systems) (March 2026) — also FlyWire LIF + flybody MuJoCo, but server-side and closed
- [FlyGM](https://arxiv.org/abs/2602.17997) (NeurIPS 2025) — connectome-as-GNN trained controller, Python
- [NeuroMechFly v2 / flygym](https://github.com/NeLy-EPFL/flygym) (Nature Methods 2024) — Python pip package, dm_control-based

The differentiators here:
- **Public, browser-tab demo**. The others need Python, a GPU, and time. This needs a URL.
- **All three layers** (brain + spine + body) wired together, not just brain+body.
- **22-test Playwright e2e suite** asserts KC sparsity in literature range, DN cascade behavior, retina detection, RL walker forward translation, etc. Catches regressions automatically.

## Quickstart (local dev)

```bash
# Pull raw FlyWire data (~855 MB, Zenodo)
bash tools/download_data.sh
python3 tools/build_csr.py        # → public/brain.bin (120 MB)

# Pull MANC spine data (~88 MB, Janelia GCS)
bash tools/download_manc.sh
.venv-tf/bin/python tools/build_vnc.py    # → public/vnc.bin (43 MB)

# Pull trained walking policy (~6.5 MB, Janelia Figshare)
bash tools/download_flybody_policies.sh
.venv-tf/bin/python tools/extract_walking_policy.py
                                  # → public/walking-policy.bin (4.8 MB)

# Pull TuragaLab flybody MJCF + 85 OBJ meshes (~149 MB)
# (see public/flybody/meshes.txt for the canonical fetch pattern)

# Run
npm install
npm run dev                       # http://localhost:8766
npm run test:e2e                  # 22 Playwright tests, ~15 min
```

## Deploy

`DEPLOY.md` covers the full Cloudflare Pages + R2 path. Heavy assets
(brain.bin, vnc.bin, flybody, walking-policy.bin) live in R2; Pages serves
the ~9 MB JS+WASM bundle. Returning visitors get instant loads via the
IndexedDB cache layer in `src/cache.ts`.

## Architecture

```
src/
  brain.ts             FlyWire CSR loader (also reads vnc.bin)
  sim.ts               WebGPU LIF runtime, alpha-synapse kernel
  shaders/lif.wgsl     fused per-step LIF compute kernel
  shaders/evolve.wgsl  parallel CPG evolution kernel
  vnc.ts               synthetic 200-neuron spine (fallback when MANC silent)
  walking-policy.ts    pure-TS forward pass for Vaxenburg 2025 RL policy
  physics.ts           mujoco_wasm wrapper, leg CPG, RL action mapping
  room.ts              three.js scene, retinal render, camera
  evolution.ts         WebGPU ARS gait evolver
  main.ts              UI, brain-stim orchestration, button wiring

tools/
  build_csr.py         FlyWire feather → brain.bin
  build_vnc.py         MANC CSV → vnc.bin (with DN-input + motor-leg metadata)
  extract_walking_policy.py    SavedModel checkpoint → walking-policy.bin
  verify_walking_policy.py     numpy ground-truth verifier (e2e fixtures)
  download_data.sh / download_manc.sh / download_flybody_policies.sh
  upload_to_r2.sh      pushes assets to Cloudflare R2

tests/smoke.spec.ts    22 Playwright e2e tests
```

## Honest gaps

These are real and documented in commit messages — not selling-points to hide:

1. The trained RL walker walks at ~13% of nominal speed because we
   substitute per-call LayerNorm for the trained env's
   `ObservationActionNorm` running mean/std stats. Body walks straight
   and stays upright; just slower than it would in flybody Python.
2. The 85-dim `joints_pos` slice and 59-dim action mapping are
   MJCF-declaration-order guesses. The fact that the body walks at
   all says we're roughly right; close to verified would need a
   flybody Python rollout.
3. CPG-mode (when "Use RL policy" is OFF) has a soft kinematic-assist
   on the freejoint translation scaled by motor command. The
   RL-policy mode has no such assist — actions go straight to actuators.
4. Closed-loop visual reflex hardcodes `turn = angle * k` when target
   is visible because the brain's optic→DN cascade takes longer than
   our 50 ms tick window to develop the contralateral wiring.

## Citations

- Dorkenwald et al. 2024. *Neuronal wiring diagram of an adult brain.* Nature 634:124-138.
- Takemura et al. 2024. *A connectome of the male Drosophila ventral nerve cord.* eLife reviewed preprint.
- Shiu et al. 2024. *Drosophila brain model recapitulates whole-brain LIF dynamics.* Nature 632:210-217.
- Vaxenburg et al. 2025. *Whole-body physics simulation of fruit fly locomotion.* Nature.
- Marin et al. 2024. *Connectomic reconstruction of a female Drosophila ventral nerve cord.* Nature 631:128-140.

## License

MIT for our code. Upstream connectome / body / policy assets carry their
own licenses (CC-BY for FlyWire and MANC; flybody is Apache-2.0; trained
policies are CC-BY).
