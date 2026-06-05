<div align="center">

<a href="https://webgpu-fly.pages.dev"><img src="./public/readme-hero.svg" alt="webgpu-fly — a real Drosophila brain, spinal cord, and body running in a browser tab" width="100%"/></a>

<br/><br/>

<a href="https://webgpu-fly.pages.dev"><img alt="Launch" src="https://img.shields.io/badge/%E2%96%B6%20LAUNCH-webgpu--fly.pages.dev-9ad7ff?style=for-the-badge&labelColor=06070a"/></a>
&nbsp;
<a href="https://webgpu-fly.pages.dev/app?mode=science"><img alt="Science view" src="https://img.shields.io/badge/SCIENCE%20VIEW-%2Fapp%3Fmode%3Dscience-6cd28a?style=for-the-badge&labelColor=06070a"/></a>

<br/><br/>

<a href="https://github.com/abgnydn/webgpu-fly/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/abgnydn/webgpu-fly/actions/workflows/ci.yml/badge.svg?branch=main"/></a>
<img alt="version" src="https://img.shields.io/badge/v0.1.0-0ea5e9?style=flat-square&labelColor=06070a"/>
<img alt="license" src="https://img.shields.io/badge/code-MIT-22c55e?style=flat-square&labelColor=06070a"/>
<img alt="data" src="https://img.shields.io/badge/data-CC--BY%20%2F%20Apache--2.0-c084fc?style=flat-square&labelColor=06070a"/>
<img alt="webgpu" src="https://img.shields.io/badge/WebGPU-required-ff7849?style=flat-square&labelColor=06070a"/>
<img alt="install-free" src="https://img.shields.io/badge/install-0%20bytes-eab308?style=flat-square&labelColor=06070a"/>

</div>

<br/>

You control a fly by firing real **descending neurons** in the
[FlyWire](https://flywire.ai) connectome. The spike cascade propagates through
the real wiring; the fly walks because the connectome says so. There is no
scripted animation — every step you see is an LIF cascade through a real fly's
brain map, into a real fly's spinal cord, driving a physically simulated body.

---

<table border="0">
<tr>
<td valign="top" width="33%">

**What this is**

- A whole-animal *Drosophila* nervous system — brain, spinal cord, and body — running end-to-end in a browser tab on WebGPU, no install and no server.
- Two real connectomes (FlyWire brain + Janelia MANC spine) simulated as leaky integrate-and-fire networks, one fused GPU dispatch per timestep.
- A physically simulated fly body (TuragaLab flybody in MuJoCo/WASM) driven by the spine's motor neurons, with a retina feeding vision back into the brain.
- A game with **replay-as-URL**: a shared link deterministically re-executes the identical neuron cascade against the same connectome.

</td>
<td valign="top" width="33%">

**What this isn't**

- **Not a scientific simulator replacement.** NEST / Brian2 / NEURON are faster, biophysically detailed, and validated. For real fly-brain dynamics research, use those.
- **Not biophysically detailed.** Neurons are LIF with a two-state alpha synapse — no ion channels, dendritic compartments, or neuromodulation.
- **Not quantitatively validated** whole-brain. The dynamics check is qualitative (Kenyon-cell sparsity matches Shiu et al. 2024), not a cell-type-resolved rate match.
- **Not faster than the reference.** It runs *slower* than real time. The win is reachability, not throughput. See [`LIMITATIONS.md`](./LIMITATIONS.md).

</td>
<td valign="top" width="33%">

**Who it's for**

- **The curious public.** A real animal brain you can poke, with a 30-second time-to-first-spike and no setup.
- **Educators.** Every key fires a named command neuron and you watch the consequence ripple to the legs — the connectome made tangible.
- **Connectome / WebGPU folks.** A real ~140k-neuron LIF kernel benchmarked honestly in the browser, with the brain→spine→body path wired from real biology.
- **Anyone who wants reproducibility.** Runs are URLs; a replay link is a verifiable brain trace, not a video.

</td>
</tr>
</table>

---

## ▶ Play

Each key fires both copies of a famous descending neuron (DN). Race the fly to
the red target in the least **time × spikes**.

```
Q DNa01    forward         A DNp01    escape jump     Z RRN     forward step
W DNa02    forward, faster S DNp09    looming-evoked  X BPN     forward
E DNb01    backward        D DNp52    forward circuit
R DNg13    turning         F MDN      backward
                                                       SPACE     new round
                                                       M         science view
```

Win → copy the replay URL. The recipient sees the **identical** simulation —
deterministic seeded target + recorded keystrokes against the same connectome.
Daily-challenge mode uses the same target seed for everyone on the same UTC day.

The landing page at [`/`](https://webgpu-fly.pages.dev) explains the project in
plain language; the simulator itself lives at
[`/app`](https://webgpu-fly.pages.dev/app). Append `?mode=science` (or press
`M` in game) for the researcher interface: stim presets, closed-loop visual
mode, ARS evolver, raw spike-rate log.

---

## 🧠 What's actually inside

| Layer | Source | What runs |
|---|---|---|
| **Brain** | [FlyWire FAFB](https://flywire.ai/) connectome (Dorkenwald et al. 2024, *Nature*) | 139,255 neurons, ~15M synaptic edges, two-state alpha-synapse LIF on WebGPU |
| **Spine** | [Janelia MANC](https://www.janelia.org/project-team/flyem/manc-connectome) connectome (Takemura et al. 2024) | 23,188 VNC neurons, 5.2M edges, second WebGPU LIF instance |
| **Body** | [TuragaLab/flybody](https://github.com/TuragaLab/flybody) MJCF (Vaxenburg et al. 2025, *Nature*) | 67 bodies, 111 actuators, real physics in MuJoCo/WASM |
| **Eyes** | offscreen render-to-texture from fly head pose | 64×16 retinal sample fed to brain optic neurons |
| **Walker** | trained RL policy ([Vaxenburg et al. 2025 Figshare](https://janelia.figshare.com/articles/dataset/25309105)) | LayerNormMLP, 741-dim obs → 59 actions, pure-TS forward pass, verified element-wise vs the published checkpoint |

Brain → spine wiring is by **cell-type name match** (`DNa01` in the brain is the
same neuron as `DNa01` in the VNC — brain side has the soma, VNC side the axon).
VNC motor neurons drive the leg actuators. Wire-by-wire from real biology, no
learned shortcuts in the brain→spine path. (Caveat: it's a name join across two
*different* animals' connectomes, not a reconstructed synaptic bridge — see
[`LIMITATIONS.md`](./LIMITATIONS.md) §5.)

---

## ⏱️ How fast — honestly

The brain LIF kernel is **memory-bandwidth-bound**, not compute-bound, and runs
at **~0.25 kHz** of biological time on an M2 Pro (16 GB). That is **4× slower
than real time**. We measured against the scientific standards on the **same
machine**:

| Engine | Biological-time rate | Notes |
|---|---|---|
| NEST 3.10 (C++ reference) | **0.67 kHz** | `tools/nest_bench.py` |
| Hand-written Rust (multicore) | **0.45 kHz** | `tools/cpu-bench/` |
| **webgpu-fly (WebGPU)** | **0.25 kHz** | `npm run bench:brain` |

The 1 kHz target from the project's original design note turned out to be
**unreachable for all three** on M2 Pro. We report what we measured, not what we
hoped. The point of webgpu-fly was never "faster than NEST" — it's **a real fly
brain on a phone in 30 seconds**, the deployment angle the others can't touch.

---

## 🆚 How it compares

Roughly the same idea as:

- [EON Systems](https://eon.systems) (March 2026) — also FlyWire LIF + flybody MuJoCo, but server-side and closed.
- [FlyGM](https://arxiv.org/abs/2602.17997) (NeurIPS 2025) — connectome-as-GNN controller, Python.
- [NeuroMechFly v2 / flygym](https://github.com/NeLy-EPFL/flygym) (Nature Methods 2024) — Python pip package.

Differentiators: **(1)** a browser-tab game with a URL — the others need Python,
a GPU, and a setup hour; **(2)** all three layers (brain + spine + body) wired
together, not just brain+body; **(3)** replay-as-URL — every shared run is a
deterministic re-execution anyone can verify, study, or remix.

---

## ⏱️ Quickstart (local dev)

```bash
# Brain (~855 MB FlyWire pull from Zenodo)
bash tools/download_data.sh
python3 tools/build_csr.py                    # → public/brain.bin (120 MB)

# Spine (~88 MB MANC pull from Janelia GCS)
bash tools/download_manc.sh
.venv-tf/bin/python tools/build_vnc.py        # → public/vnc.bin (43 MB)

# Trained walking policy (~6.5 MB Janelia Figshare)
bash tools/download_flybody_policies.sh
.venv-tf/bin/python tools/extract_walking_policy.py

# TuragaLab flybody MJCF + 85 OBJ meshes (~149 MB) — see public/flybody/meshes.txt

npm install
npm run dev          # http://localhost:8766
npm run typecheck    # tsc --noEmit          ← what CI enforces
npm run build        # tsc && vite build     ← what CI enforces
npm run test:e2e     # Playwright (game + science modes) — needs WebGPU + assets
```

CI (`.github/workflows/ci.yml`) runs `typecheck` + `build` on Ubuntu. The
Playwright e2e suite is **not** in CI — it needs a WebGPU-capable Chromium and
~1 GB of assets, neither of which the default runners provide. It runs locally
and covers both game and science modes (KC sparsity, DN cascade, retina
detection, RL walker translation, replay roundtrip, daily-challenge).

---

## 🚀 Deploy

[`DEPLOY.md`](./DEPLOY.md) covers the full Cloudflare Pages + R2 path. The root
`/` is a lightweight landing page; the app is at `/app`. Heavy assets
(`brain.bin`, `vnc.bin`, flybody, `walking-policy.bin`) live in R2 because
Cloudflare Pages caps individual files at 25 MB; Pages serves the ~9 MB JS+WASM
bundle. `npm run deploy` builds slim and pushes to Pages; `npm run deploy:r2`
uploads the binaries. Caching is transparent-invalidation via a sha manifest
(`tools/write_manifest.py` → `?v=<sha>`); see DEPLOY.md.

---

## 🧱 Architecture

```
src/
  brain.ts             FlyWire CSR loader (also reads vnc.bin)
  sim.ts               WebGPU LIF runtime, alpha-synapse kernel
  shaders/lif.wgsl     fused per-step LIF compute kernel
  shaders/evolve.wgsl  parallel CPG evolution kernel
  vnc.ts               spine wiring + synthetic fallback
  walking-policy.ts    pure-TS forward pass for Vaxenburg 2025 RL policy
  physics.ts           mujoco_wasm wrapper, leg CPG, RL action mapping
  room.ts              three.js scene, retinal render, camera
  evolution.ts         WebGPU ARS gait evolver
  game.ts              game mode + deterministic replay URLs
  main.ts              UI, brain-stim orchestration, button wiring

tools/
  build_csr.py         FlyWire feather → brain.bin (authoritative binary spec)
  build_vnc.py         MANC CSV → vnc.bin (DN-input + motor-leg metadata)
  extract_walking_policy.py    SavedModel checkpoint → walking-policy.bin
  dump_flybody_spec.py live flybody Walking env → canonical orderings
  nest_bench.py / cpu-bench/   NEST + Rust baselines for the perf table
  upload_to_r2.sh      Cloudflare R2 push with immutable Cache-Control
```

The authoritative `brain.bin` binary-format spec lives in the
`tools/build_csr.py` docstring.

---

## 🔬 For researchers

- **How to cite** — see [`CITATION.cff`](./CITATION.cff). The repo is set up to
  mint a Zenodo DOI on the first GitHub release (`.zenodo.json` pre-populates
  the deposit); the DOI badge is wired in here once minted.
- **[Limitations](./LIMITATIONS.md)** — the honest single-page list of what
  this *cannot* do, what is *untested*, and what is *approximate*: the perf
  ceiling, the LIF simplifications, the brain→spine→body approximations, and
  the browser matrix. Read this before drawing any scientific conclusion.
- **[Contributing](./CONTRIBUTING.md)** — most-valuable work is quantitative
  dynamics validation and cross-vendor benchmarks. Honest negatives are
  first-class.
- **[Attribution](./NOTICE)** — every upstream dataset/model and its license.

---

## 📚 Citations

- Dorkenwald et al. 2024. *Neuronal wiring diagram of an adult brain.* Nature 634:124–138.
- Takemura et al. 2024. *A connectome of the male Drosophila ventral nerve cord.* eLife.
- Shiu et al. 2024. *A leaky integrate-and-fire model of the Drosophila brain.* Nature 632:210–217.
- Vaxenburg et al. 2025. *Whole-body simulation of realistic fruit-fly locomotion with deep reinforcement learning.* Nature.
- Marin et al. 2024. *Connectomic reconstruction of a female Drosophila ventral nerve cord.* Nature 631:128–140.

---

## 📜 License

Original code is **MIT** ([`LICENSE`](./LICENSE)). Upstream assets carry their
own licenses, consolidated in [`NOTICE`](./NOTICE): the TuragaLab flybody model
and MuJoCo are **Apache-2.0** ([`LICENSE-FLYBODY`](./LICENSE-FLYBODY)); the
FlyWire and Janelia MANC connectome data and the trained walking policy are
**CC-BY 4.0**.

<div align="center">
<br/>
<sub>MIT · Built with WebGPU, TypeScript, three.js, MuJoCo/WASM, Playwright<br/>
Author <a href="https://github.com/abgnydn">@abgnydn</a> · <a href="mailto:hi@barisgunaydin.com">hi@barisgunaydin.com</a></sub>
<br/><br/>
<a href="https://webgpu-fly.pages.dev"><img alt="Open in browser" src="https://img.shields.io/badge/%E2%96%B6%20OPEN%20IN%20BROWSER-webgpu--fly.pages.dev-9ad7ff?style=for-the-badge&labelColor=06070a"/></a>
</div>
