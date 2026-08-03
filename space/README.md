---
title: webgpu-fly
emoji: 🪰
colorFrom: indigo
colorTo: purple
sdk: static
app_file: index.html
header: mini
pinned: false
license: mit
short_description: A fly connectome and a physics fly body, live in WebGPU
---

# webgpu-fly

**139,255 FlyWire brain neurons and 23,188 Janelia MANC ventral-nerve-cord
neurons, simulated as leaky integrate-and-fire networks on your GPU, next to a
TuragaLab flybody fruit fly under real MuJoCo physics. No server — everything
runs in the tab.**

Two separate things move the body, and the difference matters:

- A **trained RL walking policy** (Vaxenburg et al. 2025) genuinely walks the
  fly from leg actuation and ground reaction alone. That path bypasses the
  brain and the spine entirely.
- The **connectome** drives a hand-written tripod gait. It scales that gait but
  does not generate its rhythm — the leg phase is `sin(sim_time · freq)`.

**The connectome does not walk the body.** The brain→spine link is a cell-type
name join across two different animals' connectomes, not a reconstructed
synaptic bridge. Simulated leg-motor pools in this connectome do oscillate
(5–12 Hz), but in global synchrony rather than in a tripod gait, and no global
gain setting produces tripod alternation — see `tools/vnc_rhythm.py`.

The brain LIF kernel runs at roughly 0.25 kHz of biological time on an M2 Pro,
about 4× slower than real time. NEST and Brian2 are faster and better
validated; the contribution here is reachability, not throughput.

## Running it

Requires **WebGPU** — Chrome, Edge, or Safari 26+. Firefox will not work yet.

About **314 MB** of connectome and body data downloads on first visit and is
then cached in IndexedDB, so later visits are fast.

For the best experience open the app on its own origin rather than inside the
Space frame — an embedded page gets a partitioned (or blocked) storage bucket,
so the 314 MB cache may not survive between visits.

## Credits and licensing

Code is MIT. The data is not, and each piece keeps its own terms:

| | |
|---|---|
| Brain connectome | [FlyWire](https://flywire.ai) FAFB, CC-BY |
| Ventral nerve cord | [Janelia MANC](https://www.janelia.org/project-team/flyem/manc-connectome) (Takemura et al. 2024) |
| Body model + walking policy | [TuragaLab/flybody](https://github.com/TuragaLab/flybody) (Vaxenburg et al. 2025), Apache-2.0 |
| Physics | MuJoCo compiled to WebAssembly |

Full attribution in `NOTICE`; every approximation and shortcut is inventoried
in `LIMITATIONS.md`.

Source: <https://github.com/abgnydn/webgpu-fly>
