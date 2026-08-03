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
  fly forward from leg actuation and ground reaction — 2.019 cm per simulated
  second against a 2.0 cm/s command, with the kinematic assist off. It does
  not keep the fly upright, though: a pitch/roll damper bleeds off the body's
  pitch and roll angular velocity every substep, and with that damper off the
  fly capsizes and stops walking (0.068 cm/sim s, uprightness −0.87). The
  translation is earned; the posture is not. That path bypasses the brain and
  the spine entirely.
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

The `license: mit` in this Space's header describes the source code only. The
~314 MB of data the app downloads is not MIT, and each piece keeps its own
terms:

| | |
|---|---|
| Brain connectome | [FlyWire](https://flywire.ai) FAFB, CC-BY 4.0 |
| Ventral nerve cord | [Janelia MANC](https://www.janelia.org/project-team/flyem/manc-connectome) (Takemura et al. 2024), CC-BY 4.0 |
| Body model (MJCF + meshes) | [TuragaLab/flybody](https://github.com/TuragaLab/flybody) (Vaxenburg et al. 2025), Apache-2.0 |
| Walking policy | [Janelia Figshare deposit](https://janelia.figshare.com/articles/dataset/25309105) (Vaxenburg et al. 2025), CC-BY 4.0 |
| Physics | MuJoCo compiled to WebAssembly, Apache-2.0 |

Full attribution in [NOTICE](NOTICE); the Apache-2.0 text is in
[LICENSE-FLYBODY](LICENSE-FLYBODY) and the MIT text in [LICENSE](LICENSE).
Every approximation and shortcut is inventoried in
[LIMITATIONS.md](LIMITATIONS.md).

Source: <https://github.com/abgnydn/webgpu-fly>
