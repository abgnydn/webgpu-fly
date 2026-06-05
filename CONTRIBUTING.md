# Contributing to webgpu-fly

Thanks for your interest. This is a research-grade demo project —
corrections, validation cross-checks, and honest negative results are all
welcome. Honesty about what does *not* work is a first-class contribution
here (see [`LIMITATIONS.md`](./LIMITATIONS.md)).

## Quick start

```bash
git clone https://github.com/abgnydn/webgpu-fly
cd webgpu-fly
npm install
npm run typecheck    # tsc --noEmit
npm run build        # tsc && vite build
```

`typecheck` and `build` are what CI enforces (they run on Ubuntu with no
GPU and no data). The full app and its end-to-end tests need real data and
a WebGPU GPU:

```bash
bash tools/download_data.sh   # ~855 MB FlyWire pull (Zenodo)
python3 tools/build_csr.py    # → public/brain.bin
npm run dev                   # http://localhost:8766
npm run test:e2e              # Playwright, needs WebGPU Chromium + assets
```

See the README "Quickstart" for the spine (MANC) and walking-policy pulls.

## What's most valuable

Ranked roughly by impact-per-effort:

1. **Quantitative dynamics validation.** Today the strongest dynamics claim
   is qualitative (Kenyon-cell sparsity matches Shiu et al. 2024). A
   cell-type-resolved firing-rate comparison against a published FlyWire LIF
   simulation, committed as a reproducible artifact, would be the single most
   valuable addition.
2. **Cross-browser / cross-vendor benchmarks.** Everything is measured on one
   M2 Pro + Chromium. Numbers on NVIDIA / AMD / Intel adapters and on
   Firefox / Safari WebGPU are all open.
3. **Closing the documented brain→spine→body gaps** in
   [`LIMITATIONS.md`](./LIMITATIONS.md) §4 — especially the RL-walker speed
   gap and a genuine optic→DN closed-loop reflex that doesn't fall back to
   the angle proxy.
4. **Bug reports with a reproducer** — ideally a stimulus preset or a replay
   URL that demonstrates the issue deterministically.

## Engineering standards

This repo shares the honesty discipline of its sibling projects
([`webgpu-q`](https://github.com/abgnydn/webgpu-q),
[`webgpu-dna`](https://github.com/abgnydn/webgpu-dna)). Short version:

- **Single source of truth for numbers.** A figure in the README must trace
  back to a committed benchmark or artifact, not to memory. If you change a
  measured number, change it where it's measured and let the README follow.
- **Honest negatives get committed.** A reverted optimization or a "this
  didn't work" result is documented, not deleted — it stops future-you from
  re-trying the same dead end.
- **No vanity metrics.** Describe a change by what it *does* and whether it's
  *correct against ground truth*, not by lines of code or test counts.
- **Port with attribution.** Code or model assets derived from an upstream
  (flybody, MuJoCo, PySCF-style references) keep their upstream license and
  get an entry in [`NOTICE`](./NOTICE). See the global license pattern there.

## Provenance of the heavy assets

The connectome binaries, the flybody MJCF/meshes, and the trained policy are
**not** in this repo — they are fetched from their original open-data sources
(FlyWire, Janelia MANC, TuragaLab flybody, the Vaxenburg Figshare deposit)
and rebuilt by the `tools/` scripts. See [`NOTICE`](./NOTICE) for each
source's license (CC-BY 4.0 for the connectomes and policy; Apache-2.0 for
flybody and MuJoCo). Don't commit the large binaries.

## Code of conduct

By participating you agree to the [Code of Conduct](./CODE_OF_CONDUCT.md).
Report concerns to [hi@barisgunaydin.com](mailto:hi@barisgunaydin.com).
