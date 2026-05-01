# webgpu-fly — Claude context

## Goal

Realtime LIF (leaky integrate-and-fire) simulation of the FlyWire FAFB
*Drosophila* whole-brain connectome on WebGPU. ~140k neurons, ~5M aggregated
edges. One fused dispatch per timestep, target ≥1 kHz biological-time on
M2 Pro 16 GB.

Companion to `~/Downloads/webgpu-dna`. Same thesis (Geant4-class simulator
ported to WebGPU via kernel fusion), different physics. Crucially the fusion
shape is different — see README.

## Architecture

- **Data pipeline** (`tools/build_csr.py`): FlyWire connectivity feather +
  annotations TSV → `public/brain.bin` (binary CSR, ~45 MB). Pre-signs
  weights using presynaptic neurotransmitter so kernel never branches on
  E/I at runtime.
- **Kernel** (`src/shaders/lif.wgsl`): one fused dispatch per timestep.
  Per neuron: gather presynaptic spikes via CSR row, integrate Vm with
  leak, threshold + reset, write spike bit to output buffer.
- **Snapshot exporter** (planned): every N ms, copy `vm` or `spike_count`
  buffer to disk. Consumed by the WGDNA-4D viewer with a neuron-mesh
  renderer instead of the voxel grid renderer.
- **Embodiment** (later): map descending-neuron (DN) activity to
  TuragaLab/flybody MuJoCo model.

## Binary format

`brain.bin` (little-endian):

```
Header 64 B:
  magic "WGFLYBRN" (8 B), version u32, num_neurons u32,
  num_edges u32, flags u32, voxel_to_nm vec3<f32>, reserved

Neurons N×32 B:
  pos_x f32 (nm), pos_y f32 (nm), pos_z f32 (nm),
  sign f32 (-1/0/+1), cell_type u32, super_class u32,
  flags u32, conf f32 (NT prediction confidence)

CSR row_ptr (N+1)×u32:
  row_ptr[i] = offset into col_idx where neuron i's INCOMING edges start

CSR col_idx E×u32:
  presynaptic neuron index for each incoming edge

CSR weight E×f32:
  pre-signed: sign(pre_nt) × synapse_count
```

E ≈ 5M (aggregated proofread pairs). Total bin ≈ 45 MB.

## Neurotransmitter → sign mapping

| top_nt | sign | rationale |
|---|---|---|
| acetylcholine | +1 | excitatory (nAChR) |
| GABA | −1 | inhibitory (GABA-A/B) |
| glutamate | −1 | mostly inhibitory in fly via GluCl |
| dopamine, serotonin, octopamine | 0 | modulatory; out of scope for v1 |
| (low confidence < 0.5) | 0 | leave silent |

See `tools/build_csr.py`.

## Coordinate system

FlyWire `pos_x/y/z` and `soma_x/y/z` are in voxel units. FAFB14 voxel size
is 4 × 4 × 40 nm. The loader converts to nm before writing `brain.bin`.
Use `soma_*` if non-null, fall back to `pos_*` (synapse-cloud centroid).

## Validation targets

- **Topology sanity**: in/out-degree distributions match published FlyWire
  stats; max in-degree neuron is the expected one (typically a Kenyon cell
  or a giant interneuron).
- **Dynamics sanity**: with no input, network goes silent within ~50 ms
  (no runaway). With Poisson input to ORNs, downstream Kenyon cells fire
  sparsely (~1–5% population), antennal lobe PNs show characteristic
  rates.
- **Quantitative**: cross-check firing rates against published FlyWire
  LIF simulations (Shiu et al. 2024 or Lappalainen et al. visual-system
  work).

## Known design decisions

- **Aggregated pairs, not raw synapses.** `proofread_connections_783.feather`
  pre-aggregates by (pre, post) pair. We use that directly — gives ~5M edges
  vs ~54M raw synapses. v1 LIF doesn't care about per-synapse spatial
  position; if we add dendritic compartments later, switch to the 9.5 GB
  raw table.
- **Pre-signed weights.** Saves a per-edge lookup in the inner loop.
  Tradeoff: can't easily change NT confidence threshold without rebuilding
  the bin. Acceptable.
- **Incoming-edge CSR (not outgoing).** The gather is "for each post-neuron,
  sum over presynaptic spikes" — that maps to a row read of incoming edges.
  Outgoing CSR would require a scatter (atomicAdd contention on hot
  postsynaptic targets).

## Commands

```bash
bash tools/download_data.sh   # ~855 MB Zenodo pull
python3 tools/build_csr.py    # → public/brain.bin
npm run dev                   # localhost:8766
npm run test
npm run typecheck
```

## Cross-refs

- `~/Downloads/webgpu-dna/CLAUDE.md` — sister project; the kernel-fusion
  pattern that motivated this.
- `~/Documents/github/webgpu-fly/tools/build_csr.py` — authoritative
  binary format spec lives in this file's docstring.
