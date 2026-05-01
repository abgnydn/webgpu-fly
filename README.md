# webgpu-fly

Realtime LIF simulation of the FlyWire whole-brain *Drosophila* connectome on
WebGPU. ~140k neurons, ~5M aggregated synaptic connections, fused per-timestep
kernel, snapshots compatible with the WGDNA-4D viewer pipeline.

Companion to [webgpu-dna](https://github.com/abgunaydin/webgpu-dna) — same
"port a Geant4-class simulator to WebGPU via kernel fusion" thesis, applied to
neural dynamics instead of radiation track structure.

## Status

Bootstrapping. Data pipeline first, then LIF kernel, then 4D viewer hookup,
then embodiment via [TuragaLab/flybody](https://github.com/TuragaLab/flybody).

## Data sources (v1)

| Source | What | Size |
|---|---|---|
| [Zenodo 10676866](https://zenodo.org/records/10676866) — `proofread_connections_783.feather` | Aggregated pre→post synapse counts | 852 MB |
| [Zenodo 10676866](https://zenodo.org/records/10676866) — `proofread_root_ids_783.npy` | Verified neuron ID list | 1.1 MB |
| [flyconnectome/flywire_annotations](https://github.com/flyconnectome/flywire_annotations) — `Supplemental_file1_neuron_annotations.tsv` | Soma position + neurotransmitter + cell type per neuron | ~30 MB |

**Skipped for v1** (added later if needed): full per-synapse table (9.5 GB),
NBLAST scores, downsampled skeleton meshes (5.4 GB).

## Quickstart

```bash
# 1. Pull raw data (~855 MB, mostly the connections feather)
bash tools/download_data.sh

# 2. Build the binary CSR brain blob (→ public/brain.bin, ~45 MB)
python3 tools/build_csr.py

# 3. Run the simulator
npm install
npm run dev   # http://localhost:8766
```

## Binary brain format

See `tools/build_csr.py` docstring. Layout:

```
[ Header 64 B ] [ Neurons N×32 B ] [ row_ptr (N+1)×4 B ] [ col_idx E×4 B ] [ weight E×4 B ]
```

CSR rows are **incoming** edges (post→pre) so the per-timestep gather is a
single pass over each neuron's row. Weights are pre-signed at build time:
`weight = sign(presynaptic_NT) × synapse_count`. The kernel never has to look
up neurotransmitter signs at runtime.

## Why kernel fusion ≠ Geant4-DNA's fusion

Geant4-DNA fuses per-primary (one thread, full history in a `for` loop)
because primaries are independent. Fly brain neurons are **not** independent
— each timestep's state depends on the current state of presynaptic
neighbors. So fusion here means **one dispatch per timestep** (gather +
integrate + spike + reset, all inline), not one dispatch per simulation. Still
a real win vs Brian2/NEST per-equation dispatching.

## License

MIT. FlyWire data: CC-BY-4.0 (Dorkenwald et al. 2024). Annotations: see
[flyconnectome/flywire_annotations](https://github.com/flyconnectome/flywire_annotations).
