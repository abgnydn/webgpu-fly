"""nest_bench.py — apples-to-apples speed comparison vs the WebGPU and
Rust LIF benches. Loads the same brain.bin, builds the same network in
NEST, runs the same warmup + measure structure, reports kHz.

This is a SPEED benchmark, not a dynamics check. The point is "how fast
does the standard scientific tool process 139,255 LIF neurons + 15M
synapses on this M2 Pro?" — not "do the spikes match exactly". So we
use iaf_psc_alpha (NEST's standard alpha-synapse LIF, same model family
as our kernel) with sensible parameters; the total work is the same
regardless of exact weight magnitudes.

Run: PYTHONPATH=/opt/homebrew/lib/python3.14/site-packages \\
     python3 tools/nest_bench.py
"""

from __future__ import annotations
import struct
import sys
import time
from pathlib import Path

import numpy as np

# Match src/sim.ts DEFAULT_PARAMS where the model supports it.
DT_MS = 1.0
TAU_MS = 20.0
V_THRESH = -45.0
V_RESET = -52.0
V_REST = -52.0
REFRACTORY_MS = 2.2

WARMUP_MS = 200.0
STEPS_PER_BATCH_MS = 200.0
BATCHES = 10


def load_brain(path: Path):
    with open(path, "rb") as f:
        buf = f.read()
    assert buf[:8] == b"WGFLYBRN"
    num_neurons = struct.unpack_from("<I", buf, 12)[0]
    num_edges = struct.unpack_from("<I", buf, 16)[0]
    header = 64
    neurons_bytes = num_neurons * 32
    row_off = header + neurons_bytes
    col_off = row_off + (num_neurons + 1) * 4
    w_off = col_off + num_edges * 4
    row_ptr = np.frombuffer(buf, dtype=np.uint32, count=num_neurons + 1, offset=row_off)
    col_idx = np.frombuffer(buf, dtype=np.uint32, count=num_edges, offset=col_off)
    weight = np.frombuffer(buf, dtype=np.float32, count=num_edges, offset=w_off)
    return num_neurons, num_edges, row_ptr.copy(), col_idx.copy(), weight.copy()


def main():
    brain_path = Path(sys.argv[1] if len(sys.argv) > 1 else "public/brain.bin")
    print(f"loading {brain_path} …")
    t = time.perf_counter()
    n, e, row_ptr, col_idx, weight = load_brain(brain_path)
    print(f"brain: {n:,} neurons · {e:,} edges · loaded in {(time.perf_counter()-t)*1000:.0f} ms")

    print("importing NEST …")
    t = time.perf_counter()
    import nest  # noqa
    print(f"NEST {nest.__version__} ready in {(time.perf_counter()-t)*1000:.0f} ms")

    nest.set_verbosity("M_WARNING")
    nest.ResetKernel()
    nest.local_num_threads = 8
    nest.resolution = DT_MS
    print(f"threads: {nest.local_num_threads} · resolution: {nest.resolution} ms")

    # Build network. iaf_psc_alpha = LIF + alpha post-synaptic current.
    # Same model family as our WGSL/Rust kernels (alpha synapse + LIF).
    print(f"creating {n:,} iaf_psc_alpha neurons …")
    t = time.perf_counter()
    pop = nest.Create(
        "iaf_psc_alpha",
        n,
        params={
            "V_th": V_THRESH,
            "V_reset": V_RESET,
            "E_L": V_REST,
            "V_m": V_REST,
            "tau_m": TAU_MS,
            "t_ref": REFRACTORY_MS,
            "C_m": 250.0,
            "tau_syn_ex": 5.0,
            "tau_syn_in": 5.0,
        },
    )
    print(f"  done in {(time.perf_counter()-t):.2f} s")

    # Connect. NEST stores presynaptic source per outgoing edge; our CSR
    # is incoming-edge so col_idx[k] is the presynaptic source for
    # post-neuron i where row_ptr[i] <= k < row_ptr[i+1]. Build flat
    # source/target arrays. NEST uses 1-based node ids.
    print("expanding incoming-CSR to source/target arrays …")
    t = time.perf_counter()
    counts = np.diff(row_ptr).astype(np.int64)
    target = np.repeat(np.arange(n, dtype=np.int64), counts) + 1
    source = col_idx.astype(np.int64) + 1
    print(f"  {len(source):,} synapses ready in {(time.perf_counter()-t):.2f} s")

    # Use a single weight magnitude — speed bench, not dynamics. Sign
    # split keeps NEST happy (excitatory vs inhibitory routes use
    # tau_syn_ex / tau_syn_in respectively).
    print("connecting …")
    t = time.perf_counter()
    weights_arr = np.full(len(source), 1.0, dtype=np.float64)
    delays_arr = np.full(len(source), 1.0, dtype=np.float64)
    nest.Connect(
        source, target,
        conn_spec="one_to_one",
        syn_spec={"weight": weights_arr, "delay": delays_arr},
    )
    print(f"  connected in {(time.perf_counter()-t):.2f} s")

    # Sparse Poisson stim on every 100th neuron — matches src/bench.ts
    # ext stim shape (1% of neurons driven).
    print("attaching Poisson stim …")
    drive = nest.Create("poisson_generator", params={"rate": 200.0})
    stim_targets = pop[::100]
    nest.Connect(drive, stim_targets, syn_spec={"weight": 200.0, "delay": 1.0})

    # Warmup forces NEST to compile / pre-allocate buffers, JIT
    # connection routing, etc. — the analogous step to GPU pipeline
    # warmup in our other benches.
    print(f"warmup: {WARMUP_MS:.0f} ms biological time …")
    t = time.perf_counter()
    nest.Simulate(WARMUP_MS)
    print(f"  warmup wall: {(time.perf_counter()-t)*1000:.1f} ms")

    print(f"measuring: {BATCHES} × {STEPS_PER_BATCH_MS:.0f} ms biological time …")
    batch_ms = []
    t_start = time.perf_counter()
    for _ in range(BATCHES):
        tb = time.perf_counter()
        nest.Simulate(STEPS_PER_BATCH_MS)
        batch_ms.append((time.perf_counter() - tb) * 1000.0)
    wall_ms = (time.perf_counter() - t_start) * 1000.0

    bio_ms = BATCHES * STEPS_PER_BATCH_MS
    bio_khz = bio_ms / wall_ms
    edges_per_sec = (bio_ms / DT_MS) * e / (wall_ms / 1000.0)

    sb = sorted(batch_ms)
    median = sb[BATCHES // 2]
    p95 = sb[int(BATCHES * 0.95)]

    print()
    print("───── result ─────")
    print(f"neurons   : {n:,}")
    print(f"edges     : {e:,}")
    print(f"steps     : {int(bio_ms / DT_MS):,} × {DT_MS} ms = {bio_ms:.0f} ms biological")
    print(f"wall      : {wall_ms:.1f} ms")
    print(f"per-batch : median {median:.2f} ms · p95 {p95:.2f} ms · min {sb[0]:.2f} ms")
    print(f"bio rate  : {bio_khz:.3f} kHz biological-time-per-wall-second")
    print(f"throughput: {edges_per_sec/1e9:.2f} G-edges/sec")
    print()
    target = 1.0
    if bio_khz >= target:
        print(f"target ≥{target} kHz: PASS ({bio_khz/target:.2f}× headroom)")
    else:
        print(f"target ≥{target} kHz: FAIL ({bio_khz/target:.3f}× of target)")


if __name__ == "__main__":
    main()
