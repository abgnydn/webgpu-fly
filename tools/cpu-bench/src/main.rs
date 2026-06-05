// cpu-bench — port of src/shaders/lif.wgsl to multicore Rust, run on the
// same brain.bin produced by tools/build_csr.py.
//
// Goal: measure honest CPU baseline so we know whether the WebGPU
// kernel is actually unlocking anything compared to a tuned native
// implementation. Mirrors the WGSL kernel exactly:
//   1. Synaptic gather over incoming CSR row → delta_in.
//   2. Two-state alpha synapse (a_syn = exp(-1/5)).
//   3. External drive × ext_gain.
//   4. Refractory countdown OR leaky-integrate Vm + threshold + reset.
//
// Same DEFAULT_PARAMS as src/sim.ts. Same 1% sparse stim as src/bench.ts.
// Same warmup/measure structure (200 warmup, 10 × 200 batches, 2000 total).

use rayon::prelude::*;
use std::fs::File;
use std::io::Read;
use std::time::Instant;

const MAGIC: &[u8] = b"WGFLYBRN";

// Matches src/sim.ts DEFAULT_PARAMS.
const DT_MS: f32 = 1.0;
const TAU_MS: f32 = 20.0;
const V_THRESH: f32 = -45.0;
const V_RESET: f32 = -52.0;
const V_REST: f32 = -52.0;
const REFRACTORY_MS: f32 = 2.2;
const EXT_GAIN: f32 = 2.0;
const W_SYN: f32 = 0.005;

// a_syn = exp(-dt/tau_syn) for dt=1ms, tau_syn=5ms. Hardcoded in lif.wgsl.
const A_SYN: f32 = 0.81873;

const WARMUP_STEPS: usize = 200;
const STEPS_PER_BATCH: usize = 200;
const BATCHES: usize = 10;

struct Brain {
    num_neurons: usize,
    num_edges: usize,
    row_ptr: Vec<u32>,
    col_idx: Vec<u32>,
    weight: Vec<f32>,
}

fn load_brain(path: &str) -> Brain {
    let mut buf = Vec::new();
    File::open(path).expect("open brain.bin").read_to_end(&mut buf).unwrap();

    assert_eq!(&buf[0..8], MAGIC, "bad magic");
    let read_u32 = |off: usize| u32::from_le_bytes(buf[off..off + 4].try_into().unwrap());

    let _version = read_u32(8);
    let num_neurons = read_u32(12) as usize;
    let num_edges = read_u32(16) as usize;
    // header is 64 B, neurons N×32 B, then CSR.
    let header = 64;
    let neurons_bytes = num_neurons * 32;

    let row_ptr_off = header + neurons_bytes;
    let row_ptr_len = num_neurons + 1;
    let col_idx_off = row_ptr_off + row_ptr_len * 4;
    let weight_off = col_idx_off + num_edges * 4;

    let mut row_ptr = vec![0u32; row_ptr_len];
    let mut col_idx = vec![0u32; num_edges];
    let mut weight = vec![0f32; num_edges];

    for i in 0..row_ptr_len {
        row_ptr[i] = read_u32(row_ptr_off + i * 4);
    }
    for i in 0..num_edges {
        col_idx[i] = read_u32(col_idx_off + i * 4);
        let bytes: [u8; 4] = buf[weight_off + i * 4..weight_off + i * 4 + 4].try_into().unwrap();
        weight[i] = f32::from_le_bytes(bytes);
    }

    Brain { num_neurons, num_edges, row_ptr, col_idx, weight }
}

#[inline(always)]
fn spike_bit(spikes: &[u32], idx: u32) -> f32 {
    let word = spikes[(idx >> 5) as usize];
    ((word >> (idx & 31)) & 1) as f32
}

fn step_lif(
    brain: &Brain,
    spikes_prev: &[u32],
    spikes_curr: &mut [u32],
    vm: &mut [f32],
    refrac: &mut [u32],
    g_x: &mut [f32],
    g_y: &mut [f32],
    ext: &[f32],
    refractory_steps: u32,
    alpha: f32,
) {
    // Per-neuron updates collected, then we OR spike bits into a per-thread
    // local word buffer and reduce. Avoids atomic contention.
    spikes_curr.iter_mut().for_each(|w| *w = 0);

    let n = brain.num_neurons;
    let chunk_size = (n + rayon::current_num_threads() * 4 - 1)
        / (rayon::current_num_threads() * 4);

    // Build (vm, refrac, gx, gy, spike_words) updates in parallel.
    let row_ptr = &brain.row_ptr;
    let col_idx = &brain.col_idx;
    let weight = &brain.weight;
    let words = spikes_curr.len();

    let updates: Vec<(usize, Vec<u32>)> = (0..n)
        .into_par_iter()
        .chunks(chunk_size)
        .map(|chunk| {
            let mut local_words = vec![0u32; words];
            for &i in &chunk {
                let row_start = row_ptr[i] as usize;
                let row_end = row_ptr[i + 1] as usize;

                let mut delta_in = 0f32;
                for k in row_start..row_end {
                    let pre = col_idx[k];
                    delta_in += weight[k] * spike_bit(spikes_prev, pre);
                }

                let gx_old = unsafe { *(g_x.as_ptr().add(i)) };
                let gy_old = unsafe { *(g_y.as_ptr().add(i)) };
                let gy_new = gy_old * A_SYN + gx_old;
                let gx_new = gx_old * A_SYN + delta_in;
                let i_syn = gy_new * W_SYN;
                let i_in = i_syn + ext[i] * EXT_GAIN;

                let r = unsafe { *(refrac.as_ptr().add(i)) };
                let v_old = unsafe { *(vm.as_ptr().add(i)) };

                let (new_vm, new_refrac, fired) = if r > 0 {
                    (V_RESET, r - 1, false)
                } else {
                    let v_new = V_REST + alpha * (v_old - V_REST) + i_in;
                    if v_new >= V_THRESH {
                        (V_RESET, refractory_steps, true)
                    } else {
                        (v_new, 0, false)
                    }
                };

                // Safe: each i is touched by exactly one thread (chunked).
                unsafe {
                    *(vm.as_ptr().add(i) as *mut f32) = new_vm;
                    *(refrac.as_ptr().add(i) as *mut u32) = new_refrac;
                    *(g_x.as_ptr().add(i) as *mut f32) = gx_new;
                    *(g_y.as_ptr().add(i) as *mut f32) = gy_new;
                }

                if fired {
                    let word_i = (i >> 5) as usize;
                    let bit = 1u32 << (i & 31);
                    local_words[word_i] |= bit;
                }
            }
            (chunk.len(), local_words)
        })
        .collect();

    for (_, local) in updates {
        for (dst, src) in spikes_curr.iter_mut().zip(local.iter()) {
            *dst |= *src;
        }
    }
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "../../public/brain.bin".to_string());
    println!("loading {} …", path);
    let t0 = Instant::now();
    let brain = load_brain(&path);
    println!(
        "brain: {} neurons · {} edges · loaded in {} ms",
        brain.num_neurons,
        brain.num_edges,
        t0.elapsed().as_millis()
    );
    println!("threads: {}", rayon::current_num_threads());

    let n = brain.num_neurons;
    let words = (n + 31) >> 5;

    let mut vm = vec![V_REST; n];
    let mut refrac = vec![0u32; n];
    let mut g_x = vec![0f32; n];
    let mut g_y = vec![0f32; n];
    let mut spikes_a = vec![0u32; words];
    let mut spikes_b = vec![0u32; words];

    // Sparse stim: every 100th neuron driven at 1.0 → matches src/bench.ts.
    let mut ext = vec![0f32; n];
    for i in (0..n).step_by(100) {
        ext[i] = 1.0;
    }

    let alpha = (-DT_MS / TAU_MS).exp();
    let refractory_steps = (REFRACTORY_MS / DT_MS).round().max(0.0) as u32;

    let mut prev_is_a = true;

    // Warmup
    println!("warmup: {} steps …", WARMUP_STEPS);
    for _ in 0..WARMUP_STEPS {
        let (prev, curr) = if prev_is_a {
            (&spikes_a, &mut spikes_b)
        } else {
            (&spikes_b, &mut spikes_a)
        };
        step_lif(
            &brain, prev, curr,
            &mut vm, &mut refrac, &mut g_x, &mut g_y, &ext,
            refractory_steps, alpha,
        );
        prev_is_a = !prev_is_a;
    }

    // Measure
    let total = STEPS_PER_BATCH * BATCHES;
    println!(
        "measuring: {} × {} steps ({} steps × {} ms = {} ms biological time) …",
        BATCHES, STEPS_PER_BATCH, total, DT_MS, total as f32 * DT_MS
    );

    let mut batch_ms = Vec::with_capacity(BATCHES);
    let t_start = Instant::now();
    for _ in 0..BATCHES {
        let t_batch = Instant::now();
        for _ in 0..STEPS_PER_BATCH {
            let (prev, curr) = if prev_is_a {
                (&spikes_a, &mut spikes_b)
            } else {
                (&spikes_b, &mut spikes_a)
            };
            step_lif(
                &brain, prev, curr,
                &mut vm, &mut refrac, &mut g_x, &mut g_y, &ext,
                refractory_steps, alpha,
            );
            prev_is_a = !prev_is_a;
        }
        batch_ms.push(t_batch.elapsed().as_secs_f64() * 1000.0);
    }
    let wall_ms = t_start.elapsed().as_secs_f64() * 1000.0;
    let bio_ms = total as f64 * DT_MS as f64;
    let bio_khz = bio_ms / wall_ms;
    let edges_per_sec = (total as f64 * brain.num_edges as f64) / (wall_ms / 1000.0);

    let mut sorted = batch_ms.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted[BATCHES / 2];
    let p95 = sorted[(BATCHES as f64 * 0.95) as usize];

    println!();
    println!("───── result ─────");
    println!("neurons   : {}", n);
    println!("edges     : {}", brain.num_edges);
    println!("steps     : {} × {} ms = {} ms biological", total, DT_MS, bio_ms);
    println!("wall      : {:.1} ms", wall_ms);
    println!("per-batch : median {:.2} ms · p95 {:.2} ms · min {:.2} ms",
        median, p95, sorted[0]);
    println!("bio rate  : {:.3} kHz biological-time-per-wall-second", bio_khz);
    println!("throughput: {:.2} G-edges/sec", edges_per_sec / 1e9);
    println!();
    let target = 1.0;
    if bio_khz >= target {
        println!("target ≥{} kHz: PASS ({:.2}× headroom)", target, bio_khz / target);
    } else {
        println!("target ≥{} kHz: FAIL ({:.3}× of target)", target, bio_khz / target);
    }
}
