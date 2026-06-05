// bench.ts — headless wall-clock benchmark for the brain LIF kernel.
//
// Goal: answer the headline question — does the fused dispatch hit the
// ≥1 kHz biological-time target? (CLAUDE.md: "target ≥1 kHz biological-
// time on M2 Pro 16 GB.")
//
// What it measures: GPU completion time for N LIF steps, including the
// per-step `writeBuffer(params)` call and bind-group flip — i.e. the
// real per-tick cost of the renderer's inner loop, with no rendering
// overhead.
//
// Why `onSubmittedWorkDone()`: WebGPU `submit()` returns immediately
// after queueing. Without awaiting completion we'd be timing
// command-encoding throughput, not GPU work. We submit one batch,
// await done, repeat — same as a real frame loop.
//
// Output: writes a JSON blob to `window.__benchResult` for Playwright
// to scrape, and renders a human-readable summary in #out.

import { loadBrain } from "./brain";
import { FlySim, DEFAULT_PARAMS } from "./sim";
import { loadManifest } from "./manifest";

type BenchResult = {
  numNeurons: number;
  numEdges: number;
  dtMs: number;
  warmupSteps: number;
  totalSteps: number;
  batches: number;
  stepsPerBatch: number;
  wallSeconds: number;
  bioSeconds: number;
  bioKHz: number;        // bio-ms per wall-ms = bio-kHz
  edgesPerSecond: number;
  target: number;
  passed: boolean;
};

const out = document.getElementById("out") as HTMLPreElement;
const log = (s: string, cls = "") => {
  const t = new Date().toISOString().slice(11, 19);
  out.innerHTML += `<span class="${cls}">[${t}] ${s}</span>\n`;
};
const num = (n: number, dp = 2) =>
  `<span class="num">${n.toFixed(dp)}</span>`;

async function run() {
  log("loading manifest…");
  const versionFor = await loadManifest();

  const brainUrl = (import.meta.env.VITE_BRAIN_URL || "/brain.bin")
    + versionFor("brain.bin");
  log(`loading brain from ${brainUrl} …`);
  const t0 = performance.now();
  const brain = await loadBrain(brainUrl);
  const loadMs = performance.now() - t0;
  log(`brain: ${brain.header.numNeurons.toLocaleString()} neurons · `
      + `${brain.header.numEdges.toLocaleString()} edges · `
      + `loaded in ${loadMs.toFixed(0)} ms`, "ok");

  log("creating FlySim (compiles WGSL, allocates GPU buffers)…");
  const tCreate0 = performance.now();
  const sim = await FlySim.create(brain, { ...DEFAULT_PARAMS });
  log(`FlySim ready in ${(performance.now() - tCreate0).toFixed(0)} ms`, "ok");

  // Sparse stim: drive ~1% of neurons. Keeps the kernel doing real work
  // (spike bits propagate, alpha synapses integrate) while staying in
  // the regime the renderer actually sees. With ext_input=0 the kernel
  // still walks every CSR row — the dominant cost — so this isn't
  // strictly needed to measure throughput, but it gives an honest
  // mid-load number.
  const N = brain.header.numNeurons;
  const ext = new Float32Array(N);
  const stride = 100;
  for (let i = 0; i < N; i += stride) ext[i] = 1.0;
  sim.setExternalInput(ext);

  // Force a kernel compile + first dispatch so we don't time pipeline
  // creation. WebGPU compiles lazily on first use.
  sim.reset();
  sim.step(1);
  await sim.device.queue.onSubmittedWorkDone();

  const WARMUP = 200;
  const STEPS_PER_BATCH = 200;
  const BATCHES = 10;
  const TOTAL = STEPS_PER_BATCH * BATCHES;

  log(`warmup: ${WARMUP} steps…`);
  sim.reset();
  sim.setExternalInput(ext);
  sim.step(WARMUP);
  await sim.device.queue.onSubmittedWorkDone();

  log(`measuring: ${BATCHES} × ${STEPS_PER_BATCH} steps`
      + ` (${TOTAL} steps × ${sim.params.dtMs} ms = `
      + `${(TOTAL * sim.params.dtMs).toFixed(0)} ms biological time)…`);

  const batchTimes: number[] = [];
  const tStart = performance.now();
  for (let b = 0; b < BATCHES; b++) {
    const tBatch = performance.now();
    sim.step(STEPS_PER_BATCH);
    await sim.device.queue.onSubmittedWorkDone();
    batchTimes.push(performance.now() - tBatch);
  }
  const wallMs = performance.now() - tStart;

  const bioMs = TOTAL * sim.params.dtMs;
  const bioKHz = bioMs / wallMs;
  // Edges processed: each LIF step gathers every CSR edge once.
  const edgesPerSec = (TOTAL * brain.header.numEdges) / (wallMs / 1000);

  const target = 1.0; // 1 kHz
  const passed = bioKHz >= target;

  const result: BenchResult = {
    numNeurons: brain.header.numNeurons,
    numEdges: brain.header.numEdges,
    dtMs: sim.params.dtMs,
    warmupSteps: WARMUP,
    totalSteps: TOTAL,
    batches: BATCHES,
    stepsPerBatch: STEPS_PER_BATCH,
    wallSeconds: wallMs / 1000,
    bioSeconds: bioMs / 1000,
    bioKHz,
    edgesPerSecond: edgesPerSec,
    target,
    passed,
  };

  // Per-batch jitter: useful for spotting pipeline stalls.
  const sortedBatches = [...batchTimes].sort((a, b) => a - b);
  const median = sortedBatches[Math.floor(BATCHES / 2)];
  const p95 = sortedBatches[Math.floor(BATCHES * 0.95)];

  log("", "");
  log("───── result ─────", "");
  log(`neurons   : ${num(N, 0)}`, "");
  log(`edges     : ${num(brain.header.numEdges, 0)}`, "");
  log(`steps     : ${num(TOTAL, 0)} × ${sim.params.dtMs} ms = `
      + `${num(bioMs, 0)} ms biological time`, "");
  log(`wall      : ${num(wallMs, 1)} ms`, "");
  log(`per-batch : median ${num(median, 2)} ms · `
      + `p95 ${num(p95, 2)} ms · ` + `min ${num(sortedBatches[0], 2)} ms`, "");
  log(`bio rate  : ${num(bioKHz, 3)} kHz biological-time-per-wall-second`,
      passed ? "ok" : "warn");
  log(`throughput: ${num(edgesPerSec / 1e9, 2)} G-edges/sec`, "");
  log("", "");
  log(passed
        ? `target ≥${target} kHz: PASS (${num(bioKHz / target, 2)}× headroom)`
        : `target ≥${target} kHz: FAIL (${num(bioKHz / target, 3)}× of target)`,
      passed ? "ok" : "err");

  (window as unknown as { __benchResult: BenchResult }).__benchResult = result;
  log("", "");
  log("done. window.__benchResult set.", "ok");
}

run().catch((e) => {
  log(`FAIL: ${(e as Error).message}`, "err");
  console.error(e);
  (window as unknown as { __benchError: string }).__benchError =
    (e as Error).message;
});
