// main.ts — load brain.bin, run a short LIF burst, report rates per super_class.

import { loadBrain, type Brain } from "./brain";
import { FlySim, DEFAULT_PARAMS } from "./sim";

// Mirrors SUPER_CLASS_TABLE in tools/build_csr.py
const SUPER_CLASS = [
  "unknown", "sensory", "ascending", "intrinsic", "central",
  "descending", "motor", "endocrine", "visual_centrifugal",
  "visual_projection", "optic",
];

const out = document.getElementById("out") as HTMLPreElement;
function log(msg: string, cls: "ok" | "warn" | "err" | "" = "") {
  const span = document.createElement("span");
  if (cls) span.className = cls;
  span.textContent = msg + "\n";
  out.appendChild(span);
}

function topologyStats(brain: Brain) {
  const N = brain.header.numNeurons;
  const inDeg = new Uint32Array(N);
  for (let i = 0; i < N; i++) inDeg[i] = brain.rowPtr[i + 1] - brain.rowPtr[i];
  const outDeg = new Uint32Array(N);
  for (let k = 0; k < brain.colIdx.length; k++) outDeg[brain.colIdx[k]]++;
  const stat = (a: Uint32Array) => {
    let sum = 0, max = 0, nz = 0;
    for (let i = 0; i < a.length; i++) { sum += a[i]; if (a[i] > max) max = a[i]; if (a[i] > 0) nz++; }
    return { mean: sum / a.length, max, nonzero: nz };
  };
  return { in: stat(inDeg), out: stat(outDeg) };
}

function countSpikesPerClass(spikeBits: Uint32Array, superClass: Uint32Array): Map<number, number> {
  const counts = new Map<number, number>();
  for (let i = 0; i < superClass.length; i++) {
    const w = spikeBits[i >>> 5];
    if ((w >>> (i & 31)) & 1) {
      counts.set(superClass[i], (counts.get(superClass[i]) ?? 0) + 1);
    }
  }
  return counts;
}

function classSizes(superClass: Uint32Array): Map<number, number> {
  const m = new Map<number, number>();
  for (let i = 0; i < superClass.length; i++) m.set(superClass[i], (m.get(superClass[i]) ?? 0) + 1);
  return m;
}

async function main() {
  log("loading public/brain.bin ...");
  let brain: Brain;
  try {
    brain = await loadBrain("/brain.bin");
  } catch (e) {
    log(`failed: ${(e as Error).message}`, "err");
    log("did you run `npm run data && npm run convert`?", "warn");
    return;
  }

  const { header, neurons } = brain;
  log(`magic OK, version ${header.version}`, "ok");
  log(`neurons : ${header.numNeurons.toLocaleString()}`);
  log(`edges   : ${header.numEdges.toLocaleString()}`);
  log("");

  log("--- topology ---");
  const t = topologyStats(brain);
  log(`in-degree  mean=${t.in.mean.toFixed(1)}  max=${t.in.max}  with-input=${t.in.nonzero.toLocaleString()}`);
  log(`out-degree mean=${t.out.mean.toFixed(1)}  max=${t.out.max}  with-output=${t.out.nonzero.toLocaleString()}`);

  let pos = 0, neg = 0, zero = 0;
  for (let i = 0; i < header.numNeurons; i++) {
    if (neurons.sign[i] > 0) pos++;
    else if (neurons.sign[i] < 0) neg++;
    else zero++;
  }
  log(`NT signs   exc=${pos.toLocaleString()}  inh=${neg.toLocaleString()}  silent=${zero.toLocaleString()}`);

  const sizes = classSizes(neurons.superClass);
  log("super_class breakdown:");
  for (const [cls, n] of [...sizes.entries()].sort((a, b) => b[1] - a[1])) {
    log(`  ${SUPER_CLASS[cls] ?? cls.toString()}: ${n.toLocaleString()}`);
  }
  log("");

  log("--- WebGPU adapter ---");
  if (!("gpu" in navigator)) {
    log("navigator.gpu missing — open in Chrome / Edge", "err");
    return;
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) { log("no GPU adapter", "err"); return; }
  log(`maxStorageBufferBindingSize: ${adapter.limits.maxStorageBufferBindingSize.toLocaleString()}`);
  log(`maxBufferSize:               ${adapter.limits.maxBufferSize.toLocaleString()}`);
  const need = brain.colIdx.byteLength;
  log(`largest CSR buffer needs:    ${need.toLocaleString()}`);
  if (need > adapter.limits.maxStorageBufferBindingSize) {
    log("WARN: buffer too big for default limits — sim.ts requests adapter-max", "warn");
  }
  log("");

  // --- Build sim, drive it, measure rates ---
  log("--- LIF sim ---");
  const sim = await FlySim.create(brain, {
    ...DEFAULT_PARAMS,
    extGain: 5.0, // boost so Poisson input actually pushes neurons over threshold
  });
  log(`FlySim created. dt=${sim.params.dtMs} ms  tau=${sim.params.tauMs} ms`);

  // Drive sensory + optic neurons with Poisson @ ~50 Hz
  // expected spikes per step = 50 Hz * 0.001 s = 0.05; we stamp current proportional.
  const ext = new Float32Array(header.numNeurons);
  const RATE_HZ = 50;
  const dtSec = sim.params.dtMs / 1000;
  const expectedPerStep = RATE_HZ * dtSec;
  let driven = 0;
  for (let i = 0; i < header.numNeurons; i++) {
    const sc = SUPER_CLASS[neurons.superClass[i]];
    if (sc === "sensory" || sc === "optic") {
      ext[i] = expectedPerStep * 8.0; // amplitude in mV-like units; tuned heuristically
      driven++;
    }
  }
  sim.setExternalInput(ext);
  log(`driving ${driven.toLocaleString()} sensory+optic neurons at ${RATE_HZ} Hz`);

  const N_STEPS = 100;
  log(`running ${N_STEPS} steps (= ${(N_STEPS * sim.params.dtMs).toFixed(0)} ms biological time) ...`);
  const t0 = performance.now();
  sim.step(N_STEPS);
  const spikes = await sim.readSpikes(); // forces sync via mapAsync
  const elapsedMs = performance.now() - t0;

  const counts = countSpikesPerClass(spikes, neurons.superClass);
  log(`wall time: ${elapsedMs.toFixed(1)} ms  (${(elapsedMs / N_STEPS).toFixed(2)} ms/step)`, "ok");
  log(`real-time ratio: ${(N_STEPS * sim.params.dtMs / elapsedMs).toFixed(2)}× biological`);
  log("");

  let totalActive = 0;
  for (const v of counts.values()) totalActive += v;
  log(`active in last step: ${totalActive.toLocaleString()} / ${header.numNeurons.toLocaleString()} (${(100 * totalActive / header.numNeurons).toFixed(2)}%)`);
  log("active per super_class (last-step snapshot):");
  for (const [cls, n] of [...counts.entries()].sort((a, b) => b[1] - a[1])) {
    const total = sizes.get(cls) ?? 1;
    log(`  ${SUPER_CLASS[cls] ?? cls}: ${n.toLocaleString()} / ${total.toLocaleString()} (${(100 * n / total).toFixed(2)}%)`);
  }
}

main().catch((e) => log(`uncaught: ${(e as Error).stack ?? e}`, "err"));
