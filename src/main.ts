// main.ts — load brain.bin, run an LIF burst, capture snapshots, feed viewer.

import { loadBrain, type Brain } from "./brain";
import { FlySim, DEFAULT_PARAMS } from "./sim";
import { FlyViewer } from "./viewer";

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

function classSizes(superClass: Uint32Array) {
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

  let pos = 0, neg = 0, zero = 0;
  for (let i = 0; i < header.numNeurons; i++) {
    if (neurons.sign[i] > 0) pos++;
    else if (neurons.sign[i] < 0) neg++;
    else zero++;
  }
  log(`NT signs   exc=${pos.toLocaleString()}  inh=${neg.toLocaleString()}  silent=${zero.toLocaleString()}`);
  log("");

  // --- Boot viewer with the static neuron cloud ---
  const container = document.getElementById("canvas-container") as HTMLDivElement;
  const viewer = new FlyViewer(brain, { container, pointSize: 1200 });
  log("viewer ready — drag to rotate, wheel to zoom", "ok");
  log("");

  // --- WebGPU + sim ---
  if (!("gpu" in navigator)) {
    log("navigator.gpu missing — open in Chrome / Edge", "err");
    return;
  }

  const sim = await FlySim.create(brain, {
    ...DEFAULT_PARAMS,
    extGain: 5.0,
  });
  log(`FlySim ready. dt=${sim.params.dtMs} ms  tau=${sim.params.tauMs} ms`);

  // Drive sensory + optic with Poisson @ 50 Hz
  const ext = new Float32Array(header.numNeurons);
  let driven = 0;
  for (let i = 0; i < header.numNeurons; i++) {
    const sc = SUPER_CLASS[neurons.superClass[i]];
    if (sc === "sensory" || sc === "optic") {
      ext[i] = 0.4; // tuned heuristically; rate * gain * mV
      driven++;
    }
  }
  sim.setExternalInput(ext);
  log(`driving ${driven.toLocaleString()} sensory+optic neurons`);
  log("");

  // --- Capture snapshots ---
  const N_SNAPSHOTS = 60;        // 60 frames × 10 ms = 600 ms biological time
  const STEPS_PER_SNAPSHOT = 10; // 10 ms biological per snapshot at dt=1 ms

  log(`capturing ${N_SNAPSHOTS} snapshots × ${STEPS_PER_SNAPSHOT} steps each`);
  log(`(= ${N_SNAPSHOTS * STEPS_PER_SNAPSHOT} ms biological time, ${(N_SNAPSHOTS * STEPS_PER_SNAPSHOT * sim.params.dtMs).toFixed(0)} ms wall-clock target)`);

  const sizes = classSizes(neurons.superClass);
  const t0 = performance.now();
  for (let s = 0; s < N_SNAPSHOTS; s++) {
    const rate = await sim.captureRollingRate(STEPS_PER_SNAPSHOT);
    viewer.pushSnapshot(rate);

    // Quick stats every 10 snapshots
    if (s % 10 === 0 || s === N_SNAPSHOTS - 1) {
      const counts = new Map<number, number>();
      let totalActive = 0;
      for (let i = 0; i < rate.length; i++) {
        if (rate[i] > 0) {
          counts.set(neurons.superClass[i], (counts.get(neurons.superClass[i]) ?? 0) + 1);
          totalActive++;
        }
      }
      log(`snap ${s.toString().padStart(2)}/${N_SNAPSHOTS}: ${totalActive.toLocaleString()} active (${(100 * totalActive / header.numNeurons).toFixed(1)}%)`);
    }
  }
  const elapsed = performance.now() - t0;
  log("", );
  log(`done in ${elapsed.toFixed(0)} ms wall (${(elapsed / (N_SNAPSHOTS * STEPS_PER_SNAPSHOT)).toFixed(2)} ms/step)`, "ok");
  log(`real-time ratio: ${(N_SNAPSHOTS * STEPS_PER_SNAPSHOT * sim.params.dtMs / elapsed).toFixed(2)}× biological`);
  log("");
  log("use scrub bar to step through time, ▶ to autoplay", "ok");

  // --- Wire up controls ---
  const controls = document.getElementById("controls") as HTMLDivElement;
  const scrub = document.getElementById("scrub") as HTMLInputElement;
  const playBtn = document.getElementById("play") as HTMLButtonElement;
  const label = document.getElementById("frame-label") as HTMLSpanElement;

  controls.hidden = false;
  scrub.max = String(N_SNAPSHOTS - 1);
  scrub.value = "0";
  label.textContent = `snap 0 / ${N_SNAPSHOTS}`;

  scrub.addEventListener("input", () => {
    const idx = Number(scrub.value);
    viewer.setAutoplay(false);
    playBtn.textContent = "▶";
    viewer.applySnapshot(idx);
    const ms = idx * STEPS_PER_SNAPSHOT * sim.params.dtMs;
    label.textContent = `snap ${idx} / ${N_SNAPSHOTS}  (t=${ms.toFixed(0)} ms)`;
  });

  let playing = false;
  playBtn.addEventListener("click", () => {
    playing = !playing;
    viewer.setAutoplay(playing);
    playBtn.textContent = playing ? "⏸" : "▶";
  });

  // sync scrub bar to viewer's auto-advancing index
  setInterval(() => {
    if (playing) {
      scrub.value = String(viewer.current);
      const ms = viewer.current * STEPS_PER_SNAPSHOT * sim.params.dtMs;
      label.textContent = `snap ${viewer.current} / ${N_SNAPSHOTS}  (t=${ms.toFixed(0)} ms)`;
    }
  }, 50);
}

main().catch((e) => log(`uncaught: ${(e as Error).stack ?? e}`, "err"));
