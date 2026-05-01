// main.ts — load brain, expose stimulus presets, run on demand into the viewer.

import { loadBrain, type Brain } from "./brain";
import { FlySim, DEFAULT_PARAMS } from "./sim";
import { FlyViewer } from "./viewer";

const SUPER_CLASS = [
  "unknown", "sensory", "ascending", "intrinsic", "central",
  "descending", "motor", "endocrine", "visual_centrifugal",
  "visual_projection", "optic",
];

// HERO_CELL_TYPES from tools/build_csr.py — packed in lower 8 bits of cell_type.
const HERO = { kenyon: 1, mbon: 2, lhn: 3, pn: 4, orn: 5, gf: 6, dn: 7 };

interface Stimulus {
  id: string;
  label: string;
  hint: string;
  /** Returns a Float32Array of per-neuron external input (mV-like). */
  build(brain: Brain): { ext: Float32Array; driven: number };
}

function buildExt(
  brain: Brain,
  predicate: (i: number) => boolean,
  amplitude: number,
): { ext: Float32Array; driven: number } {
  const N = brain.header.numNeurons;
  const ext = new Float32Array(N);
  let driven = 0;
  for (let i = 0; i < N; i++) {
    if (predicate(i)) { ext[i] = amplitude; driven++; }
  }
  return { ext, driven };
}

const STIMULI: Stimulus[] = [
  {
    id: "visual",
    label: "Visual flash",
    hint: "drive optic + visual_projection",
    build: (brain) => buildExt(brain, (i) => {
      const sc = SUPER_CLASS[brain.neurons.superClass[i]];
      return sc === "optic" || sc === "visual_projection";
    }, 0.5),
  },
  {
    id: "olfactory",
    label: "Olfactory hit",
    hint: "drive ORNs (smell)",
    build: (brain) => buildExt(brain, (i) => {
      // ORN lives in lower 8 bits of cell_type per build_csr.py
      return (brain.neurons.cellType[i] & 0xff) === HERO.orn;
    }, 0.8),
  },
  {
    id: "mixed",
    label: "Mixed sensory",
    hint: "all sensory + optic",
    build: (brain) => buildExt(brain, (i) => {
      const sc = SUPER_CLASS[brain.neurons.superClass[i]];
      return sc === "sensory" || sc === "optic";
    }, 0.4),
  },
  {
    id: "spontaneous",
    label: "Spontaneous",
    hint: "no input — baseline drift",
    build: (brain) => ({ ext: new Float32Array(brain.header.numNeurons), driven: 0 }),
  },
];

const N_SNAPSHOTS = 40;
const STEPS_PER_SNAPSHOT = 10;

const out = document.getElementById("out") as HTMLPreElement;
function log(msg: string, cls: "ok" | "warn" | "err" | "" = "") {
  const span = document.createElement("span");
  if (cls) span.className = cls;
  span.textContent = msg + "\n";
  out.appendChild(span);
  out.scrollTop = out.scrollHeight;
}

function classSizes(superClass: Uint32Array) {
  const m = new Map<number, number>();
  for (let i = 0; i < superClass.length; i++) m.set(superClass[i], (m.get(superClass[i]) ?? 0) + 1);
  return m;
}

async function main() {
  log("loading brain.bin ...");
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

  const sizes = classSizes(neurons.superClass);
  let pos = 0, neg = 0;
  for (let i = 0; i < header.numNeurons; i++) {
    if (neurons.sign[i] > 0) pos++;
    else if (neurons.sign[i] < 0) neg++;
  }
  log(`exc=${pos.toLocaleString()}  inh=${neg.toLocaleString()}`);
  log("");

  const container = document.getElementById("canvas-container") as HTMLDivElement;
  const viewer = new FlyViewer(brain, { container, pointSize: 1200 });
  log("viewer ready — drag to rotate, wheel to zoom", "ok");

  if (!("gpu" in navigator)) {
    log("navigator.gpu missing — open in Chrome / Edge", "err");
    return;
  }
  const sim = await FlySim.create(brain, { ...DEFAULT_PARAMS, extGain: 5.0 });
  log(`FlySim ready. dt=${sim.params.dtMs} ms  tau=${sim.params.tauMs} ms`);
  log("");

  // --- Build stimulus buttons ---
  const stimRow = document.getElementById("stim-row") as HTMLDivElement;
  const buttons: HTMLButtonElement[] = [];
  for (const stim of STIMULI) {
    const btn = document.createElement("button");
    btn.className = "stim-btn";
    btn.innerHTML = `<span class="label">${stim.label}</span><span class="hint">${stim.hint}</span>`;
    btn.addEventListener("click", () => runStimulus(stim, btn));
    stimRow.appendChild(btn);
    buttons.push(btn);
  }

  // --- Wire scrub + play ---
  const controls = document.getElementById("controls") as HTMLDivElement;
  const scrub = document.getElementById("scrub") as HTMLInputElement;
  const playBtn = document.getElementById("play") as HTMLButtonElement;
  const label = document.getElementById("frame-label") as HTMLSpanElement;
  let playing = false;

  scrub.addEventListener("input", () => {
    const idx = Number(scrub.value);
    viewer.setAutoplay(false);
    playing = false;
    playBtn.textContent = "▶";
    viewer.applySnapshot(idx);
    const ms = idx * STEPS_PER_SNAPSHOT * sim.params.dtMs;
    label.textContent = `snap ${idx} / ${viewer.numSnapshots}  (t=${ms.toFixed(0)} ms)`;
  });
  playBtn.addEventListener("click", () => {
    playing = !playing;
    viewer.setAutoplay(playing);
    playBtn.textContent = playing ? "⏸" : "▶";
  });
  setInterval(() => {
    if (playing) {
      scrub.value = String(viewer.current);
      const ms = viewer.current * STEPS_PER_SNAPSHOT * sim.params.dtMs;
      label.textContent = `snap ${viewer.current} / ${viewer.numSnapshots}  (t=${ms.toFixed(0)} ms)`;
    }
  }, 50);

  let busy = false;
  async function runStimulus(stim: Stimulus, btn: HTMLButtonElement) {
    if (busy) return;
    busy = true;
    buttons.forEach((b) => { b.disabled = true; b.classList.remove("active"); });
    btn.classList.add("active");
    controls.hidden = true;

    log("");
    log(`--- ${stim.label} ---`, "ok");
    const { ext, driven } = stim.build(brain);
    log(`driving ${driven.toLocaleString()} neurons`);

    sim.reset();
    sim.setExternalInput(ext);
    viewer.clearSnapshots();

    const t0 = performance.now();
    for (let s = 0; s < N_SNAPSHOTS; s++) {
      const rate = await sim.captureRollingRate(STEPS_PER_SNAPSHOT);
      viewer.pushSnapshot(rate);
    }
    const elapsed = performance.now() - t0;
    const totalSteps = N_SNAPSHOTS * STEPS_PER_SNAPSHOT;
    log(`${totalSteps} steps in ${elapsed.toFixed(0)} ms wall (${(elapsed / totalSteps).toFixed(2)} ms/step)`, "ok");

    // Per-class peak active count
    const peak = new Map<number, number>();
    for (const snap of [...Array(viewer.numSnapshots)].map((_, i) => viewer["snapshots"][i] as Float32Array)) {
      const live = new Map<number, number>();
      for (let i = 0; i < snap.length; i++) {
        if (snap[i] > 0) live.set(neurons.superClass[i], (live.get(neurons.superClass[i]) ?? 0) + 1);
      }
      for (const [k, v] of live) {
        if (v > (peak.get(k) ?? 0)) peak.set(k, v);
      }
    }
    log("peak active / total per super_class:");
    for (const [cls, n] of [...peak.entries()].sort((a, b) => b[1] - a[1]).slice(0, 6)) {
      const total = sizes.get(cls) ?? 1;
      log(`  ${SUPER_CLASS[cls] ?? cls}: ${n.toLocaleString()} / ${total.toLocaleString()} (${(100 * n / total).toFixed(1)}%)`);
    }

    // Wire scrub bar to new snapshot count, autoplay
    scrub.max = String(viewer.numSnapshots - 1);
    scrub.value = "0";
    label.textContent = `snap 0 / ${viewer.numSnapshots}  (t=0 ms)`;
    controls.hidden = false;
    playing = true;
    viewer.setAutoplay(true);
    playBtn.textContent = "⏸";

    buttons.forEach((b) => { b.disabled = false; });
    busy = false;
  }

  // Auto-run the first preset (visual flash) so there's something on screen.
  runStimulus(STIMULI[0], buttons[0]);
}

main().catch((e) => log(`uncaught: ${(e as Error).stack ?? e}`, "err"));
