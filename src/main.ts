// main.ts — load brain, expose stimulus presets, run on demand into the viewer.

import { loadBrain, type Brain } from "./brain";
import { FlySim, DEFAULT_PARAMS } from "./sim";
import { FlyViewer } from "./viewer";
import { Room } from "./room";
import { Physics } from "./physics";

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

  // --- Embodiment: 3D room with procedural fly under DN drive ---
  const roomContainer = document.getElementById("room-container") as HTMLDivElement;
  const room = new Room({ container: roomContainer });
  const driveReadout = document.getElementById("drive-readout") as HTMLDivElement;

  // Load real flybody MJCF in the background — fetches fruitfly.xml +
  // 85 OBJ meshes, compiles via VFS, then asks the room to build its
  // body graph. Don't block the brain init on it.
  Physics.create((msg) => log(`flybody: ${msg}`))
    .then(async (p) => {
      await room.attachPhysics(p);
      log(`flybody attached (${p.bodyCount} bodies)`, "ok");
    })
    .catch((e) => log(`flybody failed to load: ${(e as Error).message}`, "warn"));

  // Pre-bin DN indices by hemisphere using pos_x relative to brain centroid.
  // Left hemisphere = pos_x < cx (anatomical left when looking at the brain
  // from in front). Cache once.
  let cx = 0, np = 0;
  for (let i = 0; i < header.numNeurons; i++) {
    const x = neurons.pos[3 * i];
    if (x !== 0) { cx += x; np++; }
  }
  cx = np > 0 ? cx / np : 0;
  const dnLeft: number[] = [];
  const dnRight: number[] = [];
  for (let i = 0; i < header.numNeurons; i++) {
    if ((neurons.cellType[i] & 0xff) !== HERO.dn) continue;
    if (neurons.pos[3 * i] < cx) dnLeft.push(i);
    else dnRight.push(i);
  }
  log(`DN drive : ${dnLeft.length} left + ${dnRight.length} right`);

  // Central-brain index cache for eye-glow modulation (super_class 4).
  const centralIdxs: number[] = [];
  for (let i = 0; i < header.numNeurons; i++) {
    if (neurons.superClass[i] === 4) centralIdxs.push(i);
  }

  // Hero-group index buckets for the validation table.
  const heroBuckets = new Map<number, number[]>();
  for (let i = 0; i < header.numNeurons; i++) {
    const h = neurons.cellType[i] & 0xff;
    if (h === 0) continue;
    if (!heroBuckets.has(h)) heroBuckets.set(h, []);
    heroBuckets.get(h)!.push(i);
  }
  // Honest annotations — what the FlyWire / mushroom-body literature
  // expects under broad sensory drive. KC sparsity is the load-bearing
  // canonical result (Honegger 2011, Lin 2014); the rest are
  // coarse-grained intuitions, not specific paper numbers.
  const HERO_LABELS: Record<number, { name: string; expect: string }> = {
    1: { name: "KC",   expect: "5-10% (canonical sparsity)" },
    2: { name: "MBON", expect: "variable" },
    3: { name: "LHN",  expect: "variable" },
    4: { name: "PN",   expect: "50-80% under broad ORN drive" },
    5: { name: "ORN",  expect: "depends on driven set" },
    6: { name: "GF",   expect: "rare; escape-only" },
    7: { name: "DN",   expect: "30-80% under sensory drive" },
  };

  /** Peak active count over a snapshot stack, restricted to a given index list. */
  function peakActive(snapshots: Float32Array[], idxs: number[]): number {
    let peak = 0;
    for (const snap of snapshots) {
      let n = 0;
      for (const i of idxs) if (snap[i] > 0) n++;
      if (n > peak) peak = n;
    }
    return peak;
  }
  function logHeroValidation(snapshots: Float32Array[]) {
    log("hero peak active / total (literature expectation):");
    for (const heroId of [1, 2, 3, 4, 5, 7] as const) {
      const idxs = heroBuckets.get(heroId);
      if (!idxs || idxs.length === 0) continue;
      const peak = peakActive(snapshots, idxs);
      const tot = idxs.length;
      const pct = 100 * peak / tot;
      const lbl = HERO_LABELS[heroId];
      const cls = heroId === 1
        ? (pct <= 12 ? "ok" : "warn")  // KC sparsity is the hard-floor check
        : "";
      log(`  ${lbl.name.padEnd(4)} ${peak.toString().padStart(5)} / ${tot.toString().padEnd(5)} (${pct.toFixed(1)}%)  — ${lbl.expect}`, cls);
    }
  }

  let driveFwd = 0, driveTurn = 0; // smoothed
  function applyDriveFromSnapshot(rate: Float32Array) {
    let sumL = 0;
    for (const i of dnLeft) sumL += rate[i];
    let sumR = 0;
    for (const i of dnRight) sumR += rate[i];
    let sumC = 0;
    for (const i of centralIdxs) sumC += rate[i];
    const meanL = dnLeft.length ? sumL / dnLeft.length : 0;
    const meanR = dnRight.length ? sumR / dnRight.length : 0;
    const meanC = centralIdxs.length ? sumC / centralIdxs.length : 0;
    room.setEyeGlow(Math.min(1, meanC * 25));

    // Rates are in [0, 1] per step (1 = spike every step). Boost to a
    // visible commanded velocity, clamp to ±1.
    // Forward gets the boost; turn is normalised by total activity so
    // small inherent L/R count imbalances (601 vs 716 DNs in this
    // dataset) don't get amplified into a permanent bias. Deadband at
    // |asym| < 0.08 zeroes near-symmetric drive — the fly walks
    // straight unless the DN imbalance is genuinely asymmetric.
    const gain = 30.0;
    const total = meanL + meanR + 1e-6;
    const asym = (meanR - meanL) / total;
    const asymTrim = Math.abs(asym) < 0.08 ? 0 : asym - Math.sign(asym) * 0.08;
    const targetFwd  = Math.max(-1, Math.min(1, gain * (meanL + meanR) * 0.5));
    const targetTurn = Math.max(-1, Math.min(1, asymTrim * 0.8));

    // Smoothing — exponential blend so gait feels less jittery.
    const alpha = 0.35;
    driveFwd  = driveFwd  + alpha * (targetFwd  - driveFwd);
    driveTurn = driveTurn + alpha * (targetTurn - driveTurn);

    room.setDrive(driveFwd, driveTurn);
    driveReadout.textContent = `fwd ${driveFwd.toFixed(2)}  turn ${driveTurn.toFixed(2)}  L=${meanL.toFixed(3)} R=${meanR.toFixed(3)}`;
  }
  function decayDrive() {
    driveFwd  *= 0.85;
    driveTurn *= 0.85;
    room.setDrive(driveFwd, driveTurn);
    driveReadout.textContent = `fwd ${driveFwd.toFixed(2)}  turn ${driveTurn.toFixed(2)}`;
  }

  if (!("gpu" in navigator)) {
    log("navigator.gpu missing — open in Chrome / Edge", "err");
    return;
  }
  const sim = await FlySim.create(brain, { ...DEFAULT_PARAMS });
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

  // --- Wire scrub + play + record ---
  const controls = document.getElementById("controls") as HTMLDivElement;
  const scrub = document.getElementById("scrub") as HTMLInputElement;
  const playBtn = document.getElementById("play") as HTMLButtonElement;
  const recBtn = document.getElementById("rec") as HTMLButtonElement;
  const label = document.getElementById("frame-label") as HTMLSpanElement;
  let playing = false;
  let stopRec: (() => Promise<Blob>) | null = null;
  recBtn.addEventListener("click", async () => {
    if (!stopRec) {
      stopRec = viewer.startRecording(30);
      recBtn.classList.add("recording");
      recBtn.textContent = "■ stop";
      log("recording started", "ok");
    } else {
      const blob = await stopRec();
      stopRec = null;
      recBtn.classList.remove("recording");
      recBtn.textContent = "● rec";
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `webgpu-fly-${Date.now()}.webm`;
      a.click();
      URL.revokeObjectURL(url);
      log(`saved ${(blob.size / 1e6).toFixed(1)} MB clip`, "ok");
    }
  });

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
    // Append body speed to the embodiment readout each tick.
    const sp = room.bodySpeed();
    const cur = driveReadout.textContent ?? "";
    const tag = `  speed ${sp.toFixed(2)} cm/s`;
    driveReadout.textContent = cur.replace(/\s+speed[^\s]*\s*cm\/s$/, "") + tag;
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
    room.resetFly();

    const t0 = performance.now();
    for (let s = 0; s < N_SNAPSHOTS; s++) {
      const rate = await sim.captureRollingRate(STEPS_PER_SNAPSHOT);
      viewer.pushSnapshot(rate);
      applyDriveFromSnapshot(rate);
    }
    const elapsed = performance.now() - t0;
    const totalSteps = N_SNAPSHOTS * STEPS_PER_SNAPSHOT;
    log(`${totalSteps} steps in ${elapsed.toFixed(0)} ms wall (${(elapsed / totalSteps).toFixed(2)} ms/step)`, "ok");
    // Drive persists at the stim's end-of-window value so the user
    // can watch the body keep walking after the brain sim completes.
    // Click another stim (or Spontaneous) to change it.

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
    const snaps = [...Array(viewer.numSnapshots)].map((_, i) => viewer["snapshots"][i] as Float32Array);
    logHeroValidation(snaps);

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

  // --- Click-to-stim: pulse a single neuron, watch the cascade ---
  async function runSingleNeuronStim(idx: number) {
    if (busy) return;
    busy = true;
    buttons.forEach((b) => { b.disabled = true; b.classList.remove("active"); });
    controls.hidden = true;

    const sc = SUPER_CLASS[neurons.superClass[idx]] ?? "?";
    const hero = neurons.cellType[idx] & 0xff;
    const heroName = ["", "KC", "MBON", "LHN", "PN", "ORN", "GF", "DN"][hero] ?? "";
    const tag = heroName ? `${heroName} (${sc})` : sc;
    log("");
    log(`--- single-neuron stim: idx ${idx}  [${tag}] ---`, "ok");

    const ext = new Float32Array(header.numNeurons);
    ext[idx] = 2.0; // strong pulse on this one cell
    sim.reset();
    sim.setExternalInput(ext);
    viewer.clearSnapshots();
    viewer.highlightNeuron(idx);
    room.resetFly();

    const t0 = performance.now();
    for (let s = 0; s < N_SNAPSHOTS; s++) {
      const rate = await sim.captureRollingRate(STEPS_PER_SNAPSHOT);
      viewer.pushSnapshot(rate);
      applyDriveFromSnapshot(rate);
    }
    const elapsed = performance.now() - t0;
    log(`${N_SNAPSHOTS * STEPS_PER_SNAPSHOT} steps in ${elapsed.toFixed(0)} ms`, "ok");

    let recruited = 0;
    const last = viewer["snapshots"][viewer.numSnapshots - 1] as Float32Array;
    for (let i = 0; i < last.length; i++) if (last[i] > 0) recruited++;
    log(`final-window recruits: ${recruited.toLocaleString()} / ${header.numNeurons.toLocaleString()}`);
    if (recruited > 100) {
      const snaps = [...Array(viewer.numSnapshots)].map((_, j) => viewer["snapshots"][j] as Float32Array);
      logHeroValidation(snaps);
    }

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
  viewer.onPick((idx) => { void runSingleNeuronStim(idx); });

  // Auto-run the first preset (visual flash) so there's something on screen.
  runStimulus(STIMULI[0], buttons[0]);
}

main().catch((e) => log(`uncaught: ${(e as Error).stack ?? e}`, "err"));
