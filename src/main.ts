// main.ts — load brain, expose stimulus presets, run on demand into the viewer.

import { loadBrain, type Brain } from "./brain";
import { FlySim, DEFAULT_PARAMS } from "./sim";
import { FlyViewer } from "./viewer";
import { Room, RETINA_FOV_RAD } from "./room";
import { Physics } from "./physics";
import { motorFromBrain, resetVnc, vncSnapshot, type MotorContext } from "./vnc";
import { GaitEvolver } from "./evolution";

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

// Boot overlay handles. Updated as each pipeline stage progresses.
function bootStage(name: "brain" | "vnc" | "body", state: "run" | "ok", detail: string) {
  const dot = document.getElementById(`boot-${name}-dot`);
  const det = document.getElementById(`boot-${name}-detail`);
  if (dot) dot.className = `dot ${state}`;
  if (det) det.textContent = detail;
}
function bootMaybeDismiss() {
  const allOk = ["brain", "vnc", "body"].every((n) => {
    const dot = document.getElementById(`boot-${n}-dot`);
    return dot?.classList.contains("ok") || dot?.dataset.skipped === "1";
  });
  if (!allOk) return;
  const boot = document.getElementById("boot");
  if (!boot) return;
  boot.classList.add("hidden");
  setTimeout(() => boot.remove(), 700);
}
function bootSkip(name: "brain" | "vnc" | "body", detail: string) {
  const dot = document.getElementById(`boot-${name}-dot`);
  if (dot) {
    dot.dataset.skipped = "1";
    dot.style.background = "#6c7480";
  }
  const det = document.getElementById(`boot-${name}-detail`);
  if (det) det.textContent = detail;
  bootMaybeDismiss();
}

async function main() {
  // Allow the deploy to point at an external host for brain.bin /
  // brain.meta.json (Vercel Hobby's 50 MB-per-file limit blocks the
  // 120 MB connectome). Set VITE_BRAIN_URL at build time.
  const brainUrl = import.meta.env.VITE_BRAIN_URL || "/brain.bin";
  const metaUrl = import.meta.env.VITE_BRAIN_META_URL || "/brain.meta.json";
  log(`loading brain from ${brainUrl} ...`);
  bootStage("brain", "run", "fetching connectome (120 MB)…");
  let brain: Brain;
  try {
    brain = await loadBrain(brainUrl);
  } catch (e) {
    log(`failed: ${(e as Error).message}`, "err");
    log("did you run `npm run data && npm run convert`?", "warn");
    bootStage("brain", "run", `failed: ${(e as Error).message}`);
    return;
  }
  bootStage("brain", "ok", `${brain.header.numNeurons.toLocaleString()} neurons, ${brain.header.numEdges.toLocaleString()} edges`);
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

  // Load famous-DN index map from brain.meta.json. These are
  // descending neurons whose behavioural roles are documented in the
  // fly literature (DNa01 = forward, DNb01 = backward "moonwalker",
  // DNp01 = giant-fiber escape, etc.). Clicking a famous-DN button
  // single-stims those neurons and lets the connectome's wiring
  // determine the resulting behaviour — no hardcoded motor mapping.
  let famousDns: Record<string, number[]> = {};
  let famousDnLabels: Record<string, string> = {};
  try {
    const meta = await (await fetch(metaUrl)).json();
    famousDns = meta.famous_dns ?? {};
    famousDnLabels = meta.famous_dn_descriptions ?? {};
  } catch {}
  log("");

  const container = document.getElementById("canvas-container") as HTMLDivElement;
  const viewer = new FlyViewer(brain, { container, pointSize: 1200 });
  log("viewer ready — drag to rotate, wheel to zoom", "ok");

  // --- Embodiment: 3D room with procedural fly under DN drive ---
  const roomContainer = document.getElementById("room-container") as HTMLDivElement;
  const room = new Room({ container: roomContainer });
  const driveReadout = document.getElementById("drive-readout") as HTMLDivElement;

  // Retina overlay: paint the fly's forward retinal sample into the
  // top-right canvas each frame. Lets the user see what the closed-loop
  // is actually sampling. The fly's retina is small (64×16) so we copy
  // raw pixels; CSS upscales with image-rendering: pixelated for that
  // crunchy bug-vision look.
  const retinaCanvas = document.getElementById("retina-overlay") as HTMLCanvasElement;
  const retinaCtx = retinaCanvas.getContext("2d")!;
  const retinaImg = retinaCtx.createImageData(64, 16);
  // Repaint via the room's onRetinaUpdate callback — the room renders
  // the retina each frame in lockstep with body physics, so every paint
  // here matches the freshest pixels the closed-loop will read.
  room.onRetinaUpdate = () => {
    const frame = room.retinaFrame();
    const src = frame.pixels;
    const dst = retinaImg.data;
    // RGBA8 from WebGL has Y flipped relative to ImageData; mirror so
    // the overlay reads "as the fly sees" (sky up, floor down).
    for (let y = 0; y < frame.h; y++) {
      const srcRow = (frame.h - 1 - y) * frame.w * 4;
      const dstRow = y * frame.w * 4;
      for (let i = 0; i < frame.w * 4; i++) dst[dstRow + i] = src[srcRow + i];
    }
    retinaCtx.putImageData(retinaImg, 0, 0);
  };

  // Load real flybody MJCF in the background — fetches fruitfly.xml +
  // 85 OBJ meshes, compiles via VFS, then asks the room to build its
  // body graph. Don't block the brain init on it.
  let physics: Physics | null = null;
  bootStage("body", "run", "fetching flybody MJCF + 85 meshes…");
  Physics.create((msg) => {
    log(`flybody: ${msg}`);
    bootStage("body", "run", msg);
  })
    .then(async (p) => {
      physics = p;
      await room.attachPhysics(p);
      log(`flybody attached (${p.bodyCount} bodies)`, "ok");
      bootStage("body", "ok", `${p.bodyCount} bodies, 85 meshes ready`);
      bootMaybeDismiss();
    })
    .catch((e) => {
      log(`flybody failed to load: ${(e as Error).message}`, "warn");
      bootStage("body", "run", `failed: ${(e as Error).message}`);
    });

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

  // Optic-lobe index buckets split by hemisphere — used for closed-
  // loop visual drive. Target on fly's right → drive right-optic
  // neurons; target on left → drive left-optic. The connectome's
  // optic→central→DN wiring then decides the motor response.
  const opticLeft: number[] = [];
  const opticRight: number[] = [];
  for (let i = 0; i < header.numNeurons; i++) {
    if (neurons.superClass[i] !== 10) continue;
    if (neurons.pos[3 * i] < cx) opticLeft.push(i);
    else opticRight.push(i);
  }
  log(`optic   : ${opticLeft.length} left + ${opticRight.length} right`);

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

  /**
   * Run the real MANC spine: feed brain DN spike rates into the VNC's
   * DN-input neurons, step the VNC LIF, return motor rates per leg+side.
   * No-op (returns null) if vnc.bin wasn't loaded — caller falls back
   * to the synthetic spine path.
   */
  interface MancMotor {
    legs: Record<string, number>;          // T1_left … T3_right
    wing: number;                           // mean wing motor rate
    neck: number;                           // mean neck motor rate
    abdomen: number;                        // mean abdomen motor rate
  }

  async function runRealSpine(brainRate: Float32Array): Promise<MancMotor | null> {
    if (!vncSim || !vncMeta) return null;
    const N = vncSim.brain.header.numNeurons;
    const ext = new Float32Array(N);
    // Brain DN cell types match VNC DN cell types (same neurons,
    // brain side has soma, VNC side has axon terminal). Average the
    // brain's rate across each DN's L+R copies and write that into
    // every VNC neuron sharing the same name.
    for (const [name, brainIdxs] of Object.entries(famousDns)) {
      const vncIdxs = vncMeta.dn_inputs[name];
      if (!vncIdxs || vncIdxs.length === 0) continue;
      let sum = 0;
      for (const i of brainIdxs) sum += brainRate[i];
      const meanBrain = brainIdxs.length ? sum / brainIdxs.length : 0;
      const drive = meanBrain * 30;
      for (const idx of vncIdxs) ext[idx] = drive;
    }
    vncSim.setExternalInput(ext);
    const vncRate = await vncSim.captureRollingRate(20);
    // Aggregate motor neuron rates per leg+side.
    const legs: Record<string, number> = {};
    for (const [legKey, group] of Object.entries(vncMeta.motor)) {
      let sum = 0;
      for (const i of group.all) sum += vncRate[i];
      legs[legKey] = group.all.length ? sum / group.all.length : 0;
    }
    // Subclass aggregates (wm = wing, nm = neck, ad = abdomen).
    const aggSub = (key: string): number => {
      const idxs = vncMeta!.motor_by_subclass[key];
      if (!idxs || idxs.length === 0) return 0;
      let s = 0;
      for (const i of idxs) s += vncRate[i];
      return s / idxs.length;
    };
    return {
      legs,
      wing: aggSub("wm"),
      neck: aggSub("nm"),
      abdomen: aggSub("ad"),
    };
  }

  let driveFwd = 0, driveTurn = 0; // smoothed
  let lastSpineMode = "synthetic"; // tracked for the diagnostic readout
  let lastMancActivity = { legs: 0, wing: 0, neck: 0, abdomen: 0 };
  // Build VNC stand-in context once; reused by every drive update.
  const vncCtx: MotorContext = {
    famousDns,
    dnLeft,
    dnRight,
  };
  async function applyDriveFromSnapshot(rate: Float32Array, visual?: { angle: number; area: number }) {
    let sumC = 0;
    for (const i of centralIdxs) sumC += rate[i];
    const meanC = centralIdxs.length ? sumC / centralIdxs.length : 0;
    room.setEyeGlow(Math.min(1, meanC * 25));

    // Try the real Janelia MANC spine first. Falls back to the synthetic
    // 200-neuron VNC if vnc.bin wasn't loaded.
    const realMotor = await runRealSpine(rate);
    let targetFwd: number, targetTurn: number, jump: number;
    // Always run the synthetic spine — it owns direction sign + jump
    // detection. MANC contributes magnitude when its motor pools are
    // actually firing (which depends on the DN's natural pathway
    // strength in the connectome, not always strong on direct stim).
    const fbCmd = motorFromBrain(rate, { ...vncCtx, visual });
    let mancTotal = 0, mancAsym = 0;
    if (realMotor) {
      const leftKeys = ["T1_left", "T2_left", "T3_left"];
      const rightKeys = ["T1_right", "T2_right", "T3_right"];
      const meanOf = (ks: string[]) => {
        let s = 0, n = 0;
        for (const k of ks) if (k in realMotor.legs) { s += realMotor.legs[k]; n++; }
        return n ? s / n : 0;
      };
      const meanL = meanOf(leftKeys);
      const meanR = meanOf(rightKeys);
      mancTotal = meanL + meanR;
      mancAsym = meanR - meanL;
      lastMancActivity = { legs: mancTotal, wing: realMotor.wing, neck: realMotor.neck, abdomen: realMotor.abdomen };
    }
    const visualActive = visual && Number.isFinite(visual.angle) && visual.area > 0;
    if (visualActive) {
      // Visual reflex (computed inside motorFromBrain) owns the motor.
      // Skip the MANC magnitude blending — under closed-loop optic
      // stim the asymmetric motor signal is wrong-direction relative
      // to the visual cue (50-ms cascade can't develop the brain's
      // contralateral wiring). Synthetic VNC's reflex has correct sign.
      lastSpineMode = "visual reflex";
      targetFwd = Math.max(-1, Math.min(1, fbCmd.fwd));
      targetTurn = Math.max(-1, Math.min(1, fbCmd.turn));
    } else if (mancTotal > 0.001) {
      // MANC is firing meaningfully — use its magnitude with synthetic
      // direction sign.
      lastSpineMode = "MANC";
      const sign = fbCmd.fwd >= 0 ? 1 : -1;
      targetFwd = sign * Math.min(1, mancTotal * 60);
      targetTurn = Math.max(-1, Math.min(1, mancAsym * 80 + fbCmd.turn * 0.5));
    } else {
      // MANC silent — fall back to synthetic spine.
      lastSpineMode = realMotor ? "synthetic (MANC quiet)" : "synthetic";
      targetFwd = Math.max(-1, Math.min(1, fbCmd.fwd));
      targetTurn = Math.max(-1, Math.min(1, fbCmd.turn));
    }
    jump = fbCmd.jump;

    // Smoothing — exponential blend so gait feels less jittery.
    const alpha = 0.35;
    driveFwd  = driveFwd  + alpha * (targetFwd  - driveFwd);
    driveTurn = driveTurn + alpha * (targetTurn - driveTurn);

    room.setDrive(driveFwd, driveTurn);
    if (jump > 0.5) room.jumpImpulse(jump);

    driveReadout.textContent = `fwd ${driveFwd.toFixed(2)}  turn ${driveTurn.toFixed(2)}  jump=${jump.toFixed(2)}  spine=${lastSpineMode}`;
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

  // --- Real VNC (Janelia MANC connectome) ---------------------------------
  // Optional: if public/vnc.bin is present, load the 23k-neuron Male
  // Adult Nerve Cord connectome alongside the brain. Brain DN spike
  // rates flow into VNC DN-input neurons, the VNC LIF runs, and motor
  // neurons drive the leg actuators per leg + side.
  // If vnc.bin is missing, the synthetic 200-neuron spine in vnc.ts
  // remains the motor source — same code path, different controller.
  let vncSim: FlySim | null = null;
  let vncMeta: {
    dn_inputs: Record<string, number[]>;
    motor: Record<string, { all: number[] } & Record<string, number[]>>;
    motor_by_subclass: Record<string, number[]>;
    motor_by_target: Record<string, number[]>;
    num_neurons: number;
    num_edges: number;
  } | null = null;
  const vncUrl = import.meta.env.VITE_VNC_URL || "/vnc.bin";
  const vncMetaUrl = import.meta.env.VITE_VNC_META_URL || "/vnc.meta.json";
  bootStage("vnc", "run", "fetching MANC connectome (43 MB)…");
  try {
    const vncBrain = await loadBrain(vncUrl);
    const vncMetaResp = await fetch(vncMetaUrl);
    vncMeta = await vncMetaResp.json();
    vncSim = await FlySim.create(vncBrain, { ...DEFAULT_PARAMS });
    log(`real VNC loaded: ${vncBrain.header.numNeurons.toLocaleString()} neurons, ${vncBrain.header.numEdges.toLocaleString()} edges (Janelia MANC)`, "ok");
    log(`  ${Object.keys(vncMeta!.dn_inputs).length} DN types, ${Object.keys(vncMeta!.motor).length} leg groups`);
    bootStage("vnc", "ok", `${vncBrain.header.numNeurons.toLocaleString()} neurons, ${Object.keys(vncMeta!.motor).length} leg groups`);
  } catch (e) {
    log(`real VNC unavailable (${(e as Error).message}); using synthetic spine`, "warn");
    bootSkip("vnc", "synthetic 200-neuron fallback (vnc.bin missing)");
  }
  bootMaybeDismiss();
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

  // --- Famous-DN buttons: literature-documented descending neurons ---
  // Each button stims both L and R copies. Behaviour comes from the
  // connectome's existing wiring — we don't hand-pick a motor mapping.
  if (Object.keys(famousDns).length) {
    const dnSection = document.createElement("div");
    dnSection.style.marginTop = "10px";
    const h = document.createElement("h2");
    h.style.cssText = "margin: 0 0 6px 0; font-size: 11px; letter-spacing: 0.06em; text-transform: uppercase; color: #6c7480";
    h.textContent = "Famous DNs";
    dnSection.appendChild(h);
    const dnRow = document.createElement("div");
    dnRow.id = "dn-row";
    dnRow.style.cssText = "display: flex; flex-wrap: wrap; gap: 6px";
    dnSection.appendChild(dnRow);
    stimRow.parentElement?.appendChild(dnSection);

    for (const [name, idxs] of Object.entries(famousDns)) {
      const btn = document.createElement("button");
      btn.className = "stim-btn";
      btn.style.position = "relative";
      const desc = famousDnLabels[name] ?? "";
      // The ↗ link opens FlyWire Codex at this cell type, where the
      // user can browse the actual EM-traced neuron — both hemispheres,
      // synapse partners, neuropil maps. We use stopPropagation so the
      // link click doesn't also run the stim.
      btn.innerHTML = `
        <span class="label">${name}</span>
        <span class="hint">${desc}</span>
        <a class="ext-link" href="https://codex.flywire.ai/app/cell_details?cell_names_or_id=${encodeURIComponent(name)}" target="_blank" rel="noopener" title="open in FlyWire Codex">↗</a>
      `;
      const link = btn.querySelector(".ext-link") as HTMLAnchorElement;
      link.addEventListener("click", (e) => e.stopPropagation());
      btn.addEventListener("click", () => runDnStim(name, idxs, btn));
      dnRow.appendChild(btn);
      buttons.push(btn);
    }
  }

  // --- Closed-loop visual button ---
  // Continuously read fly→target angle, drive optic neurons accordingly,
  // step the brain in small bursts, push DN drive back to body. The fly
  // should turn toward the red sphere by virtue of its connectome wiring.
  const loopSection = document.createElement("div");
  loopSection.style.marginTop = "10px";
  const loopH = document.createElement("h2");
  loopH.style.cssText = "margin: 0 0 6px 0; font-size: 11px; letter-spacing: 0.06em; text-transform: uppercase; color: #6c7480";
  loopH.textContent = "Closed-loop";
  loopSection.appendChild(loopH);
  const loopBtn = document.createElement("button");
  loopBtn.className = "stim-btn";
  loopBtn.style.flex = "1 0 100%";
  loopBtn.innerHTML = `<span class="label">Track target</span><span class="hint">vision → brain → body, continuous</span>`;
  loopSection.appendChild(loopBtn);
  stimRow.parentElement?.appendChild(loopSection);
  buttons.push(loopBtn);

  // --- Evolve gait (WebGPU ARS) ---
  // Spawns 128 candidate CPG policies on the GPU, evaluates each via
  // a stripped hexapod fitness function, keeps the top 20%, resamples,
  // repeats for ~60 generations. Best policy is written into the live
  // mujoco_wasm body so the fly walks under evolved parameters. Real
  // sim-to-real, in your tab.
  const evoSection = document.createElement("div");
  evoSection.style.marginTop = "10px";
  const evoH = document.createElement("h2");
  evoH.style.cssText = "margin: 0 0 6px 0; font-size: 11px; letter-spacing: 0.06em; text-transform: uppercase; color: #6c7480";
  evoH.textContent = "Evolve";
  evoSection.appendChild(evoH);
  const evoBtn = document.createElement("button");
  evoBtn.className = "stim-btn";
  evoBtn.style.flex = "1 0 100%";
  evoBtn.innerHTML = `<span class="label">Evolve gait (WebGPU ARS)</span><span class="hint">128 policies × 60 generations</span>`;
  evoSection.appendChild(evoBtn);
  stimRow.parentElement?.appendChild(evoSection);
  buttons.push(evoBtn);

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
    // Compose the embodiment readout: motor command, body speed, and a
    // live snapshot of the VNC's forward / backward / escape pool rates
    // so you can see the spine is alive even when the brain is silent.
    const sp = room.bodySpeed();
    const vnc = vncSnapshot();
    const bar = (r: number, w = 10) => {
      const n = Math.max(0, Math.min(w, Math.round(r * w * 3)));
      return "█".repeat(n) + "·".repeat(w - n);
    };
    const lines = [
      `fwd ${driveFwd.toFixed(2)}  turn ${driveTurn.toFixed(2)}  speed ${sp.toFixed(2)} cm/s  <span style="opacity:0.5">[${lastSpineMode}]</span>`,
      `<span style="opacity:0.7">synth: fwd ${bar(vnc.fwdRate)}  bwd ${bar(vnc.bwdRate)}  esc ${bar(vnc.escape)}</span>`,
    ];
    if (vncMeta) {
      const m = lastMancActivity;
      lines.push(`<span style="opacity:0.7">MANC : leg ${bar(m.legs)}  wing ${bar(m.wing)}  abd ${bar(m.abdomen)}</span>`);
    }
    driveReadout.innerHTML = lines.join("<br>");
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

    sim.reset(); resetVnc();
    sim.setExternalInput(ext);
    viewer.clearSnapshots();
    room.resetFly();

    const t0 = performance.now();
    for (let s = 0; s < N_SNAPSHOTS; s++) {
      const rate = await sim.captureRollingRate(STEPS_PER_SNAPSHOT);
      viewer.pushSnapshot(rate);
      await applyDriveFromSnapshot(rate);
    }
    const elapsed = performance.now() - t0;
    const totalSteps = N_SNAPSHOTS * STEPS_PER_SNAPSHOT;
    log(`${totalSteps} steps in ${elapsed.toFixed(0)} ms wall (${(elapsed / totalSteps).toFixed(2)} ms/step)`, "ok");

    // Diagnostic: read back Vm and report distribution. If kernel ran
    // and synaptic drive is reaching neurons, max(Vm) should approach
    // v_thresh = -45 mV. If Vm sits at -52 (=v_rest) for everyone, the
    // kernel didn't move — that's a structural bug, not calibration.
    const vm = await sim.readVm();
    let vmMin = Infinity, vmMax = -Infinity, vmSum = 0;
    let aboveRest = 0;
    for (let i = 0; i < vm.length; i++) {
      if (vm[i] < vmMin) vmMin = vm[i];
      if (vm[i] > vmMax) vmMax = vm[i];
      vmSum += vm[i];
      if (vm[i] > sim.params.vRest + 0.001) aboveRest++;
    }
    log(`vm: min=${vmMin.toFixed(2)} max=${vmMax.toFixed(2)} mean=${(vmSum / vm.length).toFixed(2)} above-rest=${aboveRest.toLocaleString()}`);
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

  // --- Famous-DN stim: drive both L+R copies of a named DN ---
  async function runDnStim(name: string, idxs: number[], btn: HTMLButtonElement) {
    if (busy) return;
    busy = true;
    buttons.forEach((b) => { b.disabled = true; b.classList.remove("active"); });
    btn.classList.add("active");
    controls.hidden = true;

    log("");
    log(`--- DN stim: ${name} (${idxs.length} neurons, ${famousDnLabels[name] ?? ""}) ---`, "ok");
    const ext = new Float32Array(header.numNeurons);
    // Direct stim of just 2 DN neurons needs a hefty amplitude to
    // ignite a real cascade through the alpha-synapse-shaped fan-out.
    // Lower than ~3.0 leaves DN-with-weak-downstream (DNb01, DNg13,
    // DNp01) firing only the 2 stimmed cells with no propagation.
    for (const idx of idxs) ext[idx] = 4.0;
    sim.reset(); resetVnc();
    sim.setExternalInput(ext);
    viewer.clearSnapshots();
    if (idxs.length > 0) viewer.highlightNeuron(idxs[0]);
    room.resetFly();

    const t0 = performance.now();
    for (let s = 0; s < N_SNAPSHOTS; s++) {
      const rate = await sim.captureRollingRate(STEPS_PER_SNAPSHOT);
      viewer.pushSnapshot(rate);
      await applyDriveFromSnapshot(rate);
    }
    const elapsed = performance.now() - t0;
    log(`${N_SNAPSHOTS * STEPS_PER_SNAPSHOT} steps in ${elapsed.toFixed(0)} ms`, "ok");

    {
      const vm = await sim.readVm();
      let vmMax = -Infinity;
      for (let i = 0; i < vm.length; i++) if (vm[i] > vmMax) vmMax = vm[i];
      const vmAtIdx = idxs.length ? vm[idxs[0]] : NaN;
      log(`vm: max=${vmMax.toFixed(2)} mV  driven[0]=${vmAtIdx.toFixed(2)} mV`);
    }

    let recruited = 0;
    const last = viewer["snapshots"][viewer.numSnapshots - 1] as Float32Array;
    for (let i = 0; i < last.length; i++) if (last[i] > 0) recruited++;
    log(`final-window recruits: ${recruited.toLocaleString()} / ${header.numNeurons.toLocaleString()}`);
    if (recruited > 100) {
      const snaps = [...Array(viewer.numSnapshots)].map((_, j) => viewer["snapshots"][j] as Float32Array);
      logHeroValidation(snaps);
    }

    // Motor command was already produced each step by applyDriveFromSnapshot
    // → motorFromBrain, which reads this DN's *actual* spike rate from the
    // window and multiplies by its canonical primitive. No lookup table.
    log(`brain-driven motor: fwd=${driveFwd.toFixed(2)} turn=${driveTurn.toFixed(2)}`, "ok");

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

  // --- Closed-loop visual mode ---
  // Continuous: each tick reads fly→target angle, drives optic neurons
  // proportional to alignment, runs a small brain burst, applies DN
  // drive to body, repeats. Fly should turn toward target via the
  // connectome's existing optic→DN wiring — no hard-coded mapping.
  let continuousMode = false;
  async function runContinuousLoop(btn: HTMLButtonElement) {
    if (busy && !continuousMode) return;
    if (continuousMode) {
      continuousMode = false;
      btn.classList.remove("active");
      log("closed-loop tracking stopped");
      return;
    }
    continuousMode = true;
    busy = true;
    buttons.forEach((b) => { if (b !== btn) b.classList.remove("active"); });
    btn.classList.add("active");
    controls.hidden = true;
    log("");
    log(`--- closed-loop visual: track red target ---`, "ok");
    sim.reset(); resetVnc();
    viewer.clearSnapshots();
    room.resetFly();
    // Reset stale drive from previous stims so the fly starts from
    // standstill and reacts to THIS loop's sensor signal, not the
    // last preset's residual.
    driveFwd = 0;
    driveTurn = 0;
    room.setDrive(0, 0);
    // Yield long enough for the room's render tick to repaint the
    // retina from the just-reset body pose. Otherwise tick 1 reads
    // stale retina pixels (from before the reset, when the body had
    // wandered) and falsely reports the target lost.
    await new Promise((r) => setTimeout(r, 100));
    // Sample 4000 optic neurons per side — enough cascade to reach DN
    // through the connectome's optic→central wiring. Below ~2000 the
    // signal dissipates before producing meaningful DN activity.
    const sampleN = 4000;
    const stride = Math.max(1, Math.floor(opticLeft.length / sampleN));
    const lSubset: number[] = [];
    for (let i = 0; i < opticLeft.length; i += stride) lSubset.push(opticLeft[i]);
    const rStride = Math.max(1, Math.floor(opticRight.length / sampleN));
    const rSubset: number[] = [];
    for (let i = 0; i < opticRight.length; i += rStride) rSubset.push(opticRight[i]);

    const ext = new Float32Array(header.numNeurons);
    let tick = 0;
    let lostTicks = 0;
    let lastKnownTurnDir = 1;
    while (continuousMode) {
      // Sense from a real retinal render at the fly's head pose. No
      // geometry shortcut — pixels of the scene get sampled, red blob
      // centroid → angle. If target is behind, angle is NaN.
      const sample = room.retinalSample();
      const angle = sample.angle;
      const dist = room.targetDistance();
      ext.fill(0);
      if (Number.isFinite(angle) && sample.area > 0) {
        lostTicks = 0;
        lastKnownTurnDir = angle > 0 ? 1 : -1;
        const align = 1 - Math.abs(angle) / RETINA_FOV_RAD;     // 0..1
        const lScale = align * (angle > 0 ? 1.0 : 0.3);
        const rScale = align * (angle < 0 ? 1.0 : 0.3);
        const amp = 1.5 + 12 * sample.area;
        for (const i of lSubset) ext[i] = amp * lScale;
        for (const i of rSubset) ext[i] = amp * rScale;
      } else {
        lostTicks++;
      }
      sim.setExternalInput(ext);

      // Step brain in a short burst (50 ms simulated).
      const rate = await sim.captureRollingRate(50);
      viewer.pushSnapshot(rate);

      if (lostTicks > 0) {
        // Target off-camera. Bypass brain→spine motor (the residual
        // VNC state would keep pushing the fly forward off-axis) and
        // set drive directly. Fwd=0 to hold position; turn alternates
        // every 8 ticks. Start in the opposite direction of last seen
        // angle since fly almost always overshoots the target before
        // losing it.
        const scanCycle = Math.floor((lostTicks - 1) / 8) % 2 === 0 ? -1 : 1;
        driveFwd = 0;
        driveTurn = lastKnownTurnDir * scanCycle * 0.5;
        room.setDrive(driveFwd, driveTurn);
      } else {
        // Brain → VNC stand-in → body via the normal motor path.
        await applyDriveFromSnapshot(rate, sample);
      }

      tick++;
      if (tick <= 10 || tick % 10 === 0) {
        log(`  tick ${tick}: angle=${(angle * 180 / Math.PI).toFixed(0)}° dist=${dist.toFixed(1)}cm fwd=${driveFwd.toFixed(2)} turn=${driveTurn.toFixed(2)}`);
      }
      // Yield to render.
      await new Promise((r) => setTimeout(r, 0));
    }
    // Cleanup when loop exits.
    btn.classList.remove("active");
    buttons.forEach((b) => { b.disabled = false; });
    busy = false;
    controls.hidden = false;
  }
  loopBtn.addEventListener("click", () => runContinuousLoop(loopBtn));

  // Evolve-gait click: spin up GaitEvolver, run, hand winner to body.
  let evolver: GaitEvolver | null = null;
  evoBtn.addEventListener("click", async () => {
    if (busy) return;
    busy = true;
    buttons.forEach((b) => { b.disabled = true; b.classList.remove("active"); });
    evoBtn.classList.add("active");
    log("");
    log("--- evolving gait on WebGPU ---", "ok");
    try {
      if (!evolver) evolver = await GaitEvolver.create();
      const t0 = performance.now();
      const winner = await evolver.evolve({
        population: 128,
        generations: 60,
        onProgress: (gen, best, mean) => {
          if (gen % 5 === 0 || gen === 59) {
            log(`gen ${gen.toString().padStart(2)}  best=${best.toFixed(2)}  mean=${mean.toFixed(2)}`);
          }
        },
      });
      const elapsed = performance.now() - t0;
      log(`evolved in ${(elapsed / 1000).toFixed(1)} s`, "ok");
      log(`winner: freq=${winner.freq.toFixed(1)}Hz coxa=${winner.coxaAmp.toFixed(2)} femur=${winner.femurAmp.toFixed(2)} tibia=${winner.tibiaAmp.toFixed(2)} swing=${winner.swingRatio.toFixed(2)} lift=${winner.liftOffset.toFixed(2)}`);
      if (physics) {
        physics.applyEvolvedGait(winner);
        log("applied evolved gait to live body. press a DN button to walk.", "ok");
      } else {
        log("flybody not loaded yet; winner saved but not applied", "warn");
      }
    } catch (e) {
      log(`evolution failed: ${(e as Error).message}`, "err");
    }
    evoBtn.classList.remove("active");
    buttons.forEach((b) => { b.disabled = false; });
    busy = false;
  });

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
    sim.reset(); resetVnc();
    sim.setExternalInput(ext);
    viewer.clearSnapshots();
    viewer.highlightNeuron(idx);
    room.resetFly();

    const t0 = performance.now();
    for (let s = 0; s < N_SNAPSHOTS; s++) {
      const rate = await sim.captureRollingRate(STEPS_PER_SNAPSHOT);
      viewer.pushSnapshot(rate);
      await applyDriveFromSnapshot(rate);
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
