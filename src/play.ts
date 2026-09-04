// play.ts — game-first landing page.
//
// Reuses the brain/room/viewer/Game engine without modification. The body is
// synthetic-spine only (no VNC.bin load). WASD + arrow keys drive the fly;
// the first round is placed dead ahead so first-time players immediately see
// a response.
//
// OPEN ACCEPTANCE RISK: A/D hemisphere-set stimulation fires hundreds of
// descending neurons at STIM_AMP (4.0) versus the two-neuron famous-DN stims
// (W/S). This may saturate the cascade and/or turn the wrong way. The human
// playtest will measure heading response and remap if needed.

import { loadBrain, type Brain } from "./brain";
import { FlySim, DEFAULT_PARAMS } from "./sim";
import { FlyViewer } from "./viewer";
import { Room } from "./room";
import { Physics } from "./physics";
import { motorFromBrain, resetVnc, type MotorContext } from "./vnc";
import { loadManifest, type VersionFor } from "./manifest";
import { progressText } from "./cache";
import { Game, type DnEntry } from "./game";

const HERO_DN = 7; // cell_type lower 8 bits → descending neuron

const out = document.getElementById("out") as HTMLDivElement;

function log(msg: string, cls: "ok" | "warn" | "err" | "" = "") {
  const span = document.createElement("span");
  if (cls) span.className = cls;
  span.textContent = msg + "\n";
  out.appendChild(span);
  out.scrollTop = out.scrollHeight;
}

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

function bootFail(msg: string, link?: { href: string; text: string }) {
  const el = document.querySelector<HTMLElement>("#boot .blink");
  if (!el) return;
  el.textContent = msg;
  el.style.color = "#ff6b6b";
  el.style.animation = "none";
  if (link) {
    const a = document.createElement("a");
    a.href = link.href;
    a.textContent = link.text;
    a.style.cssText = "display: block; margin-top: 10px; color: #9ad7ff";
    el.appendChild(a);
  }
}

async function main() {
  const gpuOk = await (async () => {
    try { return !!(navigator.gpu && await navigator.gpu.requestAdapter()); }
    catch { return false; }
  })();
  if (!gpuOk) {
    log("no usable WebGPU adapter — needs Chrome, Edge, or Safari 26+", "err");
    bootFail(
      "WebGPU unavailable here — the simulator needs Chrome, Edge, or Safari 26+ (Firefox: Windows only, 141+).",
      { href: "index.html", text: "→ read what this project is (no WebGPU needed)" },
    );
    return;
  }

  if (window.self !== window.top) {
    const hint = document.querySelector<HTMLElement>("#boot .blink");
    if (hint) {
      const a = document.createElement("a");
      a.href = window.location.href;
      a.target = "_blank";
      a.rel = "noopener";
      a.textContent = "↗ open in its own tab so the download is cached";
      a.style.cssText = "display: block; margin-top: 8px; color: #9ad7ff; font-size: 12px";
      hint.appendChild(a);
    }
  }

  const versionFor: VersionFor = await loadManifest();
  const bundleVer = versionFor("flybody.bundle.bin");
  if (bundleVer) {
    (globalThis as unknown as { __flybodyBundleVersion?: string }).__flybodyBundleVersion =
      bundleVer.replace(/^\?v=/, "");
  }

  const brainUrl = import.meta.env.VITE_BRAIN_URL || "/brain.bin";
  const metaUrl = import.meta.env.VITE_BRAIN_META_URL || "/brain.meta.json";
  log(`loading brain from ${brainUrl} ...`);
  bootStage("brain", "run", "fetching connectome (120 MB)…");
  let brain: Brain;
  try {
    brain = await loadBrain(brainUrl + versionFor("brain.bin"), (got, total) => {
      bootStage("brain", "run", `fetching connectome — ${progressText(got, total)}`);
    });
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

  let pos = 0, neg = 0;
  for (let i = 0; i < header.numNeurons; i++) {
    if (neurons.sign[i] > 0) pos++;
    else if (neurons.sign[i] < 0) neg++;
  }
  log(`exc=${pos.toLocaleString()}  inh=${neg.toLocaleString()}`);

  let famousDns: Record<string, number[]> = {};
  let famousDnLabels: Record<string, string> = {};
  try {
    const r = await fetch(metaUrl + versionFor("brain.meta.json"));
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const meta = await r.json();
    famousDns = meta.famous_dns ?? {};
    famousDnLabels = meta.famous_dn_descriptions ?? {};
  } catch (e) {
    log(`brain.meta.json unavailable (${(e as Error).message}); famous-DN buttons disabled`, "warn");
  }

  const viewer = new FlyViewer(brain, {
    container: document.getElementById("canvas-container") as HTMLElement,
    bg: 0x06070a,
  });
  const room = new Room({
    container: document.getElementById("room-container") as HTMLElement,
    bg: 0x0a0d12,
  });

  const sim = await FlySim.create(brain, { ...DEFAULT_PARAMS });
  log(`FlySim ready. dt=${sim.params.dtMs} ms  tau=${sim.params.tauMs} ms`);

  // Pre-bin DN indices by hemisphere using pos_x relative to brain centroid.
  let cx = 0, np = 0;
  for (let i = 0; i < header.numNeurons; i++) {
    const x = neurons.pos[3 * i];
    if (x !== 0) { cx += x; np++; }
  }
  cx = np > 0 ? cx / np : 0;
  const dnLeft: number[] = [];
  const dnRight: number[] = [];
  for (let i = 0; i < header.numNeurons; i++) {
    if ((neurons.cellType[i] & 0xff) !== HERO_DN) continue;
    if (neurons.pos[3 * i] < cx) dnLeft.push(i);
    else dnRight.push(i);
  }

  // Start flybody loading in the background; the game starts once it is ready.
  bootStage("body", "run", "fetching flybody MJCF + 85 meshes…");
  let physicsResolve!: (p: Physics) => void;
  let physicsReject!: (e: Error) => void;
  const physicsReady = new Promise<Physics>((res, rej) => {
    physicsResolve = res;
    physicsReject = rej;
  });
  Physics.create((msg) => {
    if (!msg.startsWith("fetching flybody bundle —")) log(`flybody: ${msg}`);
    bootStage("body", "run", msg);
  })
    .then(async (p) => {
      await room.attachPhysics(p);
      // Expose for e2e probing — same read-only hook as main.ts.
      (window as unknown as { __physicsForTest: Physics }).__physicsForTest = p;
      (window as unknown as { __roomForTest: Room }).__roomForTest = room;
      log(`flybody attached (${p.bodyCount} bodies)`, "ok");
      bootStage("body", "ok", `${p.bodyCount} bodies, 85 meshes ready`);
      bootMaybeDismiss();
      physicsResolve(p);
    })
    .catch((e) => {
      log(`flybody failed to load: ${(e as Error).message}`, "warn");
      bootStage("body", "run", `failed: ${(e as Error).message}`);
      physicsReject(e);
    });

  // No real VNC in the game-first page — the synthetic spine in vnc.ts is enough.
  bootSkip("vnc", "not loaded in play mode");

  // Synthetic-spine drive (no MANC, no eye glow, no honest brain-cascade path).
  let driveFwd = 0, driveTurn = 0;
  const vncCtx: MotorContext = { famousDns, dnLeft, dnRight };
  async function applyDrive(rate: Float32Array) {
    const fbCmd = motorFromBrain(rate, vncCtx);
    const targetFwd = Math.max(-1, Math.min(1, fbCmd.fwd));
    const straightDamp = 1 - 0.75 * Math.min(1, Math.abs(targetFwd));
    const targetTurn = Math.max(-1, Math.min(1, fbCmd.turn * straightDamp));
    const alpha = 0.35;
    driveFwd += alpha * (targetFwd - driveFwd);
    driveTurn += alpha * (targetTurn - driveTurn);
    room.setDrive(driveFwd, driveTurn);
    if (fbCmd.jump > 0.5) room.jumpImpulse(fbCmd.jump);
  }

  function resetSim() {
    sim.reset();
    resetVnc();
    viewer.clearSnapshots();
    driveFwd = 0;
    driveTurn = 0;
  }

  // WASD controls. A/D are entire left/right DN hemispheres — see acceptance
  // risk comment at the top of this file.
  const dns: DnEntry[] = [
    {
      key: "w",
      name: "DNa01",
      description: famousDnLabels.DNa01 ?? "forward walking",
      neurons: famousDns.DNa01 ?? [],
    },
    {
      key: "s",
      name: "DNb01",
      description: famousDnLabels.DNb01 ?? "backward / moonwalker",
      neurons: famousDns.DNb01 ?? [],
    },
    {
      key: "a",
      name: "DN-left",
      description: "left descending neurons · steering",
      neurons: dnLeft,
    },
    {
      key: "d",
      name: "DN-right",
      description: "right descending neurons · steering",
      neurons: dnRight,
    },
  ];

  // Arrow keys dispatch synthetic WASD events so Game's existing listener
  // handles them (it does not check isTrusted). Repeats are ignored, matching
  // Game's own guard.
  const ARROW_TO_KEY: Record<string, string> = {
    ArrowUp: "w",
    ArrowDown: "s",
    ArrowLeft: "a",
    ArrowRight: "d",
  };
  window.addEventListener("keydown", (e) => {
    const mapped = ARROW_TO_KEY[e.key];
    if (!mapped) return;
    if (e.repeat) return;
    e.preventDefault();
    window.dispatchEvent(new KeyboardEvent("keydown", { key: mapped, bubbles: true }));
  });
  window.addEventListener("keyup", (e) => {
    const mapped = ARROW_TO_KEY[e.key];
    if (!mapped) return;
    e.preventDefault();
    window.dispatchEvent(new KeyboardEvent("keyup", { key: mapped, bubbles: true }));
  });

  // Retina mini-overlay: Y-flip so the image reads "as the fly sees".
  const retinaCanvas = document.getElementById("retina-mini") as HTMLCanvasElement;
  const retinaCtx2d = retinaCanvas.getContext("2d")!;
  const retinaImg = retinaCtx2d.createImageData(64, 16);
  room.onRetinaUpdate = () => {
    const frame = room.retinaFrame();
    const src = frame.pixels;
    const dst = retinaImg.data;
    for (let y = 0; y < frame.h; y++) {
      const srcRow = (frame.h - 1 - y) * frame.w * 4;
      const dstRow = y * frame.w * 4;
      for (let i = 0; i < frame.w * 4; i++) dst[dstRow + i] = src[srcRow + i];
    }
    retinaCtx2d.putImageData(retinaImg, 0, 0);
  };

  const headingEl = document.getElementById("play-heading") as HTMLDivElement;
  const bestEl = document.getElementById("play-best") as HTMLSpanElement;
  let bestDist = Infinity;

  function tick() {
    const angle = room.targetAngle();
    // room.targetAngle(): positive radians = target on the fly's left;
    // negative = target on the right. CSS rotate() is clockwise for positive
    // degrees, so we negate the math angle to make the compass arrow point
    // toward the target.
    if (Number.isFinite(angle)) {
      const deg = -angle * (180 / Math.PI);
      headingEl.style.transform = `rotate(${deg}deg)`;
    }

    const dist = room.targetDistance();
    if (!Number.isFinite(dist)) {
      bestDist = Infinity;
    } else {
      bestDist = Math.min(bestDist, dist);
      bestEl.textContent = `${bestDist.toFixed(2)} cm`;
    }

    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);

  physicsReady
    .then(() => {
      const game = new Game({
        dns,
        numNeurons: header.numNeurons,
        sim,
        room,
        viewer,
        applyDrive,
        resetSim,
        log,
        firstTarget: { angleDeg: 0, radiusCm: 1.5 },
      });
      game.start();
      (window as unknown as { __game: Game }).__game = game;
      log("play mode: ready", "ok");
    })
    .catch(() => { /* boot stage already logged */ });
}

main().catch((e) => {
  log(`uncaught: ${(e as Error).stack ?? e}`, "err");
  bootFail(`failed: ${(e as Error).message}`);
});
