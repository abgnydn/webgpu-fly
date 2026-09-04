// game.ts — "Drosophila": race a real fly brain to a target by firing
// real DN neurons with keyboard keys. Every frame is a real LIF cascade
// through 139,255 FlyWire neurons; nothing is animated.
//
// Architecture:
//   - Game owns a state machine: idle → playing → won (or replay).
//   - Continuous brain loop: each tick builds ext_input from currently-
//     pressed keys, runs sim.captureRollingRate(50), feeds rate through
//     the existing applyDriveFromSnapshot pipeline. Same code path as
//     the closed-loop-visual button, just driven by keys not retina.
//   - Body (mujoco_wasm) renders at 60 fps via the room's render loop,
//     independent of the brain tick rate. So input feels responsive
//     even though the brain ticks at ~10-15 Hz.
//   - HUD updates each animation frame: timer, distance, score.
//   - Win = body within WIN_RADIUS cm of target. Stop clock, show result.
//   - Replay: record (t_step, key_idx) tuples; encode with target seed
//     into URL hash. On load with hash, enter replay mode and replay
//     keys at the recorded brain steps against the same seeded target.

import type { FlySim } from "./sim";
import type { Room } from "./room";
import type { FlyViewer } from "./viewer";

/** A famous DN button mapped to a keyboard key. */
export interface DnEntry {
  name: string;
  description: string;
  neurons: number[];
  key: string;            // single-character lowercase, e.g. "q"
}

export interface GameContext {
  dns: DnEntry[];                                       // ≤10 entries
  numNeurons: number;
  sim: FlySim;
  room: Room;
  viewer: FlyViewer;
  applyDrive: (rate: Float32Array, visual?: { angle: number; area: number }) => Promise<void>;
  /** Called when game wants to reset visualization snapshots. */
  resetSim: () => void;
  log: (s: string, cls?: "" | "ok" | "warn" | "err") => void;
  /** Optional fixed placement for the very first round (game-first landing page). */
  firstTarget?: { angleDeg: number; radiusCm: number };
}

const WIN_RADIUS_CM = 0.6;          // body within 6 mm of target = win
const STIM_AMP = 4.0;                // matches runDnStim in main.ts
const BURST_STEPS = 50;              // brain steps per tick (50 ms biological)
const TARGET_RADIUS_MIN_CM = 3.0;
const TARGET_RADIUS_MAX_CM = 6.0;

type State = "boot" | "idle" | "countdown" | "playing" | "won" | "replay";

/** Today's date as a deterministic 32-bit seed. Same value for every
 * player on the same UTC day → daily-challenge social hook without
 * any backend. */
function dailySeed(): number {
  const d = new Date();
  const ymd = d.getUTCFullYear() * 10000 + (d.getUTCMonth() + 1) * 100 + d.getUTCDate();
  // Splittable hash to spread bits.
  let s = ymd >>> 0;
  s = ((s ^ (s >>> 16)) * 0x85ebca6b) >>> 0;
  s = ((s ^ (s >>> 13)) * 0xc2b2ae35) >>> 0;
  s = (s ^ (s >>> 16)) >>> 0;
  return s;
}

interface ReplayEvent {
  t: number;     // brain steps since round start
  key: number;   // index into dns[]
  down: boolean; // press or release
}

export class Game {
  private ctx: GameContext;
  private state: State = "boot";
  private pressed = new Set<number>();
  private roundStart = 0;
  private roundStartStep = 0;
  private elapsedMs = 0;
  private events: ReplayEvent[] = [];
  private spikeCount = 0;
  private targetSeed = 0;
  private replayQueue: ReplayEvent[] = [];
  private replayIdx = 0;
  private brainLoopRunning = false;
  private dailyMode = false;
  private firstRound = true;

  // HUD elements
  private hud!: HTMLDivElement;
  private timerEl!: HTMLSpanElement;
  private distEl!: HTMLSpanElement;
  private spikesEl!: HTMLSpanElement;
  private bestEl!: HTMLSpanElement;
  private overlay!: HTMLDivElement;
  private winEl!: HTMLDivElement;
  private keyStripEl!: HTMLDivElement;

  constructor(ctx: GameContext) {
    this.ctx = ctx;
  }

  start() {
    this.buildHud();
    this.bindKeys();
    this.startBrainLoop().catch((e) => this.ctx.log(`brain loop stopped: ${(e as Error).message}`, "err"));
    this.startHudLoop();
    this.maybeEnterReplay();
    if (this.state === "boot") this.enterIdle();
  }

  /** Round clock in brain steps. Replay events are stamped and drained
   * on this clock, not wall time, so playback does not depend on the
   * display refresh rate or on how fast the brain loop is scheduled. */
  private roundStep(): number {
    return this.ctx.sim.currentStep - this.roundStartStep;
  }

  // ───── HUD construction ──────────────────────────────────────────

  private buildHud() {
    const hud = document.createElement("div");
    hud.id = "game-hud";
    hud.innerHTML = `
      <div id="game-top">
        <span class="hud-item">⏱ <span id="game-timer">00:00.000</span></span>
        <span class="hud-item">📏 <span id="game-dist">— cm</span></span>
        <span class="hud-item">⚡ <span id="game-spikes">0</span></span>
        <span class="hud-item">★ <span id="game-best">— —</span></span>
      </div>
      <div id="game-overlay"></div>
      <div id="game-keys"></div>
    `;
    document.body.appendChild(hud);

    this.hud = hud;
    this.timerEl = hud.querySelector("#game-timer") as HTMLSpanElement;
    this.distEl = hud.querySelector("#game-dist") as HTMLSpanElement;
    this.spikesEl = hud.querySelector("#game-spikes") as HTMLSpanElement;
    this.bestEl = hud.querySelector("#game-best") as HTMLSpanElement;
    this.overlay = hud.querySelector("#game-overlay") as HTMLDivElement;
    this.keyStripEl = hud.querySelector("#game-keys") as HTMLDivElement;

    // Win overlay sits on top; created on demand
    this.winEl = document.createElement("div");
    this.winEl.id = "game-win";
    this.winEl.style.display = "none";
    document.body.appendChild(this.winEl);

    this.renderKeyStrip();
    this.updateBest();
  }

  private renderKeyStrip() {
    this.keyStripEl.innerHTML = "";
    for (let i = 0; i < this.ctx.dns.length; i++) {
      const dn = this.ctx.dns[i];
      const div = document.createElement("div");
      div.className = "game-key";
      div.dataset.dn = dn.name;
      div.dataset.idx = String(i);
      div.innerHTML = `
        <span class="game-key-label">${dn.key.toUpperCase()}</span>
        <span class="game-key-name">${dn.name}</span>
        <span class="game-key-desc">${dn.description}</span>
      `;
      // Pointer events for mobile / mouse — press to fire, release to
      // stop. Same effect as keyboard down/up.
      const press = (down: boolean) => {
        if (this.state !== "playing") return;
        if (down) {
          if (!this.pressed.has(i)) {
            this.pressed.add(i);
            this.events.push({ t: this.roundStep(), key: i, down: true });
            this.flashKey(i, true);
          }
        } else {
          if (this.pressed.has(i)) {
            this.pressed.delete(i);
            this.events.push({ t: this.roundStep(), key: i, down: false });
            this.flashKey(i, false);
          }
        }
      };
      div.addEventListener("pointerdown", (e) => { e.preventDefault(); press(true); div.setPointerCapture(e.pointerId); });
      div.addEventListener("pointerup", (e) => { press(false); try { div.releasePointerCapture(e.pointerId); } catch {} });
      div.addEventListener("pointercancel", () => press(false));
      div.addEventListener("pointerleave", (e) => { if (e.buttons) press(false); });
      this.keyStripEl.appendChild(div);
    }
    // SPACE / M tap-affordances. Click = same effect as the keyboard
    // shortcut, so mobile players don't need a keyboard.
    const ctrl = document.createElement("div");
    ctrl.className = "game-key game-key-ctrl";
    ctrl.innerHTML = `
      <span class="game-key-label">SPC</span>
      <span class="game-key-name">reset</span>
      <span class="game-key-desc">new round</span>
    `;
    ctrl.addEventListener("click", () => {
      if (this.state === "idle" || this.state === "won") this.enterCountdown();
      else if (this.state === "playing") this.enterCountdown();
      else if (this.state === "replay") {
        history.replaceState(null, "", window.location.pathname + window.location.search);
        this.replayQueue = [];
        this.dailyMode = false;
        this.enterCountdown();
      }
    });
    this.keyStripEl.appendChild(ctrl);
    const sci = document.createElement("div");
    sci.className = "game-key game-key-ctrl";
    sci.innerHTML = `
      <span class="game-key-label">M</span>
      <span class="game-key-name">science</span>
      <span class="game-key-desc">switch view</span>
    `;
    sci.addEventListener("click", () => {
      const url = new URL(window.location.href);
      url.searchParams.set("mode", "science");
      window.location.href = url.toString();
    });
    this.keyStripEl.appendChild(sci);
  }

  // ───── State transitions ─────────────────────────────────────────

  private enterIdle() {
    this.state = "idle";
    const dateStr = new Date().toISOString().slice(0, 10);
    this.overlay.innerHTML = `
      <div class="overlay-card">
        <div class="overlay-title">Drosophila</div>
        <div class="overlay-sub">a real fly brain you can play</div>
        <div class="overlay-body">
          <p>You control a fly by firing real descending neurons in the
          <b>FlyWire</b> connectome (139,255 cells, 15M synapses).</p>
          <p>Each key fires both copies of a famous DN. The cascade
          propagates through the real wiring; the fly walks because the
          connectome says so.</p>
          <p>Race the timer. Reach the red target.</p>
        </div>
        <div class="overlay-modes">
          <button id="game-mode-random" class="overlay-btn primary">
            <span class="overlay-btn-title">random round</span>
            <span class="overlay-btn-key">SPACE</span>
          </button>
          <button id="game-mode-daily" class="overlay-btn">
            <span class="overlay-btn-title">daily challenge · ${dateStr}</span>
            <span class="overlay-btn-key">click</span>
          </button>
        </div>
      </div>
    `;
    this.overlay.style.display = "flex";
    this.winEl.style.display = "none";
    this.ctx.room.setDrive(0, 0);
    const randomBtn = this.overlay.querySelector("#game-mode-random") as HTMLButtonElement;
    const dailyBtn  = this.overlay.querySelector("#game-mode-daily")  as HTMLButtonElement;
    randomBtn.addEventListener("click", () => { this.dailyMode = false; this.enterCountdown(); });
    dailyBtn.addEventListener("click", () => { this.dailyMode = true; this.enterCountdown(); });
  }

  private enterCountdown() {
    this.state = "countdown";
    this.events = [];
    this.spikeCount = 0;
    this.pressed.clear();

    if (this.firstRound && this.ctx.firstTarget) {
      // Game-first landing page: place the first target dead ahead so the
      // player immediately sees the fly respond. Later rounds are random.
      const { angleDeg, radiusCm } = this.ctx.firstTarget;
      const angle = (angleDeg * Math.PI) / 180;
      const tx = Math.cos(angle) * radiusCm;
      const ty = Math.sin(angle) * radiusCm;
      this.ctx.room.targetPos[0] = tx;
      this.ctx.room.targetPos[1] = ty;
      this.ctx.room.setTargetPos(tx, ty);
      this.targetSeed = 0;
      this.ctx.firstTarget = undefined;
      this.firstRound = false;
    } else {
      this.firstRound = false;
      this.targetSeed = this.dailyMode ? dailySeed() : ((Math.random() * 0xffffffff) >>> 0);
      this.placeTargetFromSeed(this.targetSeed);
    }

    this.ctx.resetSim();
    this.ctx.room.resetFly();
    this.ctx.room.setDrive(0, 0);

    const seq = ["3", "2", "1", "GO!"];
    let i = 0;
    const tick = () => {
      this.overlay.innerHTML = `<div class="overlay-count">${seq[i]}</div>`;
      this.overlay.style.display = "flex";
      i++;
      if (i < seq.length) setTimeout(tick, 600);
      else setTimeout(() => this.enterPlaying(), 400);
    };
    tick();
  }

  private enterPlaying() {
    this.state = "playing";
    this.roundStart = performance.now();
    this.roundStartStep = this.ctx.sim.currentStep;
    this.overlay.style.display = "none";
    this.ctx.log(`game: round started, target seed ${this.targetSeed.toString(16)}`, "ok");
  }

  private enterWon() {
    this.state = "won";
    this.elapsedMs = performance.now() - this.roundStart;
    // Release whatever is still held, otherwise the encoded replay has
    // an unmatched down and the replayed fly stays stimulated forever.
    for (const k of this.pressed) this.events.push({ t: this.roundStep(), key: k, down: false });
    this.pressed.clear();
    this.ctx.room.setDrive(0, 0);
    const score = this.computeScore(this.elapsedMs, this.spikeCount);
    const isNewBest = this.recordBest(score);
    const replayUrl = this.encodeReplayUrl();
    const timeStr = this.formatTime(this.elapsedMs);
    // Behavior recipe: which DNs the player used, how long held in
    // total. Press events have `down: true`; pair each with the next
    // matching `down: false` to compute hold duration.
    // Event stamps are brain steps; DEFAULT_PARAMS.dtMs is 1.0, so one
    // step is 1 ms of simulated time and the durations are already ms.
    const holdMs = new Array(this.ctx.dns.length).fill(0);
    const lastDown = new Array(this.ctx.dns.length).fill(-1);
    for (const ev of this.events) {
      if (ev.down) lastDown[ev.key] = ev.t;
      else if (lastDown[ev.key] >= 0) {
        holdMs[ev.key] += Math.max(0, ev.t - lastDown[ev.key]);
        lastDown[ev.key] = -1;
      }
    }
    // Close any keys still held at win.
    for (let k = 0; k < holdMs.length; k++) {
      if (lastDown[k] >= 0) holdMs[k] += Math.max(0, this.roundStep() - lastDown[k]);
    }
    const ranked = holdMs
      .map((ms, i) => ({ ms, i }))
      .filter((x) => x.ms > 50)
      .sort((a, b) => b.ms - a.ms)
      .slice(0, 4);
    const recipeHtml = ranked.length ? `
      <div class="win-recipe">
        <div class="win-recipe-title">your fly's recipe</div>
        <div class="win-recipe-list">
          ${ranked.map((x) => `
            <div class="win-recipe-row">
              <span class="win-recipe-key">${this.ctx.dns[x.i].key.toUpperCase()}</span>
              <span class="win-recipe-name">${this.ctx.dns[x.i].name}</span>
              <span class="win-recipe-desc">${this.ctx.dns[x.i].description}</span>
              <span class="win-recipe-ms">${(x.ms / 1000).toFixed(2)}s</span>
            </div>`).join("")}
        </div>
      </div>
    ` : "";

    this.winEl.innerHTML = `
      <div class="win-card">
        <div class="win-title">${isNewBest ? "NEW BEST" : "caught it"}</div>
        <div class="win-stats">
          <div><span class="win-num">${timeStr}</span><span class="win-lbl">time</span></div>
          <div><span class="win-num">${this.spikeCount.toLocaleString()}</span><span class="win-lbl">spikes</span></div>
          <div><span class="win-num">${score.toLocaleString()}</span><span class="win-lbl">score</span></div>
        </div>
        ${recipeHtml}
        <div class="win-cta">press <b>SPACE</b> for another round</div>
        <div class="win-share">
          <button id="game-share-btn">↗ copy replay URL</button>
          <button id="game-tweet-btn">𝕏 share replay</button>
        </div>
      </div>
    `;
    this.winEl.style.display = "flex";
    const shareBtn = this.winEl.querySelector("#game-share-btn") as HTMLButtonElement;
    shareBtn.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(replayUrl);
        shareBtn.textContent = "✓ copied!";
        setTimeout(() => { shareBtn.textContent = "↗ copy replay URL"; }, 1800);
      } catch {
        prompt("copy this URL", replayUrl);
      }
    });
    const tweetBtn = this.winEl.querySelector("#game-tweet-btn") as HTMLButtonElement;
    tweetBtn.addEventListener("click", () => {
      const text = `caught the fly in ${timeStr} · ${this.spikeCount.toLocaleString()} real spikes · score ${score} · play it: `;
      const url = `https://twitter.com/intent/tweet?text=${encodeURIComponent(text)}&url=${encodeURIComponent(replayUrl)}`;
      window.open(url, "_blank", "noopener,noreferrer");
    });
    this.updateBest();
    this.ctx.log(`game: won in ${timeStr}, ${this.spikeCount} spikes, score ${score}${isNewBest ? " (new best!)" : ""}`, "ok");
  }

  private enterReplay(seed: number, events: ReplayEvent[]) {
    this.state = "replay";
    this.targetSeed = seed;
    this.replayQueue = events.slice().sort((a, b) => a.t - b.t);
    this.replayIdx = 0;
    this.events = [];
    this.spikeCount = 0;
    this.pressed.clear();
    this.placeTargetFromSeed(seed);
    this.ctx.resetSim();
    this.ctx.room.resetFly();
    this.ctx.room.setDrive(0, 0);
    this.overlay.innerHTML = `
      <div class="overlay-card">
        <div class="overlay-title">replay</div>
        <div class="overlay-sub">deterministic re-run · seed ${seed.toString(16).padStart(8, "0")}</div>
        <div class="overlay-body">
          <p>${events.length} keystrokes captured. The cascade and
          motor response will be identical to the original run because
          it is the same connectome on the same brain seed.</p>
          <p style="color:#6c7480;font-size:11px;">starting in 3s — press
          <b style="color:#9ad7ff;">SPACE</b> to play your own round instead</p>
        </div>
      </div>
    `;
    this.overlay.style.display = "flex";
    // Auto-start the replay so shared URLs need zero clicks. SPACE
    // jumps the user out into a fresh round.
    setTimeout(() => {
      if (this.state === "replay" && this.roundStart === 0) {
        this.startReplayPlayback();
      }
    }, 3000);
  }

  private startReplayPlayback() {
    if (this.state !== "replay") return;
    this.overlay.style.display = "none";
    this.roundStart = performance.now();
    this.roundStartStep = this.ctx.sim.currentStep;
    this.replayIdx = 0;
    this.ctx.log(`game: replaying ${this.replayQueue.length} events`, "ok");
    // Show a persistent "watching replay" banner during playback that
    // links out to your own round (SPACE).
    let banner = document.getElementById("game-replay-banner");
    if (!banner) {
      banner = document.createElement("div");
      banner.id = "game-replay-banner";
      banner.innerHTML = `<span>👁 watching replay</span><span class="bnr-cta">press <b>SPACE</b> for your own round</span>`;
      this.hud.appendChild(banner);
    }
    banner.style.display = "flex";
  }

  // ───── Keyboard ──────────────────────────────────────────────────

  private bindKeys() {
    const keyToIdx = new Map<string, number>();
    for (let i = 0; i < this.ctx.dns.length; i++) {
      keyToIdx.set(this.ctx.dns[i].key, i);
    }

    window.addEventListener("keydown", (e) => {
      const k = e.key.toLowerCase();
      if (k === "m") {
        // toggle to science mode
        const url = new URL(window.location.href);
        url.searchParams.set("mode", "science");
        window.location.href = url.toString();
        return;
      }
      if (k === " " || k === "spacebar" || e.code === "Space") {
        e.preventDefault();
        if (this.state === "idle") { this.dailyMode = false; this.enterCountdown(); }
        else if (this.state === "won") this.enterCountdown();
        else if (this.state === "replay") {
          // Either pre-playback (auto-start) or mid-playback (jump out
          // to a fresh, non-replay round). Clearing the URL hash makes
          // the "play your own" promise concrete.
          if (this.replayIdx === 0 && this.roundStart === 0) {
            // Starting before auto-timer? Eject to a fresh round.
            history.replaceState(null, "", window.location.pathname + window.location.search);
            this.dailyMode = false;
            this.enterCountdown();
          } else {
            // Mid-replay or after — eject to a fresh round.
            history.replaceState(null, "", window.location.pathname + window.location.search);
            this.replayQueue = [];
            this.dailyMode = false;
            this.enterCountdown();
          }
        }
        return;
      }
      const idx = keyToIdx.get(k);
      if (idx === undefined) return;
      if (this.state !== "playing") return;
      if (e.repeat) return;
      this.pressed.add(idx);
      this.events.push({ t: this.roundStep(), key: idx, down: true });
      this.flashKey(idx, true);
    });

    window.addEventListener("keyup", (e) => {
      const k = e.key.toLowerCase();
      const idx = keyToIdx.get(k);
      if (idx === undefined) return;
      if (this.state !== "playing") return;
      if (this.pressed.has(idx)) {
        this.pressed.delete(idx);
        this.events.push({ t: this.roundStep(), key: idx, down: false });
        this.flashKey(idx, false);
      }
    });
  }

  private flashKey(idx: number, on: boolean) {
    const el = this.keyStripEl.children[idx] as HTMLDivElement | undefined;
    if (!el) return;
    if (on) el.classList.add("active");
    else el.classList.remove("active");
  }

  // ───── Continuous brain loop ─────────────────────────────────────

  private async startBrainLoop() {
    if (this.brainLoopRunning) return;
    this.brainLoopRunning = true;
    const ext = new Float32Array(this.ctx.numNeurons);

    // The loop runs forever; whether stim is applied depends on game state.
    while (this.brainLoopRunning) {
      // Build ext_input from currently-pressed keys (or replay state).
      ext.fill(0);
      const activeIdxs = this.activeStimIdxs();
      for (const dnIdx of activeIdxs) {
        const dn = this.ctx.dns[dnIdx];
        if (!dn) continue;
        for (const i of dn.neurons) ext[i] = STIM_AMP;
      }

      this.ctx.sim.setExternalInput(ext);
      const rate = await this.ctx.sim.captureRollingRate(BURST_STEPS);
      this.ctx.viewer.pushSnapshot(rate);

      // Sum spikes in this burst for the score.
      if (this.state === "playing" || this.state === "replay") {
        let s = 0;
        for (let i = 0; i < rate.length; i++) s += rate[i];
        // captureRollingRate returns spikes-per-step normalised, so
        // multiply by BURST_STEPS to get total spikes in the burst.
        this.spikeCount += Math.round(s * BURST_STEPS);
      }

      // Drive body via the existing brain → spine → motor pipeline.
      // Visual reflex would interfere with key-driven control, so we
      // pass no visual sample.
      await this.ctx.applyDrive(rate, undefined);

      // Replay-mode: advance the key queue by elapsed brain steps.
      if (this.state === "replay" && this.roundStart > 0) {
        const t = this.roundStep();
        while (this.replayIdx < this.replayQueue.length
               && this.replayQueue[this.replayIdx].t <= t) {
          const ev = this.replayQueue[this.replayIdx++];
          if (ev.down) {
            this.pressed.add(ev.key);
            this.flashKey(ev.key, true);
          } else {
            this.pressed.delete(ev.key);
            this.flashKey(ev.key, false);
          }
        }
      }

      // Yield to the browser (lets render + raf run).
      await new Promise((r) => setTimeout(r, 0));
    }
  }

  private activeStimIdxs(): number[] {
    if (this.state === "playing" || this.state === "replay") {
      return Array.from(this.pressed);
    }
    return [];
  }

  // ───── HUD loop (60 fps) ─────────────────────────────────────────

  private startHudLoop() {
    const tick = () => {
      this.updateHud();
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }

  private updateHud() {
    // Replay banner only visible while playback is running.
    const banner = document.getElementById("game-replay-banner");
    if (banner) banner.style.display = (this.state === "replay" && this.roundStart > 0) ? "flex" : "none";

    const dist = this.ctx.room.targetDistance();
    if (Number.isFinite(dist)) {
      this.distEl.textContent = `${dist.toFixed(2)} cm`;
    }
    if (this.state === "playing") {
      const t = performance.now() - this.roundStart;
      this.timerEl.textContent = this.formatTime(t);
      this.spikesEl.textContent = this.spikeCount.toLocaleString();
      // Win check
      if (Number.isFinite(dist) && dist < WIN_RADIUS_CM) {
        this.enterWon();
      }
    } else if (this.state === "replay" && this.roundStart > 0) {
      const t = performance.now() - this.roundStart;
      this.timerEl.textContent = this.formatTime(t);
      this.spikesEl.textContent = this.spikeCount.toLocaleString();
      // Reaching the target ends the playback, but deliberately does not
      // go through enterWon() — the score and the share card belong to
      // whoever recorded the run, not to whoever opened the link.
      if (Number.isFinite(dist) && dist < WIN_RADIUS_CM) {
        this.pressed.clear();
        this.ctx.room.setDrive(0, 0);
        this.elapsedMs = t;
        this.state = "won";
      }
    } else if (this.state === "won") {
      this.timerEl.textContent = this.formatTime(this.elapsedMs);
    }
  }

  private formatTime(ms: number): string {
    const totalSec = ms / 1000;
    const min = Math.floor(totalSec / 60);
    const sec = totalSec - min * 60;
    return `${min.toString().padStart(2, "0")}:${sec.toFixed(3).padStart(6, "0")}`;
  }

  // ───── Score / best ──────────────────────────────────────────────

  private computeScore(ms: number, spikes: number): number {
    // Time bonus: 5 sec free, then 0. Efficiency: 50K spikes = 2× bonus,
    // 200K = 0.5×, capped both ways. Reasonable plays land 600-7000.
    const timeBonus = Math.max(0, 5000 - ms);
    const efficiency = Math.max(0.5, Math.min(2.0, 100_000 / Math.max(spikes, 1)));
    return Math.max(1, Math.round(timeBonus * efficiency));
  }

  private recordBest(score: number): boolean {
    const key = "drosophila-best";
    const prev = parseInt(localStorage.getItem(key) ?? "0", 10) || 0;
    if (score > prev) {
      localStorage.setItem(key, String(score));
      localStorage.setItem("drosophila-best-time", String(this.elapsedMs));
      return true;
    }
    return false;
  }

  private updateBest() {
    const score = parseInt(localStorage.getItem("drosophila-best") ?? "0", 10);
    const t = parseInt(localStorage.getItem("drosophila-best-time") ?? "0", 10);
    if (score > 0 && t > 0) {
      this.bestEl.textContent = `${this.formatTime(t)} (${score})`;
    } else {
      this.bestEl.textContent = "— —";
    }
  }

  // ───── Target seeding ────────────────────────────────────────────

  /** Deterministic LCG. Two outputs → (radius angle) for target. */
  private placeTargetFromSeed(seed: number) {
    let s = seed >>> 0;
    const next = () => {
      s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
      return s / 0x100000000;
    };
    // Place target in front-arc of the fly: angle ∈ [-50°, +50°],
    // radius ∈ [3, 6] cm.
    const angle = (next() - 0.5) * (100 * Math.PI / 180);
    const radius = TARGET_RADIUS_MIN_CM + next() * (TARGET_RADIUS_MAX_CM - TARGET_RADIUS_MIN_CM);
    const tx = Math.cos(angle) * radius;
    const ty = Math.sin(angle) * radius;
    // Room's targetPos is [x, y, z] in MJ frame; mutate in place and
    // poke the visual marker via the room API.
    this.ctx.room.targetPos[0] = tx;
    this.ctx.room.targetPos[1] = ty;
    this.ctx.room.setTargetPos(tx, ty);
  }

  // ───── Replay encoding ───────────────────────────────────────────

  /** FNV-1a over the DN names. Events encode indices into dns[], so a
   * reorder, insert or removal has to invalidate old URLs; hashing the
   * roster does that automatically, with no version byte to maintain. */
  private dnFingerprint(): number {
    let h = 0x811c9dc5;
    for (const dn of this.ctx.dns) {
      for (let i = 0; i < dn.name.length; i++) {
        h ^= dn.name.charCodeAt(i);
        h = Math.imul(h, 0x01000193) >>> 0;
      }
      h ^= 0x2c; h = Math.imul(h, 0x01000193) >>> 0;  // separator
    }
    return h & 0xffff;
  }

  private encodeReplayUrl(): string {
    // Format: 2B dn fingerprint (LE) + 4B seed (LE) + 4B per event
    // (u24 t_steps LE + u8 (key|down)).
    const N = this.events.length;
    const buf = new Uint8Array(6 + 4 * N);
    const v = new DataView(buf.buffer);
    v.setUint16(0, this.dnFingerprint(), true);
    v.setUint32(2, this.targetSeed, true);
    for (let i = 0; i < N; i++) {
      const e = this.events[i];
      const t = Math.min(0xffffff, Math.max(0, Math.round(e.t)));
      v.setUint8(6 + i * 4 + 0, t & 0xff);
      v.setUint8(6 + i * 4 + 1, (t >> 8) & 0xff);
      v.setUint8(6 + i * 4 + 2, (t >> 16) & 0xff);
      v.setUint8(6 + i * 4 + 3, ((e.key & 0x7f) << 1) | (e.down ? 1 : 0));
    }
    let bin = "";
    for (let i = 0; i < buf.length; i++) bin += String.fromCharCode(buf[i]);
    const b64 = btoa(bin).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
    const u = new URL(window.location.href);
    u.hash = `r=${b64}`;
    u.searchParams.set("mode", "game");
    return u.toString();
  }

  private maybeEnterReplay() {
    const m = window.location.hash.match(/r=([A-Za-z0-9_-]+)/);
    if (!m) return;
    try {
      const b64 = m[1].replace(/-/g, "+").replace(/_/g, "/");
      const pad = "===".slice(0, (4 - (b64.length % 4)) % 4);
      const bin = atob(b64 + pad);
      const buf = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
      const v = new DataView(buf.buffer);
      // Anything that is not this build's roster is unplayable: the key
      // indices mean something else, and pre-fingerprint payloads stamp
      // wall-clock ms where this build expects brain steps.
      if (buf.length < 6 || (buf.length - 6) % 4 !== 0
          || v.getUint16(0, true) !== this.dnFingerprint()) {
        this.ctx.log("replay was recorded against a different DN set — ignoring", "warn");
        return;
      }
      const seed = v.getUint32(2, true);
      const N = (buf.length - 6) >> 2;
      const events: ReplayEvent[] = [];
      for (let i = 0; i < N; i++) {
        const t = v.getUint8(6 + i * 4 + 0)
                | (v.getUint8(6 + i * 4 + 1) << 8)
                | (v.getUint8(6 + i * 4 + 2) << 16);
        const packed = v.getUint8(6 + i * 4 + 3);
        const key = (packed >> 1) & 0x7f;
        if (key < this.ctx.dns.length) events.push({ t, key, down: (packed & 1) === 1 });
      }
      this.enterReplay(seed, events);
      // Set roundStart to 0 so SPACE triggers playback.
      this.roundStart = 0;
    } catch (e) {
      this.ctx.log(`replay decode failed: ${(e as Error).message}`, "warn");
    }
  }
}
