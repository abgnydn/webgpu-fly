// viewer.ts — Three.js point-cloud renderer for the FlyWire brain with
// per-snapshot activity colouring + a time scrub bar.

import * as THREE from "three";
import type { Brain } from "./brain";

export interface ViewerOpts {
  container: HTMLElement;
  /** Per-step activity colour ramp: 0 → cool, 1 → hot. */
  pointSize?: number;
  /** Background colour (default near-black). */
  bg?: number;
}

// Anatomy palette indexed by super_class enum (matches SUPER_CLASS_TABLE in
// build_csr.py). Tuned dim so additive blending lets activity overpower it.
const CLASS_BASE: Array<[number, number, number]> = [
  [0.025, 0.025, 0.030], // 0 unknown
  [0.090, 0.060, 0.030], // 1 sensory       — warm
  [0.060, 0.080, 0.030], // 2 ascending     — pale yellow-green
  [0.040, 0.040, 0.040], // 3 intrinsic
  [0.030, 0.060, 0.090], // 4 central       — blue
  [0.110, 0.025, 0.030], // 5 descending    — red
  [0.080, 0.020, 0.020], // 6 motor         — deep red
  [0.070, 0.030, 0.060], // 7 endocrine     — magenta
  [0.060, 0.040, 0.090], // 8 visual_centrifugal — lavender
  [0.030, 0.080, 0.040], // 9 visual_projection — green
  [0.030, 0.060, 0.060], // 10 optic        — teal
];

// Hero cell_type enum (lower 8 bits of cellType) overrides super_class palette
// for a few "famous" populations so they're visually identifiable at rest.
const HERO_BASE: Record<number, [number, number, number]> = {
  1: [0.110, 0.090, 0.030], // KC kenyon — yellow
  2: [0.030, 0.110, 0.110], // MBON      — cyan
  3: [0.080, 0.030, 0.110], // LHN       — violet
  4: [0.030, 0.110, 0.060], // PN        — green
  5: [0.130, 0.060, 0.020], // ORN       — orange
  6: [0.150, 0.020, 0.020], // GF        — bright red
  7: [0.120, 0.030, 0.030], // DN        — red
};

export class FlyViewer {
  readonly scene = new THREE.Scene();
  readonly camera: THREE.PerspectiveCamera;
  readonly renderer: THREE.WebGLRenderer;

  private points: THREE.Points;
  private colorAttr: THREE.BufferAttribute;
  private baseColors: Float32Array;
  private snapshots: Float32Array[] = [];
  private currentIdx = 0;
  private brain: Brain;
  private autoplay = false;
  private rafId = 0;

  // simple orbit state — drag to rotate, wheel to zoom
  private isDragging = false;
  private didDrag = false;
  private downAt = { x: 0, y: 0 };
  private prev = { x: 0, y: 0 };
  private azimuth = 0;
  private elevation = 0.3;
  private radius = 800_000; // FAFB14 brain ~700k nm wide

  // pick / record state
  private raycaster = new THREE.Raycaster();
  private pickListener: ((idx: number, screen: { x: number; y: number }) => void) | null = null;
  private highlightDot: THREE.Mesh | null = null;
  private mediaRecorder: MediaRecorder | null = null;
  private recordChunks: Blob[] = [];

  constructor(brain: Brain, opts: ViewerOpts) {
    this.brain = brain;

    const { container } = opts;
    const w = container.clientWidth;
    const h = container.clientHeight;

    this.renderer = new THREE.WebGLRenderer({ antialias: false });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(w, h);
    container.appendChild(this.renderer.domElement);

    this.scene.background = new THREE.Color(opts.bg ?? 0x06070a);

    this.camera = new THREE.PerspectiveCamera(45, w / h, 1, 1e7);
    this.updateCameraFromOrbit();

    // --- center the brain at origin ---
    const N = brain.header.numNeurons;
    const positions = new Float32Array(N * 3);
    let cx = 0, cy = 0, cz = 0, np = 0;
    for (let i = 0; i < N; i++) {
      const x = brain.neurons.pos[3 * i];
      if (x === 0) continue; // un-positioned neurons sit at origin in the bin
      cx += x; cy += brain.neurons.pos[3 * i + 1]; cz += brain.neurons.pos[3 * i + 2];
      np++;
    }
    if (np > 0) { cx /= np; cy /= np; cz /= np; }
    for (let i = 0; i < N; i++) {
      positions[3 * i] = brain.neurons.pos[3 * i] - cx;
      positions[3 * i + 1] = brain.neurons.pos[3 * i + 1] - cy;
      positions[3 * i + 2] = brain.neurons.pos[3 * i + 2] - cz;
    }

    // Per-neuron anatomy baseline: super_class palette, with hero cell_type
    // overrides on top. Activity later lerps from baseline → magma.
    this.baseColors = new Float32Array(N * 3);
    for (let i = 0; i < N; i++) {
      const sc = brain.neurons.superClass[i];
      const base = CLASS_BASE[sc] ?? CLASS_BASE[0];
      const hero = brain.neurons.cellType[i] & 0xff;
      const c = HERO_BASE[hero] ?? base;
      this.baseColors[3 * i]     = c[0];
      this.baseColors[3 * i + 1] = c[1];
      this.baseColors[3 * i + 2] = c[2];
    }
    const colors = this.baseColors.slice();

    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    this.colorAttr = new THREE.BufferAttribute(colors, 3);
    geom.setAttribute("color", this.colorAttr);

    const mat = new THREE.PointsMaterial({
      size: opts.pointSize ?? 800,
      vertexColors: true,
      transparent: true,
      opacity: 1.0,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });

    this.points = new THREE.Points(geom, mat);
    this.scene.add(this.points);

    this.attachInput();
    this.startRenderLoop();
    window.addEventListener("resize", () => this.onResize(container));
  }

  private updateCameraFromOrbit() {
    const ce = Math.cos(this.elevation), se = Math.sin(this.elevation);
    const ca = Math.cos(this.azimuth), sa = Math.sin(this.azimuth);
    this.camera.position.set(
      this.radius * ce * sa,
      this.radius * se,
      this.radius * ce * ca,
    );
    this.camera.lookAt(0, 0, 0);
  }

  private attachInput() {
    const el = this.renderer.domElement;
    el.style.touchAction = "none";
    el.addEventListener("pointerdown", (e) => {
      this.isDragging = true;
      this.didDrag = false;
      this.downAt = { x: e.clientX, y: e.clientY };
      this.prev = { x: e.clientX, y: e.clientY };
      el.setPointerCapture(e.pointerId);
    });
    el.addEventListener("pointerup", (e) => {
      this.isDragging = false;
      el.releasePointerCapture(e.pointerId);
      // Treat as click if pointer barely moved — pick neuron under cursor.
      if (!this.didDrag && this.pickListener) {
        const idx = this.pickAt(e.clientX, e.clientY);
        if (idx >= 0) this.pickListener(idx, { x: e.clientX, y: e.clientY });
      }
    });
    el.addEventListener("pointermove", (e) => {
      if (!this.isDragging) return;
      const dx = e.clientX - this.prev.x;
      const dy = e.clientY - this.prev.y;
      this.prev = { x: e.clientX, y: e.clientY };
      const totalDx = e.clientX - this.downAt.x;
      const totalDy = e.clientY - this.downAt.y;
      if (totalDx * totalDx + totalDy * totalDy > 9) this.didDrag = true;
      this.azimuth -= dx * 0.005;
      this.elevation = Math.max(-1.5, Math.min(1.5, this.elevation + dy * 0.005));
      this.updateCameraFromOrbit();
    });
    el.addEventListener("wheel", (e) => {
      e.preventDefault();
      this.radius *= Math.exp(e.deltaY * 0.001);
      this.radius = Math.max(50_000, Math.min(5_000_000, this.radius));
      this.updateCameraFromOrbit();
    }, { passive: false });
  }

  /** Pick the nearest neuron to a screen-space click. Returns -1 if none nearby. */
  private pickAt(clientX: number, clientY: number): number {
    const rect = this.renderer.domElement.getBoundingClientRect();
    const ndc = new THREE.Vector2(
      ((clientX - rect.left) / rect.width) * 2 - 1,
      -((clientY - rect.top) / rect.height) * 2 + 1,
    );
    this.raycaster.setFromCamera(ndc, this.camera);
    // Match raycaster threshold to point pixel size — coarse but works.
    this.raycaster.params.Points = { threshold: this.radius * 0.008 };
    const hits = this.raycaster.intersectObject(this.points, false);
    return hits.length ? (hits[0].index ?? -1) : -1;
  }

  /** Register a callback for click-pick events. */
  onPick(cb: ((idx: number, screen: { x: number; y: number }) => void) | null) {
    this.pickListener = cb;
  }

  /** Drop a small marker sphere on the picked neuron. */
  highlightNeuron(idx: number) {
    if (idx < 0 || idx >= this.brain.header.numNeurons) return;
    const px = (this.points.geometry.getAttribute("position") as THREE.BufferAttribute).array as Float32Array;
    if (!this.highlightDot) {
      const geo = new THREE.SphereGeometry(2500, 16, 12);
      const mat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.9 });
      this.highlightDot = new THREE.Mesh(geo, mat);
      this.scene.add(this.highlightDot);
    }
    this.highlightDot.position.set(px[3 * idx], px[3 * idx + 1], px[3 * idx + 2]);
    this.highlightDot.visible = true;
  }
  clearHighlight() { if (this.highlightDot) this.highlightDot.visible = false; }

  /** Capture canvas to a webm clip. Returns a stop() that resolves with the blob. */
  startRecording(fps = 30): () => Promise<Blob> {
    if (this.mediaRecorder) throw new Error("already recording");
    const stream = (this.renderer.domElement as HTMLCanvasElement).captureStream(fps);
    const mime = MediaRecorder.isTypeSupported("video/webm;codecs=vp9")
      ? "video/webm;codecs=vp9"
      : "video/webm";
    const rec = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: 8_000_000 });
    this.recordChunks = [];
    rec.ondataavailable = (e) => { if (e.data.size > 0) this.recordChunks.push(e.data); };
    rec.start(250);
    this.mediaRecorder = rec;
    return () => new Promise<Blob>((resolve) => {
      rec.onstop = () => {
        const blob = new Blob(this.recordChunks, { type: mime });
        this.mediaRecorder = null;
        resolve(blob);
      };
      rec.stop();
    });
  }
  get isRecording() { return this.mediaRecorder !== null; }

  private onResize(container: HTMLElement) {
    const w = container.clientWidth;
    const h = container.clientHeight;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
  }

  private startRenderLoop() {
    const tick = () => {
      this.rafId = requestAnimationFrame(tick);
      if (this.autoplay && this.snapshots.length > 0) {
        this.currentIdx = (this.currentIdx + 1) % this.snapshots.length;
        this.applySnapshot(this.currentIdx);
      }
      this.renderer.render(this.scene, this.camera);
    };
    tick();
  }

  /** Wipe captured snapshots and reset to anatomy baseline. */
  clearSnapshots() {
    this.snapshots.length = 0;
    this.currentIdx = 0;
    this.autoplay = false;
    const colors = this.colorAttr.array as Float32Array;
    colors.set(this.baseColors);
    this.colorAttr.needsUpdate = true;
  }

  /** Add one captured snapshot (per-neuron spike rate in [0, 1]). */
  pushSnapshot(rate: Float32Array) {
    if (rate.length !== this.brain.header.numNeurons) {
      throw new Error(`snapshot length ${rate.length} != neurons ${this.brain.header.numNeurons}`);
    }
    this.snapshots.push(rate);
    if (this.snapshots.length === 1) this.applySnapshot(0);
  }

  /** Render snapshot at index. */
  applySnapshot(idx: number) {
    if (idx < 0 || idx >= this.snapshots.length) return;
    this.currentIdx = idx;
    const r = this.snapshots[idx];
    const colors = this.colorAttr.array as Float32Array;
    // Magma-ish 3-stop ramp with sqrt-amplified rate. Inactive neurons sit
    // near black (additive blending → invisible); active ones blaze hot.
    //   0.0 → (0.015, 0.015, 0.025)  near-black ghost
    //   0.5 → (0.55, 0.10, 0.40)     magenta
    //   1.0 → (1.00, 0.85, 0.50)     bright yellow-orange
    for (let i = 0; i < r.length; i++) {
      const t = Math.min(1, Math.sqrt(r[i] * 20));
      // Lerp anatomy baseline → magma so quiescent neurons keep their class
      // tint and active ones blaze hot.
      let mr: number, mg: number, mb: number;
      if (t < 0.5) {
        const u = t * 2;
        mr = 0.015 + u * (0.55 - 0.015);
        mg = 0.015 + u * (0.10 - 0.015);
        mb = 0.025 + u * (0.40 - 0.025);
      } else {
        const u = (t - 0.5) * 2;
        mr = 0.55 + u * (1.00 - 0.55);
        mg = 0.10 + u * (0.85 - 0.10);
        mb = 0.40 + u * (0.50 - 0.40);
      }
      const br = this.baseColors[3 * i];
      const bg = this.baseColors[3 * i + 1];
      const bb = this.baseColors[3 * i + 2];
      colors[3 * i]     = br + t * (mr - br);
      colors[3 * i + 1] = bg + t * (mg - bg);
      colors[3 * i + 2] = bb + t * (mb - bb);
    }
    this.colorAttr.needsUpdate = true;
  }

  setAutoplay(on: boolean) { this.autoplay = on; }
  get numSnapshots() { return this.snapshots.length; }
  getSnapshot(idx: number): Float32Array | undefined { return this.snapshots[idx]; }
  getSnapshots(): readonly Float32Array[] { return this.snapshots; }
  get current() { return this.currentIdx; }

  dispose() {
    cancelAnimationFrame(this.rafId);
    this.renderer.dispose();
  }
}
