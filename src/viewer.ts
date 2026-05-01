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

export class FlyViewer {
  readonly scene = new THREE.Scene();
  readonly camera: THREE.PerspectiveCamera;
  readonly renderer: THREE.WebGLRenderer;

  private points: THREE.Points;
  private colorAttr: THREE.BufferAttribute;
  private snapshots: Float32Array[] = [];
  private currentIdx = 0;
  private brain: Brain;
  private autoplay = false;
  private rafId = 0;

  // simple orbit state — drag to rotate, wheel to zoom
  private isDragging = false;
  private prev = { x: 0, y: 0 };
  private azimuth = 0;
  private elevation = 0.3;
  private radius = 800_000; // FAFB14 brain ~700k nm wide

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

    const colors = new Float32Array(N * 3);
    // very dim baseline — inactive neurons blend with the bg under additive
    for (let i = 0; i < N; i++) {
      colors[3 * i] = 0.015;
      colors[3 * i + 1] = 0.015;
      colors[3 * i + 2] = 0.025;
    }

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
      this.prev = { x: e.clientX, y: e.clientY };
      el.setPointerCapture(e.pointerId);
    });
    el.addEventListener("pointerup", (e) => {
      this.isDragging = false;
      el.releasePointerCapture(e.pointerId);
    });
    el.addEventListener("pointermove", (e) => {
      if (!this.isDragging) return;
      const dx = e.clientX - this.prev.x;
      const dy = e.clientY - this.prev.y;
      this.prev = { x: e.clientX, y: e.clientY };
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

  /** Wipe captured snapshots and reset to baseline colors. */
  clearSnapshots() {
    this.snapshots.length = 0;
    this.currentIdx = 0;
    this.autoplay = false;
    const colors = this.colorAttr.array as Float32Array;
    for (let i = 0; i < colors.length / 3; i++) {
      colors[3 * i]     = 0.015;
      colors[3 * i + 1] = 0.015;
      colors[3 * i + 2] = 0.025;
    }
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
      let cr: number, cg: number, cb: number;
      if (t < 0.5) {
        const u = t * 2;
        cr = 0.015 + u * (0.55 - 0.015);
        cg = 0.015 + u * (0.10 - 0.015);
        cb = 0.025 + u * (0.40 - 0.025);
      } else {
        const u = (t - 0.5) * 2;
        cr = 0.55 + u * (1.00 - 0.55);
        cg = 0.10 + u * (0.85 - 0.10);
        cb = 0.40 + u * (0.50 - 0.40);
      }
      colors[3 * i]     = cr;
      colors[3 * i + 1] = cg;
      colors[3 * i + 2] = cb;
    }
    this.colorAttr.needsUpdate = true;
  }

  setAutoplay(on: boolean) { this.autoplay = on; }
  get numSnapshots() { return this.snapshots.length; }
  get current() { return this.currentIdx; }

  dispose() {
    cancelAnimationFrame(this.rafId);
    this.renderer.dispose();
  }
}
