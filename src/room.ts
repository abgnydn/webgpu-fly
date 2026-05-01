// room.ts — Three.js "3D room" with a procedural fly that walks under
// neural drive. Not biomechanically real (no MuJoCo yet) — it's a visual
// coupling that proves the brain-sim → embodiment chain.
//
// Drive contract: per snapshot, host code calls setDrive(forward, turn).
//   forward in [-1, 1]  — backward to forward speed (units/sec)
//   turn    in [-1, 1]  — left to right yaw rate (rad/sec)

import * as THREE from "three";
import { Physics } from "./physics";

export interface RoomOpts {
  container: HTMLElement;
  bg?: number;
}

const FLOOR_SIZE = 80;
const ROOM_HEIGHT = 18;

export class Room {
  readonly scene = new THREE.Scene();
  readonly camera: THREE.PerspectiveCamera;
  readonly renderer: THREE.WebGLRenderer;

  private fly: Fly;
  private physics: Physics | null = null;
  private rafId = 0;
  private lastT = performance.now();

  // orbit state
  private isDragging = false;
  private prev = { x: 0, y: 0 };
  private azimuth = 0.5;
  private elevation = 0.55;
  private radius = 28;

  // commanded velocity
  private forward = 0;
  private turn = 0;

  constructor(opts: RoomOpts) {
    const { container } = opts;
    const w = container.clientWidth;
    const h = container.clientHeight;

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(w, h);
    this.renderer.setClearColor(opts.bg ?? 0x0a0d12);
    container.appendChild(this.renderer.domElement);

    this.camera = new THREE.PerspectiveCamera(40, w / h, 0.1, 500);
    this.updateCameraFromOrbit();

    // Lights: cool ambient + warm key from above-front
    this.scene.add(new THREE.AmbientLight(0x4a5a6e, 0.6));
    const key = new THREE.DirectionalLight(0xffd9a0, 1.1);
    key.position.set(8, 14, 6);
    this.scene.add(key);
    const fill = new THREE.DirectionalLight(0x6da0ff, 0.4);
    fill.position.set(-6, 4, -8);
    this.scene.add(fill);

    // Floor: large grid + a faint disc to anchor scale
    const grid = new THREE.GridHelper(FLOOR_SIZE, 40, 0x2a3340, 0x1a2028);
    (grid.material as THREE.Material).transparent = true;
    (grid.material as THREE.Material).opacity = 0.65;
    this.scene.add(grid);

    const floor = new THREE.Mesh(
      new THREE.CircleGeometry(FLOOR_SIZE / 2, 64),
      new THREE.MeshBasicMaterial({ color: 0x10151c, transparent: true, opacity: 0.6 }),
    );
    floor.rotation.x = -Math.PI / 2;
    floor.position.y = -0.01;
    this.scene.add(floor);

    // Faint cylinder "walls" so the camera has spatial reference
    const wall = new THREE.Mesh(
      new THREE.CylinderGeometry(FLOOR_SIZE / 2, FLOOR_SIZE / 2, ROOM_HEIGHT, 48, 1, true),
      new THREE.MeshBasicMaterial({
        color: 0x1a2230,
        transparent: true,
        opacity: 0.18,
        side: THREE.BackSide,
      }),
    );
    wall.position.y = ROOM_HEIGHT / 2;
    this.scene.add(wall);

    this.fly = new Fly();
    this.scene.add(this.fly.group);

    this.attachInput();
    this.startLoop();
    window.addEventListener("resize", () => this.onResize(container));
  }

  /** Attach a MuJoCo Physics instance. Once attached, body pose comes from physics; before that, kinematic fallback is used. */
  attachPhysics(physics: Physics) {
    this.physics = physics;
  }

  /** Push commanded velocity. Units: forward = body-lengths/sec; turn = rad/sec. */
  setDrive(forward: number, turn: number) {
    this.forward = forward;
    this.turn = turn;
  }

  /** Reset fly to origin facing +Z, zero velocity. */
  resetFly() {
    this.physics?.reset();
    this.fly.group.position.set(0, 0, 0);
    this.fly.group.rotation.y = 0;
    this.fly.gaitPhase = 0;
    this.forward = 0;
    this.turn = 0;
  }

  private updateCameraFromOrbit() {
    const ce = Math.cos(this.elevation), se = Math.sin(this.elevation);
    const ca = Math.cos(this.azimuth), sa = Math.sin(this.azimuth);
    this.camera.position.set(
      this.radius * ce * sa,
      this.radius * se + 2,
      this.radius * ce * ca,
    );
    const tx = this.fly ? this.fly.group.position.x : 0;
    const tz = this.fly ? this.fly.group.position.z : 0;
    this.camera.lookAt(tx, 1, tz);
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
      this.elevation = Math.max(-0.2, Math.min(1.4, this.elevation + dy * 0.005));
      this.updateCameraFromOrbit();
    });
    el.addEventListener("wheel", (e) => {
      e.preventDefault();
      this.radius *= Math.exp(e.deltaY * 0.001);
      this.radius = Math.max(6, Math.min(80, this.radius));
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

  private startLoop() {
    const tick = (t: number) => {
      this.rafId = requestAnimationFrame(tick);
      const dt = Math.min(0.05, (t - this.lastT) / 1000);
      this.lastT = t;

      const g = this.fly.group;
      let speed: number;
      if (this.physics) {
        // Real-physics path: MuJoCo owns body pose. Map MuJoCo (x, y, yaw,
        // z-up) → Three.js (x, z, -yaw, y-up). The y-up sign flip on yaw
        // keeps the visual rotation chirality consistent.
        const pose = this.physics.step(this.forward, this.turn);
        g.position.x = pose.x;
        g.position.z = -pose.y;
        g.rotation.y = -pose.yaw;
        speed = Math.abs(this.forward * 4.0);
      } else {
        // Fallback (pre-physics-load) — kinematic integration.
        g.rotation.y += this.turn * dt;
        speed = this.forward * 4.0;
        g.position.x += Math.sin(g.rotation.y) * speed * dt;
        g.position.z += Math.cos(g.rotation.y) * speed * dt;
        const r = Math.hypot(g.position.x, g.position.z);
        const lim = FLOOR_SIZE / 2 - 1.5;
        if (r > lim) {
          const k = lim / r;
          g.position.x *= k;
          g.position.z *= k;
        }
        speed = Math.abs(speed);
      }

      this.fly.update(dt, speed);
      this.updateCameraFromOrbit();
      this.renderer.render(this.scene, this.camera);
    };
    requestAnimationFrame(tick);
  }

  dispose() {
    cancelAnimationFrame(this.rafId);
    this.renderer.dispose();
  }
}

// --- Fly: procedural body with tripod-gait legs ----------------------------
class Fly {
  readonly group = new THREE.Group();
  gaitPhase = 0;

  private legAnchors: THREE.Object3D[] = [];
  private legTips: THREE.Mesh[] = [];
  private legLines: THREE.Line[] = [];

  constructor() {
    // Body — abdomen + thorax + head, slight gold glow
    const bodyMat = new THREE.MeshStandardMaterial({
      color: 0x9a6a2e, roughness: 0.35, metalness: 0.4,
      emissive: 0x402010, emissiveIntensity: 0.4,
    });
    const abdo = new THREE.Mesh(new THREE.SphereGeometry(0.7, 24, 18), bodyMat);
    abdo.scale.set(0.8, 0.65, 1.1);
    abdo.position.set(0, 0.7, -0.7);
    this.group.add(abdo);

    const thorax = new THREE.Mesh(new THREE.SphereGeometry(0.55, 24, 18), bodyMat);
    thorax.position.set(0, 0.8, 0.2);
    this.group.add(thorax);

    const head = new THREE.Mesh(
      new THREE.SphereGeometry(0.4, 20, 14),
      new THREE.MeshStandardMaterial({
        color: 0x2a2a2a, roughness: 0.2, metalness: 0.6,
      }),
    );
    head.position.set(0, 0.85, 0.95);
    this.group.add(head);

    // Eyes — big red compound eyes
    const eyeMat = new THREE.MeshStandardMaterial({
      color: 0xff3a3a, emissive: 0x661010, emissiveIntensity: 0.8,
      roughness: 0.3, metalness: 0.2,
    });
    for (const sx of [-1, 1]) {
      const eye = new THREE.Mesh(new THREE.SphereGeometry(0.22, 16, 12), eyeMat);
      eye.position.set(0.32 * sx, 0.95, 1.0);
      this.group.add(eye);
    }

    // Wings — translucent quads
    const wingMat = new THREE.MeshStandardMaterial({
      color: 0xeaf3ff, transparent: true, opacity: 0.18,
      side: THREE.DoubleSide, roughness: 0.1, metalness: 0.0,
    });
    for (const sx of [-1, 1]) {
      const wing = new THREE.Mesh(new THREE.PlaneGeometry(1.6, 0.6), wingMat);
      wing.position.set(0.6 * sx, 1.15, -0.2);
      wing.rotation.set(-0.3, sx * 0.4, sx * -0.2);
      this.group.add(wing);
    }

    // 6 legs: front/middle/rear × left/right
    // Anchors sit on the thorax/abdomen ventrum; tips track a target on the floor.
    const anchors: Array<[number, number, number]> = [
      [-0.4, 0.55,  0.45], [ 0.4, 0.55,  0.45],   // front L/R
      [-0.5, 0.50,  0.05], [ 0.5, 0.50,  0.05],   // middle L/R
      [-0.5, 0.50, -0.45], [ 0.5, 0.50, -0.45],   // rear L/R
    ];
    const lineMat = new THREE.LineBasicMaterial({ color: 0x2a1a08 });
    for (let i = 0; i < anchors.length; i++) {
      const anchor = new THREE.Object3D();
      anchor.position.set(...anchors[i]);
      this.group.add(anchor);
      this.legAnchors.push(anchor);

      const tip = new THREE.Mesh(
        new THREE.SphereGeometry(0.06, 8, 6),
        new THREE.MeshStandardMaterial({ color: 0x1a1209, roughness: 0.5 }),
      );
      this.group.add(tip);
      this.legTips.push(tip);

      const lineGeo = new THREE.BufferGeometry();
      lineGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(6), 3));
      const line = new THREE.Line(lineGeo, lineMat);
      this.group.add(line);
      this.legLines.push(line);
    }
  }

  /** Procedural tripod gait — three-and-three out of phase. dt sec, speed body-units/s. */
  update(dt: number, speed: number) {
    // Phase advances proportional to speed; freeze when essentially still.
    if (speed < 0.05) return;
    const stepHz = 2.0 + speed * 1.2;
    this.gaitPhase = (this.gaitPhase + dt * stepHz * Math.PI * 2) % (Math.PI * 2);

    // Tripod groups: {0, 3, 4} and {1, 2, 5}
    const groupA = new Set([0, 3, 4]);
    const stride = Math.min(1.0, 0.4 + speed * 0.4);

    for (let i = 0; i < this.legAnchors.length; i++) {
      const phaseOffset = groupA.has(i) ? 0 : Math.PI;
      const ph = this.gaitPhase + phaseOffset;
      const lift = Math.max(0, Math.sin(ph)) * 0.18;     // foot up during swing
      const swing = Math.cos(ph) * 0.4 * stride;         // forward/back along z

      const anchor = this.legAnchors[i];
      // Foot target: outward from anchor, on floor, with stride along body z.
      const sx = Math.sign(anchor.position.x);
      const tx = anchor.position.x + sx * 0.55;
      const tz = anchor.position.z + swing;
      const ty = lift;

      const tip = this.legTips[i];
      tip.position.set(tx, ty, tz);

      // Line segment from anchor → tip in local body space.
      const arr = (this.legLines[i].geometry.getAttribute("position") as THREE.BufferAttribute).array as Float32Array;
      arr[0] = anchor.position.x; arr[1] = anchor.position.y; arr[2] = anchor.position.z;
      arr[3] = tx;                arr[4] = ty;               arr[5] = tz;
      (this.legLines[i].geometry.getAttribute("position") as THREE.BufferAttribute).needsUpdate = true;
    }
  }
}
