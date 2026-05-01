// room.ts — Three.js arena with the real TuragaLab flybody, posed each
// frame by MuJoCo. Body geometry comes from the vendored OBJ meshes;
// kinematics and contacts come from physics.ts.
//
// Coordinate map: MuJoCo (x, y, z, z-up, cm) → Three.js (x, z, -y, y-up,
// scaled by VISUAL_SCALE). The basis change has det=+1 so quaternions
// pass through under the same axis swap.

import * as THREE from "three";
import { OBJLoader } from "three/examples/jsm/loaders/OBJLoader.js";
import { Physics, type BodyInfo, type FlybodyPose } from "./physics";

export interface RoomOpts {
  container: HTMLElement;
  bg?: number;
}

const FLOOR_SIZE = 12;            // shrunk to keep fly visible
const ROOM_HEIGHT = 6;
const VISUAL_SCALE = 10;          // MJ cm × 10 → TJ units; fly ≈ 30 TJ long

export class Room {
  readonly scene = new THREE.Scene();
  readonly camera: THREE.PerspectiveCamera;
  readonly renderer: THREE.WebGLRenderer;

  private bodyGroups: THREE.Object3D[] = [];
  private rootGroup = new THREE.Group();   // scaled root for the fly tree
  private physics: Physics | null = null;
  private rafId = 0;

  // orbit state
  private isDragging = false;
  private prev = { x: 0, y: 0 };
  private azimuth = 0.5;
  private elevation = 0.55;
  private radius = 28;

  // commanded velocity (still smoothed in main, kept here for API parity).
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

    this.camera = new THREE.PerspectiveCamera(40, w / h, 0.05, 500);
    this.updateCameraFromOrbit();

    this.scene.add(new THREE.AmbientLight(0x4a5a6e, 0.6));
    const key = new THREE.DirectionalLight(0xffd9a0, 1.1);
    key.position.set(8, 14, 6);
    this.scene.add(key);
    const fill = new THREE.DirectionalLight(0x6da0ff, 0.4);
    fill.position.set(-6, 4, -8);
    this.scene.add(fill);

    const grid = new THREE.GridHelper(FLOOR_SIZE, 24, 0x2a3340, 0x1a2028);
    (grid.material as THREE.Material).transparent = true;
    (grid.material as THREE.Material).opacity = 0.55;
    this.scene.add(grid);
    const floor = new THREE.Mesh(
      new THREE.CircleGeometry(FLOOR_SIZE / 2, 64),
      new THREE.MeshBasicMaterial({ color: 0x10151c, transparent: true, opacity: 0.6 }),
    );
    floor.rotation.x = -Math.PI / 2;
    floor.position.y = -0.005;
    this.scene.add(floor);

    this.scene.add(this.rootGroup);

    this.attachInput();
    this.startLoop();
    window.addEventListener("resize", () => this.onResize(container));
  }

  /** Attach physics + build the fly's Three.js scene graph from its bodies. */
  async attachPhysics(physics: Physics) {
    this.physics = physics;
    await this.buildBodyGraph(physics.bodies);
  }

  setDrive(forward: number, turn: number) {
    this.forward = forward;
    this.turn = turn;
  }

  resetFly() {
    this.physics?.reset();
  }

  private async buildBodyGraph(bodies: BodyInfo[]) {
    const objLoader = new OBJLoader();
    // Cache mesh geometry so multiple bodies sharing a mesh share the geometry.
    const meshCache = new Map<string, Promise<THREE.Group>>();
    const fetchMesh = (file: string) => {
      let p = meshCache.get(file);
      if (!p) {
        p = fetch(`/flybody/${file}`)
          .then((r) => r.text())
          .then((txt) => objLoader.parse(txt));
        meshCache.set(file, p);
      }
      return p;
    };

    // Default material — warm fly-amber, lit.
    const bodyMat = new THREE.MeshStandardMaterial({
      color: 0xa07030,
      roughness: 0.55,
      metalness: 0.15,
      emissive: 0x201005,
      emissiveIntensity: 0.4,
    });

    // Pre-build all body Object3Ds so xpos/xquat can address them by id.
    this.bodyGroups = bodies.map(() => new THREE.Object3D());
    // Body 0 is worldbody; everything else parents to either world or to
    // its MuJoCo parent. We'll place all bodies as direct children of the
    // root scaled group and *write the world transform* each frame, since
    // MuJoCo gives us xpos/xquat in world frame already. That sidesteps
    // having to rebuild the parent hierarchy in three.js.
    for (let i = 1; i < this.bodyGroups.length; i++) {
      this.rootGroup.add(this.bodyGroups[i]);
    }
    // Apply visual scaling at the root so we don't scale per-body.
    this.rootGroup.scale.setScalar(VISUAL_SCALE);

    // Attach meshes asynchronously.
    let loaded = 0, failed = 0;
    await Promise.all(bodies.map(async (b) => {
      if (b.id === 0) return; // worldbody has no visual
      for (const file of b.meshFiles) {
        try {
          const src = await fetchMesh(file);
          // OBJLoader returns a Group; clone children into our body
          // Object3D so each body owns its own mesh transform.
          src.traverse((obj) => {
            if ((obj as THREE.Mesh).isMesh) {
              const m = obj as THREE.Mesh;
              const clone = new THREE.Mesh(m.geometry, bodyMat);
              // flybody's <default> applies scale="0.1 0.1 0.1" to meshes.
              clone.scale.setScalar(0.1);
              this.bodyGroups[b.id].add(clone);
            }
          });
          loaded++;
        } catch {
          failed++;
        }
      }
    }));
    console.log(`flybody meshes loaded: ${loaded} ok, ${failed} failed`);
  }

  private updateCameraFromOrbit() {
    const tx = this.bodyGroups[1] ? this.bodyGroups[1].position.x * VISUAL_SCALE : 0;
    const tz = this.bodyGroups[1] ? this.bodyGroups[1].position.z * VISUAL_SCALE : 0;
    const ce = Math.cos(this.elevation), se = Math.sin(this.elevation);
    const ca = Math.cos(this.azimuth), sa = Math.sin(this.azimuth);
    this.camera.position.set(
      tx + this.radius * ce * sa,
      this.radius * se + 1,
      tz + this.radius * ce * ca,
    );
    this.camera.lookAt(tx, 1, tz);
  }

  /** No-op for now; eyes were on the procedural fly. */
  setEyeGlow(_intensity: number) { /* TODO: re-attach when we identify head body. */ }

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
      this.radius = Math.max(2, Math.min(120, this.radius));
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
    const tick = () => {
      this.rafId = requestAnimationFrame(tick);
      if (this.physics && this.bodyGroups.length > 1) {
        const pose = this.physics.step();
        this.applyPose(pose);
      }
      this.updateCameraFromOrbit();
      this.renderer.render(this.scene, this.camera);
    };
    requestAnimationFrame(tick);
  }

  /** Map MuJoCo body poses (z-up cm) into the three.js scene (y-up). */
  private applyPose(pose: FlybodyPose) {
    const { xpos, xquat } = pose;
    for (let i = 1; i < this.bodyGroups.length; i++) {
      const obj = this.bodyGroups[i];
      // pos: (x, y, z) → (x, z, -y)
      obj.position.set(xpos[3 * i + 0], xpos[3 * i + 2], -xpos[3 * i + 1]);
      // quat: MJ (w, x, y, z) → TJ axis-swap → (w, x, z, -y) → three (x, y, z, w)
      const qw = xquat[4 * i + 0];
      const qx = xquat[4 * i + 1];
      const qy = xquat[4 * i + 2];
      const qz = xquat[4 * i + 3];
      obj.quaternion.set(qx, qz, -qy, qw);
    }
  }

  dispose() {
    cancelAnimationFrame(this.rafId);
    this.renderer.dispose();
  }
}
