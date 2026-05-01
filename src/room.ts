// room.ts — Three.js arena that renders the flybody using MuJoCo's
// mjv_updateScene scene graph. Each render frame, MuJoCo bakes
// body-transform × geom-local-transform into a list of mjvGeoms with
// world-frame pos[3] + mat[9]. We pull each mjvGeom into a cached
// THREE.Mesh and write its 4×4 transform from the baked values.
//
// Pattern adapted from Google DeepMind's official mujoco_wasm three.js
// demo (mujoco/wasm/demo_app/app.ts) — the canonical way.

import * as THREE from "three";
import { OBJLoader } from "three/examples/jsm/loaders/OBJLoader.js";
import { Physics } from "./physics";

export interface RoomOpts {
  container: HTMLElement;
  bg?: number;
}

const FLOOR_SIZE = 4;
const VISUAL_SCALE = 25;  // MJ cm × 25 → TJ units; fly thorax ≈ 7.5 TJ

// Backport of CapsuleGeometry — verbatim from the official demo.
class FlybodyCapsuleGeometry extends THREE.BufferGeometry {
  constructor(radius = 1, length = 1, capSegments = 4, radialSegments = 8) {
    const path = new THREE.Path();
    path.absarc(0, -length / 2, radius, Math.PI * 1.5, 0, false);
    path.absarc(0, length / 2, radius, 0, Math.PI * 0.5, false);
    const lathe = new THREE.LatheGeometry(path.getPoints(capSegments), radialSegments);
    super();
    const idx = lathe.getIndex();
    if (idx) this.setIndex(idx);
    this.setAttribute("position", lathe.getAttribute("position"));
    this.setAttribute("normal", lathe.getAttribute("normal"));
    this.setAttribute("uv", lathe.getAttribute("uv"));
  }
}

export class Room {
  readonly scene = new THREE.Scene();
  readonly camera: THREE.PerspectiveCamera;
  readonly renderer: THREE.WebGLRenderer;

  private flyRoot = new THREE.Group();    // z-up → y-up rotation lives here
  private meshes: THREE.Mesh[] = [];       // index aligned with scene.geoms.get(i)
  private geomCache = new Map<string, THREE.BufferGeometry>();
  private meshLoader = new OBJLoader();
  /** Per-mesh-id (mujoco mesh dataid) → loaded BufferGeometry promise. */
  private objCache = new Map<number, Promise<THREE.BufferGeometry | null>>();
  private physics: Physics | null = null;
  private rafId = 0;

  // orbit state
  private isDragging = false;
  private prev = { x: 0, y: 0 };
  private azimuth = 0.5;
  private elevation = 0.4;
  private radius = 6;
  private lookY = 1;     // dynamic lookAt y, follows fly thorax

  // commanded velocity placeholder (will drive actuators in Phase 2.1)
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

    this.scene.add(new THREE.AmbientLight(0x4a5a6e, 0.7));
    const key = new THREE.DirectionalLight(0xffd9a0, 1.2);
    key.position.set(8, 14, 6);
    this.scene.add(key);
    const fill = new THREE.DirectionalLight(0x6da0ff, 0.5);
    fill.position.set(-6, 4, -8);
    this.scene.add(fill);

    const grid = new THREE.GridHelper(FLOOR_SIZE, 16, 0x2a3340, 0x1a2028);
    (grid.material as THREE.Material).transparent = true;
    (grid.material as THREE.Material).opacity = 0.55;
    this.scene.add(grid);

    // flyRoot rotates MuJoCo z-up world into Three.js y-up world, then
    // VISUAL_SCALE makes the cm-scale fly visible at room scale.
    this.flyRoot.rotation.x = -Math.PI / 2;
    this.flyRoot.scale.setScalar(VISUAL_SCALE);
    this.scene.add(this.flyRoot);

    this.updateCameraFromOrbit();
    this.attachInput();
    this.startLoop();
    window.addEventListener("resize", () => this.onResize(container));
  }

  async attachPhysics(physics: Physics) {
    this.physics = physics;
  }

  setDrive(forward: number, turn: number) {
    this.forward = forward;
    this.turn = turn;
  }

  resetFly() {
    this.physics?.reset();
  }

  setEyeGlow(_intensity: number) { /* TODO: identify head body, modulate emissive. */ }

  private updateCameraFromOrbit() {
    const ce = Math.cos(this.elevation), se = Math.sin(this.elevation);
    const ca = Math.cos(this.azimuth), sa = Math.sin(this.azimuth);
    this.camera.position.set(
      this.radius * ce * sa,
      this.lookY + this.radius * se,
      this.radius * ce * ca,
    );
    this.camera.lookAt(0, this.lookY, 0);
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
      this.radius = Math.max(0.5, Math.min(40, this.radius));
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
      if (this.physics) this.syncFromMujoco();
      this.updateCameraFromOrbit();
      this.renderer.render(this.scene, this.camera);
    };
    requestAnimationFrame(tick);
  }

  /** Pull baked scene geoms from MuJoCo and apply transforms to TJ meshes. */
  private syncFromMujoco() {
    const phys = this.physics!;
    phys.updateScene();
    const mujoco = phys.mujoco;
    const geoms = phys.scene.geoms;
    const n = geoms.size();

    for (let i = 0; i < n; i++) {
      const g = geoms.get(i);
      if (!g) continue;

      let mesh: THREE.Mesh | undefined = this.meshes[i];
      if (!mesh) {
        const { geometry, material } = this.makeGeomMesh(g, mujoco, phys);
        mesh = new THREE.Mesh(geometry, material);
        mesh.matrixAutoUpdate = false;
        this.flyRoot.add(mesh);
        this.meshes[i] = mesh;
      }

      // Build the 4x4 from baked pos[3] + mat[9] (row-major 3x3).
      const pos = g.pos as Float32Array | Float64Array;
      const mat = g.mat as Float32Array | Float64Array;
      mesh.matrix.set(
        mat[0], mat[1], mat[2], pos[0],
        mat[3], mat[4], mat[5], pos[1],
        mat[6], mat[7], mat[8], pos[2],
        0, 0, 0, 1,
      );
      mesh.matrixWorldNeedsUpdate = true;

      // Track thorax-ish height for camera. Body 1 == thorax in flybody;
      // its first geom is what we sample. Loose heuristic.
      if (i === 1) {
        // pos[2] is the MuJoCo z (height). flyRoot rotates so MJ z → TJ y;
        // then scale by VISUAL_SCALE to get TJ world y for camera lookY.
        this.lookY = pos[2] * VISUAL_SCALE;
      }

      g.delete();
    }

    // Hide stale meshes if scene shrunk (e.g. fewer contact geoms).
    for (let i = n; i < this.meshes.length; i++) this.meshes[i].visible = false;

    geoms.delete();
  }

  /** Build the right Three.js geometry+material for an MjvGeom. */
  private makeGeomMesh(g: any, mujoco: any, phys: Physics) {
    const type: number = g.type;
    const sz = g.size as Float32Array | Float64Array;
    const rgba = g.rgba as Float32Array;

    const T = mujoco.mjtGeom;
    let geometry: THREE.BufferGeometry;

    if (type === T.mjGEOM_PLANE.value) {
      geometry = new THREE.PlaneGeometry(
        2 * (sz[0] || 200), 2 * (sz[1] || 200),
      );
      geometry.rotateX(-Math.PI / 2);  // PlaneGeometry is xy-up by default
    } else if (type === T.mjGEOM_SPHERE.value) {
      geometry = new THREE.SphereGeometry(sz[0], 16, 12);
    } else if (type === T.mjGEOM_CAPSULE.value) {
      geometry = new FlybodyCapsuleGeometry(sz[0], 2 * sz[2], 8, 12);
      geometry.rotateX(0.5 * Math.PI);
    } else if (type === T.mjGEOM_BOX.value) {
      geometry = new THREE.BoxGeometry(2 * sz[0], 2 * sz[1], 2 * sz[2]);
    } else if (type === T.mjGEOM_CYLINDER.value) {
      geometry = new THREE.CylinderGeometry(sz[0], sz[0], 2 * sz[2], 16);
      geometry.rotateX(0.5 * Math.PI);
    } else if (type === T.mjGEOM_ELLIPSOID.value) {
      geometry = new THREE.SphereGeometry(1, 16, 12);
      geometry.scale(sz[0], sz[1], sz[2]);
    } else if (type === T.mjGEOM_MESH.value) {
      // Async: kick off OBJ load, return a placeholder until it lands.
      const meshId: number = g.dataid;
      geometry = this.placeholderGeometry();
      this.loadMeshObj(meshId, phys).then((loaded) => {
        if (loaded) {
          // Swap geometry once loaded.
          // (mesh array index isn't trivially reachable here; iterate.)
          for (const m of this.meshes) {
            if (m && m.geometry === geometry) m.geometry = loaded;
          }
        }
      });
    } else {
      geometry = new THREE.BufferGeometry();
    }

    const material = new THREE.MeshStandardMaterial({
      color: new THREE.Color(rgba[0], rgba[1], rgba[2]),
      transparent: rgba[3] !== 1,
      opacity: rgba[3],
      roughness: 0.55,
      metalness: 0.1,
    });
    return { geometry, material };
  }

  private placeholderGeometry(): THREE.BufferGeometry {
    let g = this.geomCache.get("__placeholder");
    if (!g) {
      g = new THREE.SphereGeometry(0.001, 4, 3);
      this.geomCache.set("__placeholder", g);
    }
    return g;
  }

  private loadMeshObj(meshId: number, phys: Physics): Promise<THREE.BufferGeometry | null> {
    let p = this.objCache.get(meshId);
    if (p) return p;
    const file = phys.meshFileById[meshId];
    if (!file) return Promise.resolve(null);
    p = fetch(`/flybody/${file}`)
      .then((r) => r.text())
      .then((txt) => {
        const grp = this.meshLoader.parse(txt);
        // Merge all child mesh geometries into one — flybody OBJs are
        // single-object; just grab the first mesh's geometry.
        let merged: THREE.BufferGeometry | null = null;
        grp.traverse((obj) => {
          if (!merged && (obj as THREE.Mesh).isMesh) {
            merged = (obj as THREE.Mesh).geometry as THREE.BufferGeometry;
          }
        });
        if (merged) {
          // flybody's <default> applies scale="0.1 0.1 0.1" to meshes —
          // bake that in here so MuJoCo's mjvGeom transforms are correct.
          (merged as THREE.BufferGeometry).scale(0.1, 0.1, 0.1);
        }
        return merged;
      })
      .catch(() => null);
    this.objCache.set(meshId, p);
    return p;
  }

  dispose() {
    cancelAnimationFrame(this.rafId);
    this.renderer.dispose();
  }
}
