// room.ts — Three.js scene driven by MuJoCo via the canonical
// "zalo/mujoco_wasm" pattern (also used by RenaissanceTek/fly-sim, which
// puts flybody specifically in the browser). The key shift from earlier
// attempts: pull mesh vertex / face / normal / uv buffers DIRECTLY from
// `model.mesh_*` (already-processed by MuJoCo, so geometry is guaranteed
// to align with body kinematics) and keep one THREE.Group per MuJoCo
// body that we update each frame from data.xpos / data.xquat.
//
// Coordinate handling: MuJoCo is z-up, three.js y-up. We swizzle every
// vertex (and every position / quaternion fed to body groups) at the
// data layer, so no parent rotation is needed. zalo's getPosition /
// getQuaternion equivalents are inlined as swizzlePos / swizzleQuat.

import * as THREE from "three";
import { Physics } from "./physics";

export interface RoomOpts {
  container: HTMLElement;
  bg?: number;
}

const VISUAL_SCALE = 6;          // MJ cm × 6 → TJ units; ~3 cm fly → ~18 TJ
const FLOOR_TILE = 12;           // visual grid helper extent

/** target.set( x, z, -y ) — converts MJ z-up to TJ y-up. */
function swizzlePos(buf: Float32Array | Float64Array, idx: number, target: THREE.Vector3) {
  return target.set(
     buf[idx * 3 + 0],
     buf[idx * 3 + 2],
    -buf[idx * 3 + 1],
  );
}

/** Quaternion swizzle for the (x, z, -y) axis swap. MuJoCo stores
 *  (w, x, y, z); three.js wants (x, y, z, w). */
function swizzleQuat(buf: Float32Array | Float64Array, idx: number, target: THREE.Quaternion) {
  return target.set(
    -buf[idx * 4 + 1],
    -buf[idx * 4 + 3],
     buf[idx * 4 + 2],
    -buf[idx * 4 + 0],
  );
}

export class Room {
  readonly scene = new THREE.Scene();
  readonly camera: THREE.PerspectiveCamera;
  readonly renderer: THREE.WebGLRenderer;

  private mjRoot = new THREE.Group();   // VISUAL_SCALE wrapper
  private bodies: THREE.Group[] = [];    // index = MuJoCo body id
  private physics: Physics | null = null;
  private rafId = 0;

  // orbit
  private isDragging = false;
  private prev = { x: 0, y: 0 };
  private azimuth = 0.5;
  private elevation = 0.4;
  private radius = 12;

  // drive (Phase 2.1 will use this for actuator control)
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

    this.scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    const key = new THREE.DirectionalLight(0xffd9a0, 1.1);
    key.position.set(8, 14, 6);
    this.scene.add(key);
    const fill = new THREE.DirectionalLight(0x6da0ff, 0.4);
    fill.position.set(-6, 4, -8);
    this.scene.add(fill);

    const grid = new THREE.GridHelper(FLOOR_TILE, 24, 0x2a3340, 0x1a2028);
    (grid.material as THREE.Material).transparent = true;
    (grid.material as THREE.Material).opacity = 0.5;
    this.scene.add(grid);

    this.mjRoot.scale.setScalar(VISUAL_SCALE);
    this.scene.add(this.mjRoot);

    this.updateCameraFromOrbit();
    this.attachInput();
    this.startLoop();
    window.addEventListener("resize", () => this.onResize(container));
  }

  async attachPhysics(physics: Physics) {
    this.physics = physics;
    this.buildBodyGraphFromMujoco(physics);
  }

  setDrive(forward: number, turn: number) { this.forward = forward; this.turn = turn; }
  resetFly() { this.physics?.reset(); }
  setEyeGlow(_: number) { /* TODO: locate head body, modulate emissive. */ }

  // --- scene-graph build (zalo pattern) -------------------------------------
  private buildBodyGraphFromMujoco(phys: Physics) {
    const m = phys.model as any;
    const ngeom = m.ngeom as number;
    const nbody = m.nbody as number;
    const T = phys.mujoco.mjtGeom;

    // mesh-id → cached BufferGeometry, built from MuJoCo's processed buffers.
    const meshGeos = new Map<number, THREE.BufferGeometry>();

    // Pre-create body groups so geom parenting can use any id.
    this.bodies = [];
    for (let b = 0; b < nbody; b++) {
      const g = new THREE.Group();
      g.name = `body_${b}`;
      this.bodies.push(g);
    }

    // Flat body graph: all bodies attach DIRECTLY to mjRoot (worldbody).
    // We deliberately do NOT use model.body_parentid here. data.xpos /
    // data.xquat are already in world frame; if we built the real
    // parent tree, three.js would compose each child with all its
    // ancestors and the fly would explode outward (which is what
    // happened on every previous attempt). zalo/mujoco_wasm flattens
    // the graph for the same reason — see mujocoUtils.js line 571-582.
    for (let b = 0; b < nbody; b++) this.mjRoot.add(this.bodies[b]);

    // Walk every geom; build geometry, set local pos/quat (in body frame),
    // attach to body group.
    const geomGroup = m.geom_group as Int32Array;
    const geomBodyId = m.geom_bodyid as Int32Array;
    const geomType = m.geom_type as Int32Array;
    const geomDataId = m.geom_dataid as Int32Array;
    const geomSize = m.geom_size as Float32Array;
    const geomRgba = m.geom_rgba as Float32Array;
    const geomPos = m.geom_pos as Float32Array;
    const geomQuat = m.geom_quat as Float32Array;

    let visibleGeoms = 0;
    for (let g = 0; g < ngeom; g++) {
      // group<3 matches MuJoCo's `simulate` default — skip collision (4)
      // and helpers (5+).
      if (geomGroup[g] >= 3) continue;

      const type = geomType[g];
      const sx = geomSize[g * 3], sy = geomSize[g * 3 + 1], sz = geomSize[g * 3 + 2];
      let geometry: THREE.BufferGeometry;

      if (type === T.mjGEOM_PLANE.value) {
        geometry = new THREE.PlaneGeometry(2 * (sx || 100), 2 * (sy || 100));
        geometry.rotateX(-Math.PI / 2);
      } else if (type === T.mjGEOM_SPHERE.value) {
        geometry = new THREE.SphereGeometry(sx, 16, 12);
      } else if (type === T.mjGEOM_CAPSULE.value) {
        geometry = new THREE.CapsuleGeometry(sx, sy * 2, 4, 12);
      } else if (type === T.mjGEOM_CYLINDER.value) {
        geometry = new THREE.CylinderGeometry(sx, sx, sy * 2, 16);
      } else if (type === T.mjGEOM_BOX.value) {
        geometry = new THREE.BoxGeometry(2 * sx, 2 * sz, 2 * sy);
      } else if (type === T.mjGEOM_ELLIPSOID.value) {
        geometry = new THREE.SphereGeometry(1, 16, 12);
        geometry.scale(sx, sy, sz);
      } else if (type === T.mjGEOM_MESH.value) {
        const meshId = geomDataId[g];
        let cached = meshGeos.get(meshId);
        if (!cached) cached = this.buildMeshGeometry(phys, meshId);
        geometry = cached;
        meshGeos.set(meshId, cached);
      } else {
        continue; // unsupported types (hfield, sdf) — silently skip
      }

      const r = geomRgba[g * 4], gC = geomRgba[g * 4 + 1], bC = geomRgba[g * 4 + 2], a = geomRgba[g * 4 + 3];
      const material = new THREE.MeshStandardMaterial({
        color: new THREE.Color(r, gC, bC),
        transparent: a < 1.0,
        opacity: a,
        roughness: 0.55,
        metalness: 0.1,
      });

      const mesh = new THREE.Mesh(geometry, material);
      const v = new THREE.Vector3();
      const q = new THREE.Quaternion();
      swizzlePos(geomPos, g, v);
      swizzleQuat(geomQuat, g, q);
      mesh.position.copy(v);
      mesh.quaternion.copy(q);
      this.bodies[geomBodyId[g]].add(mesh);
      visibleGeoms++;
    }
    console.log(`flybody scene built: ${nbody} bodies, ${visibleGeoms} visible geoms, ${meshGeos.size} unique meshes`);
  }

  /** Replicates zalo/mujoco_wasm's mesh-from-MuJoCo path. Pulls vertex /
   * face / normal / uv buffers straight from the model so we never have
   * to re-parse OBJ on the JS side. Includes the y↔z, negate-y per-vertex
   * swizzle so the result lives in three.js y-up space without parent
   * rotation. */
  private buildMeshGeometry(phys: Physics, meshId: number): THREE.BufferGeometry {
    const m = phys.model as any;
    const mesh_vert = m.mesh_vert as Float32Array;
    const mesh_normal = m.mesh_normal as Float32Array;
    const mesh_texcoord = m.mesh_texcoord as Float32Array;
    const mesh_face = m.mesh_face as Int32Array;
    const mesh_facetexcoord = m.mesh_facetexcoord as Int32Array;
    const mesh_facenormal = m.mesh_facenormal as Int32Array;
    const mesh_vertadr = m.mesh_vertadr as Int32Array;
    const mesh_vertnum = m.mesh_vertnum as Int32Array;
    const mesh_normaladr = m.mesh_normaladr as Int32Array;
    const mesh_normalnum = m.mesh_normalnum as Int32Array;
    const mesh_texcoordadr = m.mesh_texcoordadr as Int32Array;
    const mesh_faceadr = m.mesh_faceadr as Int32Array;
    const mesh_facenum = m.mesh_facenum as Int32Array;

    const vAdr = mesh_vertadr[meshId], vNum = mesh_vertnum[meshId];
    const vert = new Float32Array(mesh_vert.buffer, mesh_vert.byteOffset + vAdr * 3 * 4, vNum * 3).slice();
    // Per-vertex swizzle: (x, y, z) → (x, z, -y). Bake into the buffer.
    for (let v = 0; v < vert.length; v += 3) {
      const ty = vert[v + 1];
      vert[v + 1] = vert[v + 2];
      vert[v + 2] = -ty;
    }

    const nAdr = mesh_normaladr[meshId], nNum = mesh_normalnum[meshId];
    const norm = new Float32Array(mesh_normal.buffer, mesh_normal.byteOffset + nAdr * 3 * 4, nNum * 3).slice();
    for (let v = 0; v < norm.length; v += 3) {
      const ty = norm[v + 1];
      norm[v + 1] = norm[v + 2];
      norm[v + 2] = -ty;
    }

    const fAdr = mesh_faceadr[meshId], fNum = mesh_facenum[meshId];
    const face = new Int32Array(
      mesh_face.buffer, mesh_face.byteOffset + fAdr * 3 * 4, fNum * 3,
    );

    // UV is per-vertex of texcoords, but faces reference texcoord
    // indices through facetexcoord. Swizzle into per-mesh-vertex form.
    const tAdr = mesh_texcoordadr[meshId];
    const haveUV = tAdr >= 0 && mesh_texcoord;
    const swUV = new Float32Array(vNum * 2);
    const swNormal = new Float32Array(vNum * 3);
    if (haveUV || true) {
      const fTexAdr = fAdr * 3;
      const fNormAdr = fAdr * 3;
      for (let t = 0; t < fNum; t++) {
        for (let k = 0; k < 3; k++) {
          const vi = face[t * 3 + k];
          if (haveUV && mesh_facetexcoord) {
            const uvi = mesh_facetexcoord[fTexAdr + t * 3 + k];
            if (uvi >= 0) {
              swUV[vi * 2 + 0] = mesh_texcoord[(tAdr + uvi) * 2 + 0];
              swUV[vi * 2 + 1] = mesh_texcoord[(tAdr + uvi) * 2 + 1];
            }
          }
          if (mesh_facenormal) {
            const ni = mesh_facenormal[fNormAdr + t * 3 + k];
            if (ni >= 0) {
              swNormal[vi * 3 + 0] = norm[ni * 3 + 0];
              swNormal[vi * 3 + 1] = norm[ni * 3 + 1];
              swNormal[vi * 3 + 2] = norm[ni * 3 + 2];
            }
          }
        }
      }
    }

    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.BufferAttribute(vert, 3));
    geom.setAttribute("normal", new THREE.BufferAttribute(swNormal.length === vNum * 3 && nNum ? swNormal : norm.subarray(0, vNum * 3), 3));
    if (haveUV) geom.setAttribute("uv", new THREE.BufferAttribute(swUV, 2));
    geom.setIndex(Array.from(face));
    geom.computeVertexNormals();   // MuJoCo normals can be off; recompute matches zalo.
    return geom;
  }

  // --- per-frame body transform update --------------------------------------
  private syncBodyTransforms() {
    const phys = this.physics!;
    const xpos = phys.data.xpos as Float64Array;
    const xquat = phys.data.xquat as Float64Array;
    const v = new THREE.Vector3();
    const q = new THREE.Quaternion();
    for (let b = 1; b < this.bodies.length; b++) {
      swizzlePos(xpos, b, v);
      swizzleQuat(xquat, b, q);
      this.bodies[b].position.copy(v);
      this.bodies[b].quaternion.copy(q);
    }
  }

  // --- camera / input -------------------------------------------------------
  private updateCameraFromOrbit() {
    let lookY = 1;
    if (this.physics) {
      const xpos = this.physics.data.xpos as Float64Array;
      // Body 1 is thorax. MJ z (height) → TJ y after swizzle.
      lookY = (xpos[3 * 1 + 2] || 0) * VISUAL_SCALE;
    }
    const ce = Math.cos(this.elevation), se = Math.sin(this.elevation);
    const ca = Math.cos(this.azimuth), sa = Math.sin(this.azimuth);
    this.camera.position.set(
      this.radius * ce * sa,
      lookY + this.radius * se,
      this.radius * ce * ca,
    );
    this.camera.lookAt(0, lookY, 0);
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
      this.radius = Math.max(1, Math.min(120, this.radius));
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
      if (this.physics && this.bodies.length) {
        // Demo drive: amplitude = total commanded drive (forward + |turn|)
        // from DN activity. Brain spikes → DN fires → wings buzz.
        const buzz = Math.min(1, Math.abs(this.forward) + Math.abs(this.turn));
        this.physics.driveWings(buzz);

        // flybody MJCF runs at dt=0.0001 s; cap sim per render frame
        // to keep frame budget sane. 32 substeps ≈ 3.2 ms simulated
        // per render — enough to see wing flap without melting CPU.
        this.physics.step(32);
        this.syncBodyTransforms();
      }
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
