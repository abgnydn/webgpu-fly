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
const FLOOR_TILE = 60;           // wide enough to fit fly + 3 cm-ahead target

// Retinal sample: tiny offscreen render from the fly's head pose. Real
// flies have ~700 ommatidia per eye over ~270° azimuth; we approximate
// with a 64×16 panoramic strip covering 180° in front. Cheap (~1024
// fragments per frame), runs on the same GL context as the main view.
const RETINA_W = 64;
const RETINA_H = 16;
// 150° hFov is the widest a flat-projection perspective camera can do
// without the projection matrix blowing up. Real flies have ~270°
// panoramic vision, but for a forward-only retina 150° is plenty —
// covers ±75° around the heading vector, more than the fly normally
// uses for visual servo.
const RETINA_FOV_DEG = 150;
export const RETINA_FOV_RAD = (RETINA_FOV_DEG * Math.PI) / 180;

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

  // Closed-loop visual target (a glowing sphere the fly can "see").
  // flybody's head is at body-local +x, so the fly's spawn heading is
  // world +x. Place the target 3 cm in that direction.
  private target: THREE.Mesh | null = null;
  private targetGlow: THREE.PointLight | null = null;
  targetPos: [number, number, number] = [3.0, 0, 0.13];

  // Retinal sample state. Render target + camera positioned each frame
  // at the fly head looking forward; readback into pixels[] gives a real
  // visual signal that the closed loop reads instead of cheating with
  // geometry.
  private retinaRT: THREE.WebGLRenderTarget;
  private retinaCam: THREE.PerspectiveCamera;
  private retinaPixels = new Uint8Array(RETINA_W * RETINA_H * 4);
  /** Called from the render tick after a fresh retina readback —
   * subscribers (e.g. the overlay canvas) can read retinaPixels. */
  onRetinaUpdate: (() => void) | null = null;

  // orbit
  private isDragging = false;
  private prev = { x: 0, y: 0 };
  private azimuth = 0.5;
  private elevation = 0.4;
  private radius = 30;             // wide enough to see fly + target

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

    this.retinaRT = new THREE.WebGLRenderTarget(RETINA_W, RETINA_H, {
      depthBuffer: true,
      stencilBuffer: false,
      magFilter: THREE.NearestFilter,
      minFilter: THREE.NearestFilter,
    });
    // Three.js perspective takes vertical FOV. We want a wide horizontal
    // strip, so derive vFov from the desired hFov and the strip aspect.
    const hFovRad = (RETINA_FOV_DEG * Math.PI) / 180;
    const aspect = RETINA_W / RETINA_H;
    const vFovRad = 2 * Math.atan(Math.tan(hFovRad / 2) / aspect);
    this.retinaCam = new THREE.PerspectiveCamera(
      (vFovRad * 180) / Math.PI, aspect, 0.05, 50,
    );

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

    // Glowing red target — placed at MuJoCo world (3, 0, 0.13) cm,
    // i.e. 3 cm in front of the fly's spawn at fly head height. Sits
    // inside the mjRoot so it shares the y-up rotation + scale.
    // Target ~5 mm radius in MJ frame so it's visible from a 30-unit
    // orbit camera even with VISUAL_SCALE=6.
    const targetGeo = new THREE.SphereGeometry(0.5, 24, 18);
    const targetMat = new THREE.MeshStandardMaterial({
      color: 0xff3a3a, emissive: 0xff1010, emissiveIntensity: 2.5,
      roughness: 0.4, metalness: 0.1,
    });
    this.target = new THREE.Mesh(targetGeo, targetMat);
    this.target.position.set(this.targetPos[0], this.targetPos[2], -this.targetPos[1]);
    this.mjRoot.add(this.target);

    // A point light at the target position so it casts a subtle glow on
    // the surrounding floor — makes it easier to spot with peripheral
    // vision when the fly is far away.
    const targetGlow = new THREE.PointLight(0xff5050, 4, 6, 1.6);
    targetGlow.position.copy(this.target.position);
    this.mjRoot.add(targetGlow);
    this.targetGlow = targetGlow;

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
  bodySpeed(): number { return this.physics?.bodySpeed ?? 0; }
  /** Triggers DNp01 escape-jump: instantaneous upward impulse. */
  jumpImpulse(speed: number) { this.physics?.jumpImpulse(speed); }

  /**
   * Sensor read for closed-loop control. Returns the signed horizontal
   * angle (radians) from the fly's body heading to the visual target,
   * in the MuJoCo world frame. Positive = target on fly's left;
   * negative = target on fly's right. NaN if physics not ready.
   */
  targetAngle(): number {
    const phys = this.physics;
    if (!phys) return NaN;
    const qpos = phys.data.qpos as Float64Array;
    if (!qpos || qpos.length < 7) return NaN;
    // Fly position (freejoint xyz at qpos[0..2])
    const fx = qpos[0], fy = qpos[1];
    // Fly heading: flybody's head is at body-local +x (per the MJCF),
    // so rotate body +x by the freejoint quat for the world heading.
    const qw = qpos[3], qx = qpos[4], qy = qpos[5], qz = qpos[6];
    const hx = 1 - 2 * (qy * qy + qz * qz);
    const hy = 2 * (qx * qy + qw * qz);
    // Vector to target in xy plane
    const tx = this.targetPos[0] - fx, ty = this.targetPos[1] - fy;
    // Angle from heading to target (signed via cross product z-component)
    const dot = hx * tx + hy * ty;
    const cross = hx * ty - hy * tx;
    return Math.atan2(cross, dot);
  }

  /**
   * Real retinal sample: returns the horizontal angle (radians) to the
   * brightest red blob in the fly's field of view, plus its visual area
   * (fraction of pixels above the red threshold). Returns NaN angle if
   * no red is visible. Positive angle = target on fly's left, matching
   * targetAngle()'s convention.
   *
   * No geometry shortcut — this is the readback of a tiny scene render
   * from the fly's head pose. If you put the target behind the fly, the
   * angle goes NaN; if you turn the fly toward the target, the angle
   * approaches 0.
   */
  retinalSample(): { angle: number; area: number } {
    if (!this.physics) return { angle: NaN, area: 0 };
    const px = this.retinaPixels;
    let weightSum = 0;
    let xSum = 0;
    let redPixels = 0;
    const cols = RETINA_W;
    const rows = RETINA_H;
    for (let y = 0; y < rows; y++) {
      for (let x = 0; x < cols; x++) {
        const o = (y * cols + x) * 4;
        const r = px[o], g = px[o + 1], b = px[o + 2];
        // Red-dominant pixel: r >> g, b. The target's emissive red pops
        // way past the dim grid (max grid color ~0x2a3340 = 42, 51, 64).
        // The retina readback is in linear color space (Three.js renders
        // to the RT without sRGB encoding), so 0xff red shows up as
        // byte ~80 instead of 255. Threshold around 50 catches the
        // target even at oblique angles where edges smear.
        if (r > 50 && r > g + b + 10) {
          const w = r;                 // brighter pixels weighted higher
          weightSum += w;
          xSum += w * x;
          redPixels++;
        }
      }
    }
    if (weightSum === 0) return { angle: NaN, area: 0 };
    const cx = xSum / weightSum;       // pixel column of red centroid
    // Map column [0..W-1] to angle [+hFov/2 .. -hFov/2]. Pixel x grows
    // rightward in screen space; left side of fly view is column 0.
    // targetAngle() convention: positive = target on left → so we
    // negate to get "left = positive".
    const norm = (cx + 0.5) / cols * 2 - 1;       // [-1, +1], +1 = right
    const angle = -norm * (RETINA_FOV_DEG * Math.PI / 360);
    return { angle, area: redPixels / (cols * rows) };
  }

  /** Render the scene from the fly's head pose into the retina RT and
   * read pixels back. Called from the main loop once per frame, before
   * any consumer can call retinalSample(). Cheap (~1024 fragments). */
  private refreshRetina() {
    if (!this.physics) return;
    const qpos = this.physics.data.qpos as Float64Array;
    if (!qpos || qpos.length < 7) return;

    // Fly thorax position (MuJoCo cm), then forward-translate to head.
    const fx = qpos[0], fy = qpos[1], fz = qpos[2];
    const qw = qpos[3], qx = qpos[4], qy_ = qpos[5], qz = qpos[6];
    // Body-forward axis (+x rotated by qpos quat). flybody head sits at
    // body-local ~(0.057, 0, -0.003) cm; collapse to forward * 0.06.
    const hx = 1 - 2 * (qy_ * qy_ + qz * qz);
    const hy = 2 * (qx * qy_ + qw * qz);
    const hz = 2 * (qx * qz - qw * qy_);
    const headX = fx + hx * 0.06;
    const headY = fy + hy * 0.06;
    const headZ = fz + hz * 0.06;

    // Convert head pose into TJ coords. Position uses swizzlePos
    // semantics (x, z, -y); the look-at point is one body-forward unit
    // ahead in MJ frame, also swizzled.
    const lookX = headX + hx;
    const lookY = headY + hy;
    const lookZ = headZ + hz;
    this.retinaCam.position.set(headX, headZ, -headY);
    this.retinaCam.up.set(0, 1, 0);            // world up = MJ +z = TJ +y
    this.retinaCam.lookAt(lookX, lookZ, -lookY);

    // mjRoot is scaled VISUAL_SCALE; the retina cam lives in *unscaled*
    // MJ coords because we operate on qpos directly. Render against
    // scene-graph by attaching cam temporarily to mjRoot equivalent —
    // simpler: render scene as-is, but apply mjRoot.scale to camera
    // position so the scene's scaled meshes line up.
    this.retinaCam.position.multiplyScalar(VISUAL_SCALE);
    const lookSc = new THREE.Vector3(lookX, lookZ, -lookY).multiplyScalar(VISUAL_SCALE);
    this.retinaCam.lookAt(lookSc);

    const renderer = this.renderer;
    const prev = renderer.getRenderTarget();
    renderer.setRenderTarget(this.retinaRT);
    renderer.clear();
    renderer.render(this.scene, this.retinaCam);
    renderer.readRenderTargetPixels(
      this.retinaRT, 0, 0, RETINA_W, RETINA_H, this.retinaPixels,
    );
    renderer.setRenderTarget(prev);
  }

  /** Expose the raw retina pixels for visualization (e.g. mini-overlay).
   * Pixels are refreshed from the render-loop tick — subscribe to
   * onRetinaUpdate to be notified, or just read whenever. */
  retinaFrame(): { pixels: Uint8Array; w: number; h: number } {
    return { pixels: this.retinaPixels, w: RETINA_W, h: RETINA_H };
  }

  /** Move the visual target to a new (x, y) in MuJoCo cm. Game mode
   * uses this to place a procedurally-seeded target each round. */
  setTargetPos(x: number, y: number) {
    this.targetPos = [x, y, this.targetPos[2]];
    if (this.target) this.target.position.set(x, this.targetPos[2], -y);
    if (this.targetGlow && this.target) this.targetGlow.position.copy(this.target.position);
  }

  /** Distance to target in MuJoCo cm (xy plane). */
  targetDistance(): number {
    const phys = this.physics;
    if (!phys) return NaN;
    const qpos = phys.data.qpos as Float64Array;
    if (!qpos || qpos.length < 2) return NaN;
    const dx = this.targetPos[0] - qpos[0];
    const dy = this.targetPos[1] - qpos[1];
    return Math.sqrt(dx * dx + dy * dy);
  }
  private eyeGlowWarned = false;
  setEyeGlow(_brightness: number) {
    // Intentional visual-hook stub: eye glow never affects physics/drive.
    void _brightness;
    if (!this.eyeGlowWarned) {
      this.eyeGlowWarned = true;
      console.warn("setEyeGlow: unimplemented — head-body emissive modulation not wired; ignoring brightness");
    }
  }

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
    // Always run: face-normal remap is required even without UVs. UV writes remain guarded below.
    {
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
    // Orbit around the fly's current world position, not origin —
    // otherwise the body walks out of frame as soon as it moves.
    let cx = 0, cy = 1, cz = 0;
    if (this.physics) {
      const xpos = this.physics.data.xpos as Float64Array;
      // Body 1 is thorax. MJ (x, y, z) → TJ (x, z, -y) after swizzle.
      cx = xpos[3 * 1 + 0] * VISUAL_SCALE;
      cy = xpos[3 * 1 + 2] * VISUAL_SCALE;
      cz = -xpos[3 * 1 + 1] * VISUAL_SCALE;
    }
    const ce = Math.cos(this.elevation), se = Math.sin(this.elevation);
    const ca = Math.cos(this.azimuth), sa = Math.sin(this.azimuth);
    this.camera.position.set(
      cx + this.radius * ce * sa,
      cy + this.radius * se,
      cz + this.radius * ce * ca,
    );
    this.camera.lookAt(cx, cy, cz);
  }

  private attachInput() {
    const el = this.renderer.domElement;
    el.style.touchAction = "none";
    // Raycaster + floor plane for click-to-place-target. Right-click
    // or shift+left-click drops the red ball wherever the cursor lands.
    const raycaster = new THREE.Raycaster();
    const ndc = new THREE.Vector2();
    const floor = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);   // world y=0
    const hit = new THREE.Vector3();
    const placeTargetAtScreen = (x: number, y: number) => {
      const r = el.getBoundingClientRect();
      ndc.set(((x - r.left) / r.width) * 2 - 1, -((y - r.top) / r.height) * 2 + 1);
      raycaster.setFromCamera(ndc, this.camera);
      if (!raycaster.ray.intersectPlane(floor, hit)) return;
      // Convert TJ world (x, y, z) → MJ frame: MJ x = TJ x / scale,
      // MJ y = -TJ z / scale, MJ z = TJ y / scale (y stays small, fly height).
      const mjX = hit.x / VISUAL_SCALE;
      const mjY = -hit.z / VISUAL_SCALE;
      this.targetPos = [mjX, mjY, this.targetPos[2]];
      if (this.target) this.target.position.set(mjX, this.targetPos[2], -mjY);
      if (this.targetGlow) this.targetGlow.position.copy(this.target!.position);
    };
    el.addEventListener("contextmenu", (e) => {
      e.preventDefault();
      placeTargetAtScreen(e.clientX, e.clientY);
    });
    el.addEventListener("pointerdown", (e) => {
      if (e.shiftKey || e.button === 2) {
        placeTargetAtScreen(e.clientX, e.clientY);
        return;
      }
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
    let t0 = performance.now();
    const tick = () => {
      this.rafId = requestAnimationFrame(tick);
      // Soft pulse on the target — makes it visible from a distance
      // and conveys "I am the thing the fly should look at."
      if (this.target && this.targetGlow) {
        const t = (performance.now() - t0) / 1000;
        // Pulse stays well above the retina's red detection floor so the
        // closed loop never loses the target during a dim phase.
        const pulse = 1.1 + 0.25 * Math.sin(t * 3.5);
        (this.target.material as THREE.MeshStandardMaterial).emissiveIntensity = 2.5 * pulse;
        this.targetGlow.intensity = 4.5 * pulse;
      }
      if (this.physics && this.bodies.length) {
        // External hook: if main.ts has the trained walker enabled,
        // it builds an observation + writes 59 actions BEFORE the
        // CPG fires. drivePolicyTick now runs its own control loop
        // (multiple policy calls + sub-stepping per render frame) to
        // match native flybody's 500 Hz policy rate. When it returns
        // true, the render-frame's sim budget has already been spent
        // and we skip the trailing step(32).
        const policyTook = (this as any).drivePolicyTick?.() ?? false;
        if (!policyTook) {
          // Brain → VNC stand-in → body. Two motor primitives selected
          // by DN drive: tripod walk gait (amplitude = forward) and
          // wing-buzz hover (amplitude = remaining drive after walking).
          const sign = this.forward < 0 ? -1 : 1;
          const walkAmp = sign * Math.min(1, Math.sqrt(Math.abs(this.forward)));
          this.physics.driveLegs(walkAmp, this.turn);
          const buzz = Math.min(1, Math.abs(this.forward) * 0.5 + Math.abs(this.turn) * 0.3);
          this.physics.driveWings(buzz, this.turn);
          // flybody MJCF runs at dt=0.0001 s; cap sim per render frame
          // to keep frame budget sane. 32 substeps ≈ 3.2 ms simulated
          // per render.
          this.physics.step(32);
        }
        this.syncBodyTransforms();
        this.refreshRetina();
        this.onRetinaUpdate?.();
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
