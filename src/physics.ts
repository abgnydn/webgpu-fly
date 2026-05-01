// physics.ts — loads the TuragaLab flybody MJCF (fruitfly.xml + 85 OBJ
// meshes) into MuJoCo. Rendering uses MuJoCo's own scene-graph baker
// (mjv_updateScene) so per-geom local transforms compose correctly with
// body kinematics — same pattern as Google DeepMind's official
// three.js demo (mujoco/wasm/demo_app/app.ts).
//
// Phase 2 scope: load + render rest pose. Stepping + actuator gait is
// Phase 2.1.

import loadMujoco from "@mujoco/mujoco";
import type {
  MainModule, MjModel, MjData,
  MjVFS, MjvScene, MjvOption, MjvPerturb, MjvCamera,
} from "@mujoco/mujoco";
import { getOrFetch, cacheStats } from "./cache";

export class Physics {
  mujoco!: MainModule;
  model!: MjModel;
  data!: MjData;
  scene!: MjvScene;

  private vfs!: MjVFS;
  private opt!: MjvOption;
  private perturb!: MjvPerturb;
  private cam!: MjvCamera;
  private catBitAll = 0;

  /** mesh id (dataid in MjvGeom for mesh geoms) → OBJ filename. */
  meshFileById: string[] = [];
  /** Raw OBJ bytes keyed by filename — keeps the renderer off the network. */
  meshBytes: Map<string, Uint8Array> | null = null;

  static async create(onProgress?: (msg: string) => void): Promise<Physics> {
    const p = new Physics();
    onProgress?.("loading mujoco_wasm");
    p.mujoco = await loadMujoco();

    // Use flybody's canonical entry point: build_fruitfly/floor.xml does
    //   <include file="fruitfly.xml"/> + floor plane + grid texture +
    //   skybox, exactly what the official Python wrapper composes.
    // We load both into the VFS so MuJoCo's <include> resolves it
    // without any string surgery on our end.
    onProgress?.("fetching floor.xml + fruitfly.xml");
    const [floorText, flyText] = await Promise.all([
      fetch("/flybody/floor.xml").then((r) => r.text()),
      fetch("/flybody/fruitfly.xml").then((r) => r.text()),
    ]);

    // Mesh refs come from fruitfly.xml; floor.xml only has texture refs.
    const meshFiles = Array.from(
      new Set(
        Array.from(flyText.matchAll(/file="([^"]+)"/g), (m) => m[1])
          .filter((f) => f.endsWith(".obj")),
      ),
    );
    const stats = await cacheStats();
    onProgress?.(`fetching ${meshFiles.length} meshes (IDB cached: ${stats.count}, ${(stats.bytes / 1e6).toFixed(0)} MB)`);

    p.vfs = new p.mujoco.MjVFS();
    // Keep raw bytes around so the renderer can parse OBJs without re-fetching.
    p.meshBytes = new Map();
    const CONCURRENCY = 4;
    let inFlight = 0, idx = 0, completed = 0, totalBytes = 0;
    const t0 = performance.now();
    await new Promise<void>((resolve, reject) => {
      const next = () => {
        while (inFlight < CONCURRENCY && idx < meshFiles.length) {
          const file = meshFiles[idx++];
          inFlight++;
          getOrFetch(file, `/flybody/${file}`)
            .then((buf) => {
              const u8 = new Uint8Array(buf);
              p.vfs.addBuffer(file, u8);
              p.meshBytes!.set(file, u8);
              completed++;
              totalBytes += buf.byteLength;
              if (completed % 20 === 0 || completed === meshFiles.length) {
                onProgress?.(`fetched ${completed}/${meshFiles.length} meshes (${(totalBytes / 1e6).toFixed(0)} MB)`);
              }
              inFlight--;
              if (idx >= meshFiles.length && inFlight === 0) resolve();
              else next();
            })
            .catch(reject);
        }
      };
      next();
    });
    onProgress?.(`fetched all meshes in ${((performance.now() - t0) / 1000).toFixed(1)} s`);

    // The compiler resolves `<include file="fruitfly.xml"/>` from the
    // VFS, so we have to register fruitfly.xml there as well.
    p.vfs.addBuffer("fruitfly.xml", new TextEncoder().encode(flyText));

    onProgress?.("compiling MJCF (synchronous; tab may freeze ~5-15s)");
    const tCompile = performance.now();
    p.model = p.mujoco.MjModel.from_xml_string(floorText, p.vfs);
    p.data = new p.mujoco.MjData(p.model);
    onProgress?.(`MJCF compiled in ${((performance.now() - tCompile) / 1000).toFixed(1)} s`);

    // Initialise to flybody's canonical rest pose. From
    // TuragaLab/flybody/flybody/fruitfly/fruitfly.py:
    //   - _SPAWN_POS = (0, 0, 0.1278): thorax xyz on the floor surface.
    //   - Each joint's qpos is set to model.qpos_spring (its natural
    //     rest target) — without this, joints sit at zero and the legs
    //     splay out instead of standing.
    p.mujoco.mj_resetData(p.model, p.data);
    const qpos = p.data.qpos as Float64Array;
    const qposSpring = p.model.qpos_spring as Float64Array;
    if (qposSpring && qposSpring.length === qpos.length) {
      qpos.set(qposSpring);
    }
    if (qpos.length >= 7) {
      qpos[0] = 0; qpos[1] = 0; qpos[2] = 0.1278;          // _SPAWN_POS
      qpos[3] = 1; qpos[4] = 0; qpos[5] = 0; qpos[6] = 0;  // identity quat
    }
    p.mujoco.mj_forward(p.model, p.data);

    // Build mesh id → file map so the renderer can resolve mjvGeom.dataid
    // → OBJ when it encounters mjGEOM_MESH geoms.
    const nmesh = p.model.nmesh as number;
    const xmlMeshFile = new Map<string, string>();
    for (const m of flyText.matchAll(/<mesh\s+name="([^"]+)"\s+file="([^"]+)"\s*\/?>/g)) {
      xmlMeshFile.set(m[1], m[2]);
    }
    for (let i = 0; i < nmesh; i++) {
      const name = p.mujoco.mj_id2name(
        p.model,
        p.mujoco.mjtObj.mjOBJ_MESH.value,
        i,
      );
      p.meshFileById[i] = xmlMeshFile.get(name) ?? "";
    }

    // Set up the visual scene-baker. maxgeom should comfortably exceed
    // the model's geom count (167 for flybody) — extra capacity covers
    // contact-point geoms that mjv_updateScene synthesizes.
    p.opt = new p.mujoco.MjvOption();
    p.perturb = new p.mujoco.MjvPerturb();
    p.cam = new p.mujoco.MjvCamera();
    p.scene = new p.mujoco.MjvScene(p.model, 4096);
    p.catBitAll = p.mujoco.mjtCatBit.mjCAT_ALL.value;
    // mjv_defaultOption initialises geomgroup[0..5]=1 (all visible),
    // sitegroup, flags, etc. — without this, fresh `new MjvOption()`
    // can leave geomgroup zeroed and visual meshes get filtered out.
    p.mujoco.mjv_defaultOption(p.opt);
    // Hide the collision group (group=4) since flybody renders the
    // pretty visual mesh on group=1 separately. Keeps the right pane
    // clean instead of showing scattered collision capsules over the
    // visuals.
    const grp = p.opt.geomgroup as Uint8Array;
    if (grp && grp.length > 4) grp[4] = 0;

    onProgress?.(`flybody ready (${p.model.nbody} bodies, ${nmesh} meshes)`);
    return p;
  }

  /** Repopulate the visual scene from current MjData. Caller iterates scene.geoms. */
  updateScene() {
    this.mujoco.mjv_updateScene(
      this.model, this.data, this.opt, this.perturb,
      this.cam, this.catBitAll, this.scene,
    );
  }

  /** Step the simulation N times. Phase 2.1 will wire actuator drive. */
  step(substeps = 1) {
    for (let s = 0; s < substeps; s++) {
      this.mujoco.mj_step(this.model, this.data);
    }
  }

  reset() {
    this.mujoco.mj_resetData(this.model, this.data);
    const qpos = this.data.qpos as Float64Array;
    const qposSpring = this.model.qpos_spring as Float64Array;
    if (qposSpring && qposSpring.length === qpos.length) {
      qpos.set(qposSpring);
    }
    if (qpos.length >= 7) {
      qpos[0] = 0; qpos[1] = 0; qpos[2] = 0.1278;
      qpos[3] = 1; qpos[4] = 0; qpos[5] = 0; qpos[6] = 0;
    }
    this.mujoco.mj_forward(this.model, this.data);
  }

  dispose() {
    this.scene.delete();
    this.cam.delete();
    this.perturb.delete();
    this.opt.delete();
    this.data.delete();
    this.model.delete();
    this.vfs.delete();
  }

  get bodyCount() { return this.model.nbody as number; }
}
