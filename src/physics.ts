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

  static async create(onProgress?: (msg: string) => void): Promise<Physics> {
    const p = new Physics();
    onProgress?.("loading mujoco_wasm");
    p.mujoco = await loadMujoco();

    onProgress?.("fetching fruitfly.xml");
    let xmlText = await (await fetch("/flybody/fruitfly.xml")).text();

    // Inject a floor + top light into the body-only fruitfly.xml so the
    // fly has somewhere to land and the right pane gets some sky light.
    xmlText = xmlText.replace(
      "<worldbody>",
      `<worldbody>
        <geom name="floor" type="plane" size="200 200 0.01" pos="0 0 0" rgba="0.06 0.07 0.10 1" condim="3" friction="1 0.005 0.0001" contype="1" conaffinity="1"/>
        <light name="top" pos="0 0 5" dir="0 0 -1" diffuse="0.4 0.4 0.4"/>`,
    );

    // Discover mesh files referenced and bin-load them into the VFS.
    const meshFiles = Array.from(
      new Set(
        Array.from(xmlText.matchAll(/file="([^"]+)"/g), (m) => m[1])
          .filter((f) => f.endsWith(".obj")),
      ),
    );
    onProgress?.(`fetching ${meshFiles.length} meshes`);

    p.vfs = new p.mujoco.MjVFS();
    const CONCURRENCY = 4;
    let inFlight = 0, idx = 0, completed = 0, totalBytes = 0;
    const t0 = performance.now();
    await new Promise<void>((resolve, reject) => {
      const next = () => {
        while (inFlight < CONCURRENCY && idx < meshFiles.length) {
          const file = meshFiles[idx++];
          inFlight++;
          fetch(`/flybody/${file}`)
            .then((r) => {
              if (!r.ok) throw new Error(`${file}: HTTP ${r.status}`);
              return r.arrayBuffer();
            })
            .then((buf) => {
              p.vfs.addBuffer(file, new Uint8Array(buf));
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

    onProgress?.("compiling MJCF (synchronous; tab may freeze ~5-15s)");
    const tCompile = performance.now();
    p.model = p.mujoco.MjModel.from_xml_string(xmlText, p.vfs);
    p.data = new p.mujoco.MjData(p.model);
    onProgress?.(`MJCF compiled in ${((performance.now() - tCompile) / 1000).toFixed(1)} s`);

    // Reset to defaults, lift thorax above the floor we injected, then
    // run forward kinematics so xpos/xquat are populated for t=0.
    p.mujoco.mj_resetData(p.model, p.data);
    const qpos = p.data.qpos as Float64Array;
    if (qpos.length >= 7) {
      qpos[0] = 0; qpos[1] = 0; qpos[2] = 0.5;
      qpos[3] = 1; qpos[4] = 0; qpos[5] = 0; qpos[6] = 0;
    }
    p.mujoco.mj_forward(p.model, p.data);

    // Build mesh id → file map so the renderer can resolve mjvGeom.dataid
    // → OBJ when it encounters mjGEOM_MESH geoms.
    const nmesh = p.model.nmesh as number;
    const xmlMeshFile = new Map<string, string>();
    for (const m of xmlText.matchAll(/<mesh\s+name="([^"]+)"\s+file="([^"]+)"\s*\/?>/g)) {
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
    if (qpos.length >= 7) {
      qpos[0] = 0; qpos[1] = 0; qpos[2] = 0.5;
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
