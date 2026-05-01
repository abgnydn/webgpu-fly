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

  /** mesh id (mujoco mesh table index) → OBJ filename. */
  meshFileById: string[] = [];
  /** Cached wing actuator ids by axis × side (flybody convention). */
  private wingActs: { yawL: number; yawR: number; rollL: number; rollR: number; pitchL: number; pitchR: number } | null = null;
  /** Cached per-leg actuator ids for the VNC stand-in CPG. */
  private legActs: Record<string, { coxa: number; femur: number; tibia: number; adhesion: number }> = {};
  /** Six tarsus-claw legs in flybody. */
  private static readonly LEG_KEYS = ["T1_left", "T1_right", "T2_left", "T2_right", "T3_left", "T3_right"] as const;

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

    // Cache the six wing actuators (yaw / roll / pitch × left / right) so
    // we can drive them with flybody's canonical wing-beat pattern.
    const id = (n: string) => p.mujoco.mj_name2id(p.model, p.mujoco.mjtObj.mjOBJ_ACTUATOR.value, n);
    const yawL = id("wing_yaw_left"), yawR = id("wing_yaw_right");
    const rollL = id("wing_roll_left"), rollR = id("wing_roll_right");
    const pitchL = id("wing_pitch_left"), pitchR = id("wing_pitch_right");
    if (yawL >= 0 && yawR >= 0 && rollL >= 0 && rollR >= 0 && pitchL >= 0 && pitchR >= 0) {
      p.wingActs = { yawL, yawR, rollL, rollR, pitchL, pitchR };
    }
    // Cache leg actuator IDs (coxa swing, femur lift, tibia extend, claw
    // adhesion) per leg. The VNC stand-in writes these each frame to
    // produce a tripod gait — adhesion modulates between stance (1.0)
    // and swing (0.0) per cycle so feet release before lifting.
    for (const leg of Physics.LEG_KEYS) {
      p.legActs[leg] = {
        coxa: id(`coxa_${leg}`),
        femur: id(`femur_${leg}`),
        tibia: id(`tibia_${leg}`),
        adhesion: id(`adhere_claw_${leg}`),
      };
    }
    // Default all six adhesion ctrls fully on so the standing fly
    // doesn't slide before any drive arrives.
    const ctrl = p.data.ctrl as Float64Array;
    for (const leg of Physics.LEG_KEYS) {
      const aId = p.legActs[leg].adhesion;
      if (aId >= 0) ctrl[aId] = 1.0;
    }

    const goodLegs = Physics.LEG_KEYS.filter((k) => p.legActs[k].coxa >= 0).length;
    onProgress?.(`flybody ready (${p.model.nbody} bodies, ${nmesh} meshes, wings=${p.wingActs ? "ok" : "missing"}, legs=${goodLegs}/6)`);
    return p;
  }

  /**
   * Drive the six wing actuators with flybody's canonical wing-beat
   * pattern (TuragaLab/flybody/flybody/tasks/pattern_generators.py:
   * default approximation; base frequency 218 Hz from constants.py
   * _WING_PARAMS['base_freq']).
   *
   * @param amp  ∈ [0, 1]  overall flap amplitude (drive intensity).
   * @param asym ∈ [-1, 1] left/right amplitude bias. Positive = right
   *   wing flaps harder than left → torque steers the fly leftward.
   *   This is the basic mechanism real flies use to turn in flight.
   *
   * Also re-asserts claw adhesion at 1.0 so the freejoint body stays
   * grounded against wing reactive force (no actuated walking yet —
   * that needs flybody's RL-trained walk_imitation policy, which is a
   * separate port).
   */
  driveWings(amp: number, asym = 0) {
    if (!this.wingActs) return;
    const ctrl = this.data.ctrl as Float64Array;
    const t = this.data.time as number;
    // Cap overall amplitude. flybody's canonical pattern peaks at
    // pitch ≈ 2.15 — that IS the full-flight amplitude meant to lift
    // a real fly. With our claw-adhesion-only grounding, anything
    // above ~0.25 produces enough lift to launch the freejoint body.
    // Real flight needs the full RL-trained controller (flight_imitation
    // task) coordinating wings + body pitch + abdomen.
    const a = Math.max(0, Math.min(1, amp)) * 0.2;
    const k = Math.max(-1, Math.min(1, asym)) * 0.4;   // ±40% L/R bias
    const aL = a * (1 - k);
    const aR = a * (1 + k);
    const x = t * 218.0 * 2 * Math.PI;
    const yaw   = 1.1  * Math.sin(x - Math.PI / 2) + 0.3;
    const roll  = 0.25 * Math.sin(1.5 * x)         - 0.1;
    const pitch = 1.35 * Math.sin(x)               + 0.8;
    const w = this.wingActs;
    ctrl[w.yawL]   = yaw   * aL;
    ctrl[w.yawR]   = yaw   * aR;
    ctrl[w.rollL]  = roll  * aL;
    ctrl[w.rollR]  = roll  * aR;
    ctrl[w.pitchL] = pitch * aL;
    ctrl[w.pitchR] = pitch * aR;
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

  /**
   * VNC stand-in: tripod-gait CPG that writes per-leg coxa swing,
   * femur lift and adhesion gating into flybody's position actuators.
   * The brain (FlyWire) provides the high-level signals — this is the
   * spinal-cord layer between brain and muscle that real flies have
   * in their VNC. Hand-coded analytic primitive, not RL-trained.
   *
   * @param walk  ∈ [0, 1]  forward locomotion drive (DN sum).
   * @param turn  ∈ [-1, 1] L/R asymmetry (DN imbalance). Positive →
   *   left legs slow, right legs fast → fly veers left.
   */
  driveLegs(walk: number, turn = 0) {
    const ctrl = this.data.ctrl as Float64Array;
    const t = this.data.time as number;
    const w = Math.max(0, Math.min(1, walk));
    const tu = Math.max(-1, Math.min(1, turn));

    // Cycle frequency scales with √drive so even modest DN levels
    // produce a visible cycle. Real fly walk freq is 5-15 Hz; we scale
    // 0 Hz (rest) → ~10 Hz at full drive.
    const drv = Math.sqrt(w);
    const freq = 10.0 * drv;
    const phase = t * freq * 2 * Math.PI;

    // Gait amplitudes pushed toward the high end of fly literature
    // (Mendes 2013, Berendes 2016) so steps are visibly large at
    // moderate drive. Coxa swing ~25° peak → 0.45 rad. Femur lift ~30°
    // peak → 0.55 rad ensures foot clears the floor in swing phase.
    const COXA_AMP = 0.55;
    const FEMUR_LIFT = 0.55;

    // Tripod groups: A = {T1L, T2R, T3L} swings together, B = the
    // others — π out of phase.
    for (const leg of Physics.LEG_KEYS) {
      const acts = this.legActs[leg];
      if (acts.coxa < 0) continue;
      const isLeft = leg.endsWith("_left");
      const tripodA = leg === "T1_left" || leg === "T2_right" || leg === "T3_left";
      const p = tripodA ? phase : phase + Math.PI;
      const swingNow = Math.sin(p) > 0;

      // Turn modulation: ipsilateral side stride shorter, contra longer.
      // turn > 0 → veer left → left legs slower (smaller stride).
      const turnMod = isLeft ? (1 - tu * 0.5) : (1 + tu * 0.5);

      // Coxa: forward-swing during swing phase, push-back during stance.
      ctrl[acts.coxa] = COXA_AMP * w * turnMod * Math.sin(p);

      // Femur: lift in swing only.
      if (acts.femur >= 0) {
        ctrl[acts.femur] = swingNow ? FEMUR_LIFT * w * Math.sin(p) : 0;
      }
      // Tibia stays at spring rest (ctrl = 0 → bias-driven default).
      if (acts.tibia >= 0) ctrl[acts.tibia] = 0;

      // Adhesion stays at 1.0 always — when the leg is in swing the
      // femur lift takes the foot off the floor, so adhesion is moot;
      // when in stance, full adhesion gives max friction so the leg's
      // backward sweep pushes the body forward instead of slipping.
      // (Earlier 0/1 gating during swing was disrupting the cycle.)
      if (acts.adhesion >= 0) ctrl[acts.adhesion] = 1.0;
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
