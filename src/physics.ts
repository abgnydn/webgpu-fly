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
  MjvScene, MjvOption, MjvPerturb, MjvCamera,
} from "@mujoco/mujoco";
import { getOrFetch } from "./cache";

export class Physics {
  mujoco!: MainModule;
  model!: MjModel;
  data!: MjData;
  scene!: MjvScene;

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
    //
    // Bundle path: tools/bake_flybody_bundle.py concats all 87 assets
    // (floor.xml, fruitfly.xml, 85 OBJs) into one binary so we get a
    // single IDB transaction instead of 87. Per-file IDB transactions
    // were costing ~300-1000 ms each (~34 s total even on cache hits!).
    const base = (import.meta.env.VITE_FLYBODY_URL || "/flybody").replace(/\/$/, "");
    const bundleUrl = (import.meta.env.VITE_FLYBODY_BUNDLE_URL || "/flybody.bundle.bin")
      + (((globalThis as unknown as { __flybodyBundleVersion?: string }).__flybodyBundleVersion)
        ? `?v=${(globalThis as unknown as { __flybodyBundleVersion?: string }).__flybodyBundleVersion}`
        : "");

    onProgress?.("fetching flybody bundle (140 MB, one IDB transaction)");
    const t0 = performance.now();
    const bundle = await getOrFetch(bundleUrl, bundleUrl);
    onProgress?.(`bundle fetched in ${((performance.now() - t0) / 1000).toFixed(1)} s`);

    // Parse bundle: tiny header + manifest, then names + data sections.
    const bv = new DataView(bundle);
    const magic = String.fromCharCode(...new Uint8Array(bundle, 0, 8));
    if (magic !== "WGFLYBND") {
      throw new Error(`flybody bundle bad magic: "${magic}", expected "WGFLYBND"`);
    }
    const bundleVersion = bv.getUint32(8, true);
    if (bundleVersion !== 1) throw new Error(`flybody bundle version ${bundleVersion} unsupported`);
    const nFiles = bv.getUint32(12, true);
    const HEADER_BYTES = 16;
    const namesSectionOff = HEADER_BYTES + nFiles * 16;
    const u8 = new Uint8Array(bundle);
    const decoder = new TextDecoder("utf-8");
    const fileBytes = new Map<string, Uint8Array>();
    for (let i = 0; i < nFiles; i++) {
      const off = HEADER_BYTES + i * 16;
      const nameOff = bv.getUint32(off + 0, true);
      const nameLen = bv.getUint32(off + 4, true);
      const dataOff = bv.getUint32(off + 8, true);
      const dataLen = bv.getUint32(off + 12, true);
      const name = decoder.decode(u8.subarray(namesSectionOff + nameOff, namesSectionOff + nameOff + nameLen));
      fileBytes.set(name, u8.subarray(dataOff, dataOff + dataLen));
    }
    onProgress?.(`bundle parsed: ${fileBytes.size} files`);

    const floorBytes = fileBytes.get("floor.xml");
    const flyBytes = fileBytes.get("fruitfly.xml");
    if (!floorBytes || !flyBytes) {
      throw new Error("flybody bundle missing floor.xml or fruitfly.xml");
    }
    const floorText = decoder.decode(floorBytes);
    const flyText = decoder.decode(flyBytes);

    // Mesh refs come from fruitfly.xml; floor.xml only has texture refs.
    const meshFiles = Array.from(
      new Set(
        Array.from(flyText.matchAll(/file="([^"]+)"/g), (m) => m[1])
          .filter((f) => f.endsWith(".obj")),
      ),
    );

    const vfs = new p.mujoco.MjVFS();
    let totalBytes = 0;
    for (const file of meshFiles) {
      const data = fileBytes.get(file);
      if (!data) {
        throw new Error(`flybody bundle missing mesh: ${file}`);
      }
      vfs.addBuffer(file, data);
      totalBytes += data.byteLength;
    }
    onProgress?.(`loaded ${meshFiles.length} meshes from bundle (${(totalBytes / 1e6).toFixed(0)} MB)`);
    void base;  // base URL retained for backward-compat env var; unused now.

    // The compiler resolves `<include file="fruitfly.xml"/>` from the
    // VFS, so we have to register fruitfly.xml there as well.
    vfs.addBuffer("fruitfly.xml", flyBytes);

    onProgress?.("compiling MJCF (synchronous; tab may freeze ~5-15s)");
    const tCompile = performance.now();
    p.model = p.mujoco.MjModel.from_xml_string(floorText, vfs);
    p.data = new p.mujoco.MjData(p.model);
    // The compiler copies everything it needs into mjModel; holding the
    // ~140 MB of OBJ bytes past this point just starves the wasm heap.
    vfs.delete();
    onProgress?.(`MJCF compiled in ${((performance.now() - tCompile) / 1000).toFixed(1)} s`);

    // Initialise to flybody's canonical rest pose, matching native
    // flybody/fruitfly/fruitfly.py:initialize_episode exactly:
    //   - mj_resetData → joints at MJCF default (zero for our model)
    //   - root xyz = _SPAWN_POS = (0, 0, 0.1278)
    //   - root quat = identity
    //   - ONLY wing joints set to qpos_spring (and only when wings
    //     are retracted, which is the trained-walking config).
    // The previous code did `qpos.set(qpos_spring)` for ALL joints —
    // legs included — which started the body in a non-rest pose the
    // trained policy never saw at episode start. That explains a big
    // chunk of why our trained walker was producing |action max|=18000
    // (raw policy output saturated way beyond [-1, 1] because the
    // input distribution was OOD).
    p.mujoco.mj_resetData(p.model, p.data);
    const qpos = p.data.qpos as Float64Array;
    if (qpos.length >= 7) {
      qpos[0] = 0; qpos[1] = 0; qpos[2] = 0.1278;          // _SPAWN_POS
      qpos[3] = 1; qpos[4] = 0; qpos[5] = 0; qpos[6] = 0;  // identity quat
    }
    // Wing-retract: copy qpos_spring for the 6 wing joints (yaw/roll/
    // pitch × left/right). Look them up by name — these are the only
    // joints that get the spring rest applied during walking init.
    const qposSpring = p.model.qpos_spring as Float64Array;
    if (qposSpring && qposSpring.length === qpos.length) {
      for (const side of ["left", "right"]) {
        for (const dof of ["yaw", "roll", "pitch"]) {
          const j = p.mujoco.mj_name2id(p.model, p.mujoco.mjtObj.mjOBJ_JOINT.value, `wing_${dof}_${side}`);
          if (j < 0) continue;
          const adr = (p.model.jnt_qposadr as Int32Array)[j];
          if (adr >= 0 && adr < qpos.length) qpos[adr] = qposSpring[adr];
        }
      }
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

    // The trained policy outputs 59 actions in flybody's _ACTION_CLASSES
    // order: adhesion → head → mouth → antennae → wings → abdomen → legs.
    // For the walk_imitation env (mouth/antennae/wings disabled), this
    // collapses to: 6 adhesion + 3 head + 2 abdomen + 48 legs = 59.
    // Cross-checked against env.action_spec() ranges from a live
    // flybody.tasks.base.Walking instantiation (tools/dump_flybody_spec.py).
    const walkingActionNames = [
      // 0..5: adhesion claws (ctrlrange [0, 1])
      "adhere_claw_T1_left",  "adhere_claw_T1_right",
      "adhere_claw_T2_left",  "adhere_claw_T2_right",
      "adhere_claw_T3_left",  "adhere_claw_T3_right",
      // 6..8: head
      "head_abduct", "head_twist", "head",
      // 9..10: abdomen
      "abdomen_abduct", "abdomen",
      // 11..58: 6 legs × 8 motors each (T1L, T1R, T2L, T2R, T3L, T3R)
      "coxa_abduct_T1_left",  "coxa_twist_T1_left",  "coxa_T1_left",
      "femur_twist_T1_left",  "femur_T1_left",       "tibia_T1_left",
      "tarsus_T1_left",       "tarsus2_T1_left",
      "coxa_abduct_T1_right", "coxa_twist_T1_right", "coxa_T1_right",
      "femur_twist_T1_right", "femur_T1_right",      "tibia_T1_right",
      "tarsus_T1_right",      "tarsus2_T1_right",
      "coxa_abduct_T2_left",  "coxa_twist_T2_left",  "coxa_T2_left",
      "femur_twist_T2_left",  "femur_T2_left",       "tibia_T2_left",
      "tarsus_T2_left",       "tarsus2_T2_left",
      "coxa_abduct_T2_right", "coxa_twist_T2_right", "coxa_T2_right",
      "femur_twist_T2_right", "femur_T2_right",      "tibia_T2_right",
      "tarsus_T2_right",      "tarsus2_T2_right",
      "coxa_abduct_T3_left",  "coxa_twist_T3_left",  "coxa_T3_left",
      "femur_twist_T3_left",  "femur_T3_left",       "tibia_T3_left",
      "tarsus_T3_left",       "tarsus2_T3_left",
      "coxa_abduct_T3_right", "coxa_twist_T3_right", "coxa_T3_right",
      "femur_twist_T3_right", "femur_T3_right",      "tibia_T3_right",
      "tarsus_T3_right",      "tarsus2_T3_right",
    ];
    p.walkingActuatorIds = walkingActionNames.map(id);
    // Per-actuator ctrlrange used to invert CanonicalSpecWrapper at
    // runtime. The trained policy emits an unbounded mean; Acme's
    // CanonicalSpec wraps it by tanh→[-1,1], then linearly remaps to
    // [lo, hi] of each actuator's ctrlrange. We bake that linear map
    // into mid+half * tanh(action[i]).
    const ctrlRange = p.model.actuator_ctrlrange as Float64Array;
    p.walkingActuatorMid = new Float32Array(59);
    p.walkingActuatorHalf = new Float32Array(59);
    for (let i = 0; i < 59; i++) {
      const a = p.walkingActuatorIds[i];
      if (a < 0) continue;
      const lo = ctrlRange[2 * a + 0];
      const hi = ctrlRange[2 * a + 1];
      p.walkingActuatorMid[i] = 0.5 * (lo + hi);
      p.walkingActuatorHalf[i] = 0.5 * (hi - lo);
    }
    // Separate ordering for the actuator_activation observable, which
    // dm_control fills in MJCF actuator declaration order via
    // observable.MJCFFeature('act', model.find_all('actuator')) — this
    // is a different order than the action vector (no adhesion-first).
    const walkingActivationNames = [
      // MJCF declaration order: 3 head, 2 abdomen, 48 leg, 6 adhesion
      "head_abduct", "head_twist", "head",
      "abdomen_abduct", "abdomen",
      "coxa_abduct_T1_left",  "coxa_twist_T1_left",  "coxa_T1_left",
      "femur_twist_T1_left",  "femur_T1_left",       "tibia_T1_left",
      "tarsus_T1_left",       "tarsus2_T1_left",
      "coxa_abduct_T1_right", "coxa_twist_T1_right", "coxa_T1_right",
      "femur_twist_T1_right", "femur_T1_right",      "tibia_T1_right",
      "tarsus_T1_right",      "tarsus2_T1_right",
      "coxa_abduct_T2_left",  "coxa_twist_T2_left",  "coxa_T2_left",
      "femur_twist_T2_left",  "femur_T2_left",       "tibia_T2_left",
      "tarsus_T2_left",       "tarsus2_T2_left",
      "coxa_abduct_T2_right", "coxa_twist_T2_right", "coxa_T2_right",
      "femur_twist_T2_right", "femur_T2_right",      "tibia_T2_right",
      "tarsus_T2_right",      "tarsus2_T2_right",
      "coxa_abduct_T3_left",  "coxa_twist_T3_left",  "coxa_T3_left",
      "femur_twist_T3_left",  "femur_T3_left",       "tibia_T3_left",
      "tarsus_T3_left",       "tarsus2_T3_left",
      "coxa_abduct_T3_right", "coxa_twist_T3_right", "coxa_T3_right",
      "femur_twist_T3_right", "femur_T3_right",      "tibia_T3_right",
      "tarsus_T3_right",      "tarsus2_T3_right",
      "adhere_claw_T1_left",  "adhere_claw_T1_right",
      "adhere_claw_T2_left",  "adhere_claw_T2_right",
      "adhere_claw_T3_left",  "adhere_claw_T3_right",
    ];
    p.walkingActivationIds = walkingActivationNames.map(id);
    const goodWalk = p.walkingActuatorIds.filter((i) => i >= 0).length;
    if (goodWalk !== 59) {
      console.warn(`walking actuator lookup: ${goodWalk}/59 resolved`);
    }

    // The 85 joints feeding joints_pos / joints_vel observables, in the
    // order returned by walker.observables.joints_pos._mjcf_element. Their
    // qpos addresses are NOT contiguous in the model — there are gaps for
    // rostrum/haustellum/labrum/antennae/wings — so we resolve each by
    // jnt_qposadr / jnt_dofadr. Source: tools/dump_flybody_spec.py.
    const walkingJointNames = [
      "head_abduct", "head_twist", "head",
      "abdomen_abduct", "abdomen",
      "abdomen_abduct_2", "abdomen_2",
      "abdomen_abduct_3", "abdomen_3",
      "abdomen_abduct_4", "abdomen_4",
      "abdomen_abduct_5", "abdomen_5",
      "abdomen_abduct_6", "abdomen_6",
      "abdomen_abduct_7", "abdomen_7",
      "haltere_left", "haltere_right",
      "coxa_abduct_T1_left",  "coxa_twist_T1_left",  "coxa_T1_left",
      "femur_twist_T1_left",  "femur_T1_left",       "tibia_T1_left",
      "tarsus_T1_left",       "tarsus2_T1_left",
      "tarsus3_T1_left",      "tarsus4_T1_left",     "tarsus5_T1_left",
      "coxa_abduct_T1_right", "coxa_twist_T1_right", "coxa_T1_right",
      "femur_twist_T1_right", "femur_T1_right",      "tibia_T1_right",
      "tarsus_T1_right",      "tarsus2_T1_right",
      "tarsus3_T1_right",     "tarsus4_T1_right",    "tarsus5_T1_right",
      "coxa_abduct_T2_left",  "coxa_twist_T2_left",  "coxa_T2_left",
      "femur_twist_T2_left",  "femur_T2_left",       "tibia_T2_left",
      "tarsus_T2_left",       "tarsus2_T2_left",
      "tarsus3_T2_left",      "tarsus4_T2_left",     "tarsus5_T2_left",
      "coxa_abduct_T2_right", "coxa_twist_T2_right", "coxa_T2_right",
      "femur_twist_T2_right", "femur_T2_right",      "tibia_T2_right",
      "tarsus_T2_right",      "tarsus2_T2_right",
      "tarsus3_T2_right",     "tarsus4_T2_right",    "tarsus5_T2_right",
      "coxa_abduct_T3_left",  "coxa_twist_T3_left",  "coxa_T3_left",
      "femur_twist_T3_left",  "femur_T3_left",       "tibia_T3_left",
      "tarsus_T3_left",       "tarsus2_T3_left",
      "tarsus3_T3_left",      "tarsus4_T3_left",     "tarsus5_T3_left",
      "coxa_abduct_T3_right", "coxa_twist_T3_right", "coxa_T3_right",
      "femur_twist_T3_right", "femur_T3_right",      "tibia_T3_right",
      "tarsus_T3_right",      "tarsus2_T3_right",
      "tarsus3_T3_right",     "tarsus4_T3_right",    "tarsus5_T3_right",
    ];
    const jId = (n: string) => p.mujoco.mj_name2id(p.model, p.mujoco.mjtObj.mjOBJ_JOINT.value, n);
    const jntQposAdr = p.model.jnt_qposadr as Int32Array;
    const jntDofAdr = p.model.jnt_dofadr as Int32Array;
    p.walkingQposAdr = new Int32Array(85);
    p.walkingDofAdr = new Int32Array(85);
    let goodJ = 0;
    for (let i = 0; i < 85; i++) {
      const j = jId(walkingJointNames[i]);
      if (j >= 0) {
        p.walkingQposAdr[i] = jntQposAdr[j];
        p.walkingDofAdr[i] = jntDofAdr[j];
        goodJ++;
      } else {
        p.walkingQposAdr[i] = -1;
        p.walkingDofAdr[i] = -1;
      }
    }
    if (goodJ !== 85) console.warn(`walking joint lookup: ${goodJ}/85 resolved`);

    // Cache sensor adr + dim by name. dm_control wires up sensors with
    // these exact names in flybody's MJCF.
    const sId = (n: string) => p.mujoco.mj_name2id(p.model, p.mujoco.mjtObj.mjOBJ_SENSOR.value, n);
    const sAdr = p.model.sensor_adr as Int32Array;
    const sDim = p.model.sensor_dim as Int32Array;
    const lookupSensor = (name: string): { adr: number; dim: number } | null => {
      const i = sId(name);
      if (i < 0) return null;
      return { adr: sAdr[i], dim: sDim[i] };
    };
    p.sensorIdx = {
      accel:  lookupSensor("accelerometer"),
      gyro:   lookupSensor("gyro"),
      vel:    lookupSensor("velocimeter"),
      forces: ["force_tarsus_T1_left", "force_tarsus_T1_right",
               "force_tarsus_T2_left", "force_tarsus_T2_right",
               "force_tarsus_T3_left", "force_tarsus_T3_right"]
              .map((n) => lookupSensor(n)),
      touches: ["touch_claw_T1_left", "touch_claw_T1_right",
                "touch_claw_T2_left", "touch_claw_T2_right",
                "touch_claw_T3_left", "touch_claw_T3_right"]
               .map((n) => lookupSensor(n)),
    };

    // appendages_pos observable in flybody is built from 7 SITES (the
    // claw tips of the 6 legs plus a head site), read in egocentric
    // frame (thorax-relative, thorax-rotated). We were previously
    // reading body xpos from the proximal tarsus body, which is the
    // wrong attachment point and gives the policy an OOD signal.
    const siteId = (n: string) => p.mujoco.mj_name2id(p.model, p.mujoco.mjtObj.mjOBJ_SITE.value, n);
    const bId = (n: string) => p.mujoco.mj_name2id(p.model, p.mujoco.mjtObj.mjOBJ_BODY.value, n);
    p.appendageSiteIds = [
      "claw_T1_left", "claw_T1_right",
      "claw_T2_left", "claw_T2_right",
      "claw_T3_left", "claw_T3_right",
      "head",
    ].map(siteId);
    p.thoraxBodyId = bId("thorax");

    onProgress?.(`flybody ready (${p.model.nbody} bodies, ${nmesh} meshes, wings=${p.wingActs ? "ok" : "missing"}, legs=${goodLegs}/6, walk-acts=${goodWalk}/59)`);
    return p;
  }

  /** 59 actuator IDs in policy ACTION order (adhesion-first). Used
   * for writing ctrl values. Filled in `create()`. */
  walkingActuatorIds: number[] = [];
  /** Per-action mid + half so we can apply the inverse of Acme's
   * CanonicalSpecWrapper: ctrl[a] = mid + tanh(action) * half. */
  walkingActuatorMid: Float32Array = new Float32Array(0);
  walkingActuatorHalf: Float32Array = new Float32Array(0);
  /** 59 actuator IDs in MJCF declaration order, used for reading the
   * actuator_activation observable (different from action order!). */
  walkingActivationIds: number[] = [];
  /** qpos / qvel addresses for the 85 joints feeding joints_pos /
   * joints_vel observables. Not contiguous — gaps for disabled
   * appendages. */
  walkingQposAdr: Int32Array = new Int32Array(0);
  walkingDofAdr: Int32Array = new Int32Array(0);
  /** Optional baked real-fly walking reference (from
   * tools/bake_walking_ref.py). When set + opt-in flag, drives
   * ref_displacement / ref_root_quat in buildWalkingObservation. */
  walkingRef: { qpos: Float32Array; numFrames: number; dt: number } | null = null;
  walkingRefStep = 0;
  /** Sensor adr + dim cache used by the trained-walker observation
   * builder. */
  sensorIdx: {
    accel: { adr: number; dim: number } | null;
    gyro:  { adr: number; dim: number } | null;
    vel:   { adr: number; dim: number } | null;
    forces: ({ adr: number; dim: number } | null)[];
    touches: ({ adr: number; dim: number } | null)[];
  } = { accel: null, gyro: null, vel: null, forces: [], touches: [] };
  /** Buffered sensor readings — flybody observables are aggregated
   * with `aggregator='mean'` and `buffer_size = control_timestep /
   * physics_timestep = 10`. We accumulate sensordata each substep
   * inside step() and divide by count when buildWalkingObservation
   * reads them. Without this the policy gets instantaneous-snapshot
   * sensor values, which is OOD for a policy trained on averaged
   * readings, and the network produces saturated garbage actions
   * (|max|=17000+ vs native's |max|=6.5).
   *
   * The buffer is reset at the start of each physics.step() call —
   * it captures the average sensor reading over THAT call's
   * substep run. The trained walker calls buildWalkingObservation
   * BEFORE physics.step, so the buffer is read for the previous
   * step's substeps and then refilled during this step.
   */
  sensorBuf = {
    accel:   new Float32Array(3),
    gyro:    new Float32Array(3),
    vel:     new Float32Array(3),
    forces:  new Float32Array(6 * 3),
    touches: new Float32Array(6),
    count: 0,
  };
  /** SITE IDs for the 7 appendages whose egocentric xpos forms the
   * appendages_pos observable (6 claw tips + head site). */
  appendageSiteIds: number[] = [];
  thoraxBodyId = -1;

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

  /** Body-velocity command from the VNC layer; re-asserted by step()
   * each substep so MuJoCo damping doesn't drain it. Set by driveLegs
   * each frame. */
  private fwdCmd = 0;
  private turnCmd = 0;

  /** When true, write the brain's motor command directly into the
   * freejoint translational+yaw qvel. Cheats over MuJoCo physics —
   * the legs visibly step but contribute nothing to body motion.
   * Default ON because browser mujoco_wasm produces ~8× less leg
   * thrust than native MuJoCo (per state.md the RL walker without
   * the assist crawls at ~0.125 cm/s vs the expected ~1 cm/s),
   * which makes the demo unwatchable and breaks every body-motion
   * e2e test. Set to false from console to demo "honest physics":
   *   (window as any).Physics.kinematicAssistEnabled = false  */
  static kinematicAssistEnabled = true;

  /** Step physics N times.
   *
   * Body is driven by leg/wing actuators only — leg motion produces
   * ground reaction, ground reaction moves body. No qvel writes on
   * the freejoint translation or yaw (was a kinematic-assist hack
   * before; toggleable now via Physics.kinematicAssistEnabled).
   *
   * Stabilizer: pitch/roll angular damper (×0.85 per substep) keeps
   * the body upright without pinning orientation; the fly can still
   * tip if it genuinely loses balance. */
  step(substeps = 1) {
    const assist = Physics.kinematicAssistEnabled;
    const hasCmd = assist && (Math.abs(this.fwdCmd) > 0.01 || Math.abs(this.turnCmd) > 0.01);
    // Reset sensor accumulators — this step's substeps fill them; the
    // next buildWalkingObservation reads the mean.
    this.sensorBuf.accel.fill(0);
    this.sensorBuf.gyro.fill(0);
    this.sensorBuf.vel.fill(0);
    this.sensorBuf.forces.fill(0);
    this.sensorBuf.touches.fill(0);
    this.sensorBuf.count = 0;
    const sensordata = this.data.sensordata as Float32Array;
    const sIdx = this.sensorIdx;
    for (let s = 0; s < substeps; s++) {
      const qpos = this.data.qpos as Float64Array;
      const qvel = this.data.qvel as Float64Array;
      if (qvel && qvel.length >= 6) {
        qvel[3] *= 0.85;   // pitch damping
        qvel[4] *= 0.85;   // roll damping
      }
      if (hasCmd && qpos && qvel && qpos.length >= 7) {
        const qw = qpos[3], qx = qpos[4], qy = qpos[5], qz = qpos[6];
        const fx = 1 - 2 * (qy * qy + qz * qz);
        const fy = 2 * (qx * qy + qw * qz);
        const v = 1.0 * this.fwdCmd;
        qvel[0] = fx * v;
        qvel[1] = fy * v;
        qvel[5] = this.turnCmd * 6.0;
      }
      this.mujoco.mj_step(this.model, this.data);
      // Sensor buffering: accumulate readings for averaging in obs.
      if (sIdx.accel) {
        const a = sIdx.accel.adr;
        this.sensorBuf.accel[0] += sensordata[a + 0];
        this.sensorBuf.accel[1] += sensordata[a + 1];
        this.sensorBuf.accel[2] += sensordata[a + 2];
      }
      if (sIdx.gyro) {
        const a = sIdx.gyro.adr;
        this.sensorBuf.gyro[0] += sensordata[a + 0];
        this.sensorBuf.gyro[1] += sensordata[a + 1];
        this.sensorBuf.gyro[2] += sensordata[a + 2];
      }
      if (sIdx.vel) {
        const a = sIdx.vel.adr;
        this.sensorBuf.vel[0] += sensordata[a + 0];
        this.sensorBuf.vel[1] += sensordata[a + 1];
        this.sensorBuf.vel[2] += sensordata[a + 2];
      }
      for (let i = 0; i < 6; i++) {
        const f = sIdx.forces[i];
        if (f) {
          this.sensorBuf.forces[3 * i + 0] += sensordata[f.adr + 0];
          this.sensorBuf.forces[3 * i + 1] += sensordata[f.adr + 1];
          this.sensorBuf.forces[3 * i + 2] += sensordata[f.adr + 2];
        }
        const t = sIdx.touches[i];
        if (t) this.sensorBuf.touches[i] += sensordata[t.adr];
      }
      this.sensorBuf.count++;
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
  /** Gait params (initially the hand-seeded baseline). The "Evolve gait"
   * UI button replaces these with whatever the WebGPU ARS optimizer
   * found in src/evolution.ts. driveLegs reads these every frame so the
   * winner takes effect instantly. */
  gait = {
    freq: 10.0,
    coxaAmp: 0.85,
    femurAmp: 0.65,
    tibiaAmp: 0.35,
    swingRatio: 0.5,
    liftOffset: 0.5,
  };

  driveLegs(walk: number, turn = 0) {
    const ctrl = this.data.ctrl as Float64Array;
    const t = this.data.time as number;
    const walkSigned = Math.max(-1, Math.min(1, walk));
    const w = Math.abs(walkSigned);
    const tu = Math.max(-1, Math.min(1, turn));

    // Sign of walk flips the leg-sweep direction so the body moves
    // backward for negative walk (DNb01 moonwalker) — same CPG, mirrored.
    const dirSign = walkSigned < 0 ? -1 : 1;
    const drv = Math.sqrt(w);
    const freq = this.gait.freq * drv;
    const phase = t * freq * 2 * Math.PI;
    const COXA_AMP = this.gait.coxaAmp;
    const FEMUR_LIFT = this.gait.femurAmp;
    const TIBIA_PUSH = this.gait.tibiaAmp;
    const swingThresh = Math.cos(this.gait.swingRatio * Math.PI);
    const liftPhaseShift = this.gait.liftOffset * Math.PI;

    for (const leg of Physics.LEG_KEYS) {
      const acts = this.legActs[leg];
      if (acts.coxa < 0) continue;
      const isLeft = leg.endsWith("_left");
      const tripodA = leg === "T1_left" || leg === "T2_right" || leg === "T3_left";
      const p = tripodA ? phase : phase + Math.PI;
      const sp = Math.sin(p);
      // Stance / swing partitioned by swingRatio: swingNow when sin(p)
      // exceeds cos(swingRatio·π), so smaller swingRatio = longer stance.
      const swingNow = sp > swingThresh;
      // Femur lift uses an offset-shifted phase so it leads (or lags)
      // the coxa sweep depending on liftOffset.
      const liftP = p + liftPhaseShift;

      const turnMod = isLeft ? (1 - tu * 0.5) : (1 + tu * 0.5);
      const sideSign = isLeft ? 1 : -1;

      ctrl[acts.coxa] = dirSign * sideSign * COXA_AMP * w * turnMod * sp;

      if (acts.femur >= 0) {
        ctrl[acts.femur] = swingNow ? FEMUR_LIFT * w * Math.sin(liftP) : 0;
      }
      if (acts.tibia >= 0) {
        ctrl[acts.tibia] = swingNow ? 0 : -TIBIA_PUSH * w * sp;
      }
      // Adhesion: HIGH during stance, LOW during swing. Stand still
      // when not commanded so the fly doesn't slide.
      if (acts.adhesion >= 0) {
        const stancePhase = swingNow ? 0 : 1.0;
        ctrl[acts.adhesion] = w > 0.01 ? stancePhase : 1.0;
      }
    }

    // Cache the kinematic command for step()'s substep loop. When walk
    // is zero this is zero, so the body sits when the brain is silent.
    this.fwdCmd = walkSigned;
    this.turnCmd = tu;
  }

  /** Replace the live gait params with an evolved policy from
   * src/evolution.ts. driveLegs picks them up on the next frame. */
  applyEvolvedGait(g: { freq: number; coxaAmp: number; femurAmp: number; tibiaAmp: number; swingRatio: number; liftOffset: number }) {
    this.gait.freq = g.freq;
    this.gait.coxaAmp = g.coxaAmp;
    this.gait.femurAmp = g.femurAmp;
    this.gait.tibiaAmp = g.tibiaAmp;
    this.gait.swingRatio = g.swingRatio;
    this.gait.liftOffset = g.liftOffset;
  }

  /**
   * Build the 741-dim observation vector the trained walking policy
   * expects from current mujoco_wasm state, in dm_control's
   * alphabetical concat order. Ref trajectory is generated forward
   * at the given target velocity (cm/s) for 65 frames at the env's
   * 50 Hz tick.
   *
   * Caller fills the result into a Float32Array via the slot offsets
   * in WALKING_OBS_LAYOUT (walking-policy.ts). This method writes
   * directly into `obs` to avoid per-frame allocation.
   */
  buildWalkingObservation(obs: Float32Array, targetSpeedCmPerS: number): void {
    if (obs.length !== 741) throw new Error(`obs length must be 741, got ${obs.length}`);
    const data = this.data;
    const sensordata = data.sensordata as Float32Array;
    const qpos = data.qpos as Float64Array;
    const qvel = data.qvel as Float64Array;
    const act = data.act as Float64Array;
    const xpos = data.xpos as Float64Array;
    const xmat = data.xmat as Float64Array;

    // ---- 0..2: accelerometer ------------------------------------------
    // All sensor channels read from the BUFFERED MEAN over the last
    // physics.step() substep run (filled by step()). flybody's
    // observables use buffer_size = control_timestep / physics_timestep
    // = 10 with aggregator='mean'. Without averaging we feed the policy
    // instantaneous-snapshot readings, which is OOD vs training.
    const buf = this.sensorBuf;
    const bufN = buf.count > 0 ? buf.count : 1;
    let off = 0;
    if (this.sensorIdx.accel) {
      obs[off + 0] = buf.accel[0] / bufN;
      obs[off + 1] = buf.accel[1] / bufN;
      obs[off + 2] = buf.accel[2] / bufN;
    }
    off += 3;
    // ---- 3..61: actuator_activation (59 in MJCF declaration order) ---
    // dm_control fills this via observable.MJCFFeature('act', actuators)
    // where actuators = mjcf_model.find_all('actuator') — i.e. MJCF
    // declaration order, NOT the policy's action class order. Adhesion
    // actuators have no act state (dyntype=none), so they appear as 0.
    for (let i = 0; i < 59; i++) {
      const a = this.walkingActivationIds[i];
      const v = (act && a >= 0 && act.length > a) ? act[a] : 0;
      obs[off + i] = v;
    }
    off += 59;
    // ---- 62..82: appendages_pos (7×3 egocentric, claw sites + head) -
    const siteXpos = (this.data as { site_xpos?: Float64Array }).site_xpos;
    if (this.thoraxBodyId >= 0 && siteXpos) {
      const tx = xpos[3 * this.thoraxBodyId + 0];
      const ty = xpos[3 * this.thoraxBodyId + 1];
      const tz = xpos[3 * this.thoraxBodyId + 2];
      // Thorax world rotation matrix (3×3, row-major in xmat[9*body]).
      const m = 9 * this.thoraxBodyId;
      const r00 = xmat[m + 0], r01 = xmat[m + 1], r02 = xmat[m + 2];
      const r10 = xmat[m + 3], r11 = xmat[m + 4], r12 = xmat[m + 5];
      const r20 = xmat[m + 6], r21 = xmat[m + 7], r22 = xmat[m + 8];
      for (let i = 0; i < 7; i++) {
        const s = this.appendageSiteIds[i];
        if (s < 0) { obs[off + 3 * i] = obs[off + 3 * i + 1] = obs[off + 3 * i + 2] = 0; continue; }
        const dx = siteXpos[3 * s + 0] - tx;
        const dy = siteXpos[3 * s + 1] - ty;
        const dz = siteXpos[3 * s + 2] - tz;
        // Body-frame projection: world_delta @ R == R^T · world_delta.
        obs[off + 3 * i + 0] = r00 * dx + r10 * dy + r20 * dz;
        obs[off + 3 * i + 1] = r01 * dx + r11 * dy + r21 * dz;
        obs[off + 3 * i + 2] = r02 * dx + r12 * dy + r22 * dz;
      }
    }
    off += 21;
    // ---- 83..100: force (6×3, buffered mean) -------------------------
    for (let i = 0; i < 6; i++) {
      if (this.sensorIdx.forces[i]) {
        obs[off + 3 * i + 0] = buf.forces[3 * i + 0] / bufN;
        obs[off + 3 * i + 1] = buf.forces[3 * i + 1] / bufN;
        obs[off + 3 * i + 2] = buf.forces[3 * i + 2] / bufN;
      }
    }
    off += 18;
    // ---- 101..103: gyro (buffered mean) -----------------------------
    if (this.sensorIdx.gyro) {
      obs[off + 0] = buf.gyro[0] / bufN;
      obs[off + 1] = buf.gyro[1] / bufN;
      obs[off + 2] = buf.gyro[2] / bufN;
    }
    off += 3;
    // ---- 104..188: joints_pos (85) ----------------------------------
    // The 85 joints feeding joints_pos are NOT contiguous in qpos —
    // there are gaps for rostrum/haustellum/labrum/antennae/wings.
    // walkingQposAdr[] holds the qpos address of each joint in
    // observation order (decoded from flybody's
    // walker.observables.joints_pos._mjcf_element list).
    for (let i = 0; i < 85; i++) {
      const a = this.walkingQposAdr[i];
      obs[off + i] = (a >= 0 && a < qpos.length) ? qpos[a] : 0;
    }
    off += 85;
    // ---- 189..273: joints_vel (85) ----------------------------------
    for (let i = 0; i < 85; i++) {
      const a = this.walkingDofAdr[i];
      obs[off + i] = (a >= 0 && a < qvel.length) ? qvel[a] : 0;
    }
    off += 85;
    // ---- 274..468: ref_displacement (65×3) ---------------------------
    // ---- 469..728: ref_root_quat (65×4) ------------------------------
    // Two paths:
    //  (a) Default: synthetic straight-line forward ref + identity quat.
    //      Robust, matches what's been e2e-tested.
    //  (b) Opt-in: replay real fly mocap from public/walking-ref.bin
    //      (baked by tools/bake_walking_ref.py from the Vaxenburg 2025
    //      mocap dataset). Closer to the policy's training distribution.
    //      Toggle:  (window as any).__walkingRefFromMocap = true
    const useMocap = this.walkingRef !== null
      && (typeof globalThis !== "undefined"
        && (globalThis as { __walkingRefFromMocap?: boolean }).__walkingRefFromMocap === true);

    if (useMocap) {
      const ref = this.walkingRef!;
      const T = ref.numFrames;
      const step = this.walkingRefStep % T;
      const px = ref.qpos[step * 7 + 0];
      const py = ref.qpos[step * 7 + 1];
      const pz = ref.qpos[step * 7 + 2];
      const qw = ref.qpos[step * 7 + 3];
      const qx = ref.qpos[step * 7 + 4];
      const qy = ref.qpos[step * 7 + 5];
      const qz = ref.qpos[step * 7 + 6];
      // Mocap pose at `step`: rotation matrix (mocap_R) and inverse quat.
      // R^T columns = mocap body x/y/z axes in world; we use R^T to
      // express world-frame deltas in mocap's body frame at step.
      const r00 = 1 - 2 * (qy * qy + qz * qz);
      const r01 = 2 * (qx * qy - qw * qz);
      const r02 = 2 * (qx * qz + qw * qy);
      const r10 = 2 * (qx * qy + qw * qz);
      const r11 = 1 - 2 * (qx * qx + qz * qz);
      const r12 = 2 * (qy * qz - qw * qx);
      const r20 = 2 * (qx * qz - qw * qy);
      const r21 = 2 * (qy * qz + qw * qx);
      const r22 = 1 - 2 * (qx * qx + qy * qy);
      // q_inv(mocap_q) — used to express mocap_q[step+f] in body frame.
      const iqw = qw, iqx = -qx, iqy = -qy, iqz = -qz;
      let dispOff = off;
      let quatOff = off + 195;
      for (let f = 0; f < 65; f++) {
        const idx = ((step + f) % T) * 7;
        const fx = ref.qpos[idx + 0] - px;
        const fy = ref.qpos[idx + 1] - py;
        const fz = ref.qpos[idx + 2] - pz;
        // R^T applied to world-frame delta = body-frame future motion.
        // R^T row i is column i of R = (r0i, r1i, r2i).
        obs[dispOff + 3 * f + 0] = r00 * fx + r10 * fy + r20 * fz;
        obs[dispOff + 3 * f + 1] = r01 * fx + r11 * fy + r21 * fz;
        obs[dispOff + 3 * f + 2] = r02 * fx + r12 * fy + r22 * fz;
        // q_local = q_inv(mocap_q[step]) ⊗ mocap_q[step+f]
        const fqw = ref.qpos[idx + 3];
        const fqx = ref.qpos[idx + 4];
        const fqy = ref.qpos[idx + 5];
        const fqz = ref.qpos[idx + 6];
        obs[quatOff + 4 * f + 0] = iqw * fqw - iqx * fqx - iqy * fqy - iqz * fqz;
        obs[quatOff + 4 * f + 1] = iqw * fqx + iqx * fqw + iqy * fqz - iqz * fqy;
        obs[quatOff + 4 * f + 2] = iqw * fqy - iqx * fqz + iqy * fqw + iqz * fqx;
        obs[quatOff + 4 * f + 3] = iqw * fqz + iqx * fqy - iqy * fqx + iqz * fqw;
      }
      this.walkingRefStep = (this.walkingRefStep + 1) % T;
      off += 195 + 260;
    } else {
      // Default: synthetic forward straight-line ref + identity quat.
      // dtFrame = flybody's _WALK_CONTROL_TIMESTEP (2 ms = 500 Hz). The
      // trained policy was trained with this exact lookahead step, so
      // feeding it 1/50 (which we used to use) made every ref value
      // 10× larger than the policy expected — pushed obs out of
      // distribution and the policy produced garbage actions.
      const dtFrame = 0.002;
      const stepCm = targetSpeedCmPerS * dtFrame;
      for (let f = 0; f < 65; f++) {
        obs[off + 3 * f + 0] = (f + 1) * stepCm;
        obs[off + 3 * f + 1] = 0;
        obs[off + 3 * f + 2] = 0;
      }
      off += 195;
      for (let f = 0; f < 65; f++) {
        obs[off + 4 * f + 0] = 1;
        obs[off + 4 * f + 1] = 0;
        obs[off + 4 * f + 2] = 0;
        obs[off + 4 * f + 3] = 0;
      }
      off += 260;
    }
    // ---- 729..734: touch (6, buffered mean) -------------------------
    for (let i = 0; i < 6; i++) {
      obs[off + i] = this.sensorIdx.touches[i] ? buf.touches[i] / bufN : 0;
    }
    off += 6;
    // ---- 735..737: velocimeter (buffered mean) ----------------------
    if (this.sensorIdx.vel) {
      obs[off + 0] = buf.vel[0] / bufN;
      obs[off + 1] = buf.vel[1] / bufN;
      obs[off + 2] = buf.vel[2] / bufN;
    }
    off += 3;
    // ---- 738..740: world_zaxis (z-axis of world in body frame) ------
    // Reading thorax xmat directly: the third COLUMN (entries 2, 5, 8)
    // is the body-frame projection of world +z. Equivalent to "what
    // direction does up point to in fly coords."
    if (this.thoraxBodyId >= 0) {
      const m = 9 * this.thoraxBodyId;
      obs[off + 0] = xmat[m + 2];
      obs[off + 1] = xmat[m + 5];
      obs[off + 2] = xmat[m + 8];
    }
    off += 3;
    if (off !== 741) throw new Error(`obs build went wrong: filled ${off} of 741 slots`);
  }

  /**
   * Apply 59 actions from the trained walking policy to the
   * corresponding 59 actuators.
   *
   * Acme's training stack wraps the env with `CanonicalSpecWrapper`,
   * which (1) tanh's the policy's mean output to bound it in [-1, 1],
   * then (2) linearly remaps to each actuator's actual ctrlrange. We
   * invert that wrapper here using the precomputed per-actuator
   * (mid, half) so the policy can produce realistic torques across
   * the full ctrlrange — for example head_twist's [-3, 3] vs
   * adhere_claw's [0, 1].
   */
  applyTrainedWalkerActions(actions: Float32Array): void {
    if (actions.length !== 59) throw new Error(`actions must be 59-dim, got ${actions.length}`);
    const ctrl = this.data.ctrl as Float64Array;
    for (let i = 0; i < 59; i++) {
      const a = this.walkingActuatorIds[i];
      if (a < 0) continue;
      // Acme's CanonicalSpec wrapper does CLIP + linear rescale (NOT
       // tanh). tanh under-saturates raw policy outputs in the [0.5, 5]
      // range — which is where the policy spends most of its
      // confidence — so legs got far less thrust than training. Native
      // flybody achieves ~2.4 cm/s forward walking; our wasm was at
      // ~0.04 cm/s before this fix.
      const a_canonical = actions[i] < -1 ? -1 : actions[i] > 1 ? 1 : actions[i];
      ctrl[a] = this.walkingActuatorMid[i] + a_canonical * this.walkingActuatorHalf[i];
    }
  }

  /** Apply an instantaneous vertical impulse to the freejoint body —
   *  used for the DNp01 (Giant Fiber) escape-jump approximation.
   *  Real fly take-off velocity is ~30 cm/s upward over ~5 ms. */
  jumpImpulse(verticalSpeed: number) {
    const qvel = this.data.qvel as Float64Array;
    if (qvel && qvel.length >= 3) qvel[2] = verticalSpeed;
  }

  /** World-frame xy speed of the thorax, cm/s, from data.qvel. */
  get bodySpeed(): number {
    const qvel = this.data.qvel as Float64Array | null;
    if (!qvel || qvel.length < 2) return 0;
    return Math.sqrt(qvel[0] * qvel[0] + qvel[1] * qvel[1]);
  }

  reset() {
    this.mujoco.mj_resetData(this.model, this.data);
    const qpos = this.data.qpos as Float64Array;
    const qposSpring = this.model.qpos_spring as Float64Array;
    if (qposSpring && qposSpring.length === qpos.length) {
      for (const side of ["left", "right"]) {
        for (const dof of ["yaw", "roll", "pitch"]) {
          const j = this.mujoco.mj_name2id(this.model, this.mujoco.mjtObj.mjOBJ_JOINT.value, `wing_${dof}_${side}`);
          if (j < 0) continue;
          const adr = (this.model.jnt_qposadr as Int32Array)[j];
          if (adr >= 0 && adr < qpos.length) qpos[adr] = qposSpring[adr];
        }
      }
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
  }

  get bodyCount() { return this.model.nbody as number; }
}
