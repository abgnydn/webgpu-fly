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
    // Allow VITE_FLYBODY_URL to point at an external host (CDN / GitHub
    // Release) for the 134 MB of flybody assets. Defaults to the local
    // public/flybody dir for dev.
    const base = (import.meta.env.VITE_FLYBODY_URL || "/flybody").replace(/\/$/, "");
    onProgress?.("fetching floor.xml + fruitfly.xml");
    const [floorText, flyText] = await Promise.all([
      fetch(`${base}/floor.xml`).then((r) => r.text()),
      fetch(`${base}/fruitfly.xml`).then((r) => r.text()),
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
          getOrFetch(file, `${base}/${file}`)
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

    // Cache the 59 actuators the trained walking policy controls,
    // in the order it expects (= flybody walk_imitation env's
    // _get_actuators output). Inferred from MJCF declaration order:
    // 3 head joints + 2 abdomen + 48 leg motors + 6 adhesion claws.
    const walkingActuatorNames = [
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
    p.walkingActuatorIds = walkingActuatorNames.map(id);
    const goodWalk = p.walkingActuatorIds.filter((i) => i >= 0).length;
    if (goodWalk !== 59) {
      console.warn(`walking actuator lookup: ${goodWalk}/59 resolved`);
    }

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

    // Cache body IDs for the 7 appendages (6 tarsi + head) so the
    // appendages_pos observable can read their xpos in egocentric
    // (root-relative, root-rotated) frame.
    const bId = (n: string) => p.mujoco.mj_name2id(p.model, p.mujoco.mjtObj.mjOBJ_BODY.value, n);
    p.appendageBodyIds = [
      "tarsus_T1_left", "tarsus_T1_right",
      "tarsus_T2_left", "tarsus_T2_right",
      "tarsus_T3_left", "tarsus_T3_right",
      "head",
    ].map(bId);
    p.thoraxBodyId = bId("thorax");

    onProgress?.(`flybody ready (${p.model.nbody} bodies, ${nmesh} meshes, wings=${p.wingActs ? "ok" : "missing"}, legs=${goodLegs}/6, walk-acts=${goodWalk}/59)`);
    return p;
  }

  /** 59 actuator IDs the trained walking policy controls, in policy
   * action-vector order. Filled in `create()`. */
  walkingActuatorIds: number[] = [];
  /** Sensor adr + dim cache used by the trained-walker observation
   * builder. */
  sensorIdx: {
    accel: { adr: number; dim: number } | null;
    gyro:  { adr: number; dim: number } | null;
    vel:   { adr: number; dim: number } | null;
    forces: ({ adr: number; dim: number } | null)[];
    touches: ({ adr: number; dim: number } | null)[];
  } = { accel: null, gyro: null, vel: null, forces: [], touches: [] };
  /** Body IDs of the 7 appendages whose egocentric xpos forms the
   * appendages_pos observable. */
  appendageBodyIds: number[] = [];
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

  /** Step physics N times.
   *
   * Body is driven by leg/wing actuators (visible in the leg motion
   * you can see), PLUS a soft kinematic assist on the freejoint
   * scaled by motor command. The leg actuators alone don't produce
   * enough thrust in browser mujoco_wasm to make walking visible at
   * normal viewing scale; the kinematic assist picks up that slack.
   * Crucially the assist is PROPORTIONAL to the brain's motor command
   * — when no DN fires, the term is zero and the body sits.
   *
   * Stabilizer: pitch/roll angular damper (×0.85 per substep) keeps
   * the body upright without pinning orientation; the fly can still
   * tip if it genuinely loses balance. */
  step(substeps = 1) {
    const hasCmd = Math.abs(this.fwdCmd) > 0.01 || Math.abs(this.turnCmd) > 0.01;
    for (let s = 0; s < substeps; s++) {
      const qpos = this.data.qpos as Float64Array;
      const qvel = this.data.qvel as Float64Array;
      if (qvel && qvel.length >= 6) {
        qvel[3] *= 0.85;   // pitch damping
        qvel[4] *= 0.85;   // roll damping
      }
      if (hasCmd && qpos && qvel && qpos.length >= 7) {
        // World-frame heading: rotate body +x (head direction in
        // flybody's MJCF) by the freejoint quaternion.
        const qw = qpos[3], qx = qpos[4], qy = qpos[5], qz = qpos[6];
        const fx = 1 - 2 * (qy * qy + qz * qz);
        const fy = 2 * (qx * qy + qw * qz);
        const v = 1.0 * this.fwdCmd;  // 1 cm/s per unit fwd command
        qvel[0] = fx * v;
        qvel[1] = fy * v;
        // Yaw gain calibrated so a closed-loop sweep at turn=±0.5
        // can find the target inside one camera-tick window. With
        // mujoco dt=0.1ms × 32 substeps × ~15 RAF/sec = ~48 ms sim
        // per closed-loop tick, gain of 6 gives ~17°/tick at |turn|=0.5.
        //
        // Sign: turnCmd > 0 means "target on fly's left, turn LEFT to
        // face it" (matches retinalSample()'s positive-angle = left
        // convention and vnc.ts's visual reflex). In MuJoCo z-up,
        // counter-clockwise yaw (looking down) = left turn = positive
        // qvel[5]. Old code had this negated which produced a
        // bug-feedback loop: fly saw target on left, turned right,
        // target moved further left, reflex got stronger, fly spun
        // away from target.
        qvel[5] = this.turnCmd * 6.0;
      }
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
    let off = 0;
    if (this.sensorIdx.accel) {
      const { adr } = this.sensorIdx.accel;
      obs[off + 0] = sensordata[adr + 0];
      obs[off + 1] = sensordata[adr + 1];
      obs[off + 2] = sensordata[adr + 2];
    }
    off += 3;
    // ---- 3..61: actuator_activation (59) ------------------------------
    for (let i = 0; i < 59; i++) {
      const a = this.walkingActuatorIds[i];
      // act[] is empty for stateless actuators; fall back to ctrl in
      // that case so we always have a 59-dim history signal.
      const v = (act && act.length > a && a >= 0) ? act[a] : 0;
      obs[off + i] = v;
    }
    off += 59;
    // ---- 62..82: appendages_pos (7×3 egocentric) ---------------------
    if (this.thoraxBodyId >= 0) {
      const tx = xpos[3 * this.thoraxBodyId + 0];
      const ty = xpos[3 * this.thoraxBodyId + 1];
      const tz = xpos[3 * this.thoraxBodyId + 2];
      // Thorax world rotation matrix (3×3, row-major in xmat[9*body]).
      const m = 9 * this.thoraxBodyId;
      const r00 = xmat[m + 0], r01 = xmat[m + 1], r02 = xmat[m + 2];
      const r10 = xmat[m + 3], r11 = xmat[m + 4], r12 = xmat[m + 5];
      const r20 = xmat[m + 6], r21 = xmat[m + 7], r22 = xmat[m + 8];
      for (let i = 0; i < 7; i++) {
        const b = this.appendageBodyIds[i];
        if (b < 0) { obs[off + 3 * i] = obs[off + 3 * i + 1] = obs[off + 3 * i + 2] = 0; continue; }
        const dx = xpos[3 * b + 0] - tx;
        const dy = xpos[3 * b + 1] - ty;
        const dz = xpos[3 * b + 2] - tz;
        // Rotate world delta into thorax frame: M^T · d.
        obs[off + 3 * i + 0] = r00 * dx + r10 * dy + r20 * dz;
        obs[off + 3 * i + 1] = r01 * dx + r11 * dy + r21 * dz;
        obs[off + 3 * i + 2] = r02 * dx + r12 * dy + r22 * dz;
      }
    }
    off += 21;
    // ---- 83..100: force (6×3) ----------------------------------------
    for (let i = 0; i < 6; i++) {
      const s = this.sensorIdx.forces[i];
      if (s) {
        obs[off + 3 * i + 0] = sensordata[s.adr + 0];
        obs[off + 3 * i + 1] = sensordata[s.adr + 1];
        obs[off + 3 * i + 2] = sensordata[s.adr + 2];
      }
    }
    off += 18;
    // ---- 101..103: gyro ---------------------------------------------
    if (this.sensorIdx.gyro) {
      const { adr } = this.sensorIdx.gyro;
      obs[off + 0] = sensordata[adr + 0];
      obs[off + 1] = sensordata[adr + 1];
      obs[off + 2] = sensordata[adr + 2];
    }
    off += 3;
    // ---- 104..188: joints_pos (85) ----------------------------------
    // qpos[0..6] is the freejoint (xyz + quat). Joint values start at 7.
    // Walking policy expects 85 joint qpos values in MJCF declaration
    // order. flybody has more than 85 joints total, but the walker
    // env config likely truncates to walking-relevant (no wing/antenna).
    for (let i = 0; i < 85; i++) {
      obs[off + i] = qpos.length > 7 + i ? qpos[7 + i] : 0;
    }
    off += 85;
    // ---- 189..273: joints_vel (85) ----------------------------------
    for (let i = 0; i < 85; i++) {
      obs[off + i] = qvel.length > 6 + i ? qvel[6 + i] : 0;
    }
    off += 85;
    // ---- 274..468: ref_displacement (65×3) ---------------------------
    // Synthetic forward trajectory: each future frame advances along
    // the body's heading at targetSpeedCmPerS for 1/50 sec.
    const dtFrame = 1 / 50;
    const stepCm = targetSpeedCmPerS * dtFrame;
    for (let f = 0; f < 65; f++) {
      // Future frames in body-local frame: forward = +x, lateral = 0,
      // vertical = 0. Reference is body-relative so heading rotation
      // doesn't enter.
      obs[off + 3 * f + 0] = (f + 1) * stepCm;
      obs[off + 3 * f + 1] = 0;
      obs[off + 3 * f + 2] = 0;
    }
    off += 195;
    // ---- 469..728: ref_root_quat (65×4) ------------------------------
    // Identity quaternion for each frame — fly doesn't rotate during
    // straight walking. (qw, qx, qy, qz) order matches MJ.
    for (let f = 0; f < 65; f++) {
      obs[off + 4 * f + 0] = 1;
      obs[off + 4 * f + 1] = 0;
      obs[off + 4 * f + 2] = 0;
      obs[off + 4 * f + 3] = 0;
    }
    off += 260;
    // ---- 729..734: touch (6) -----------------------------------------
    for (let i = 0; i < 6; i++) {
      const s = this.sensorIdx.touches[i];
      obs[off + i] = s ? sensordata[s.adr] : 0;
    }
    off += 6;
    // ---- 735..737: velocimeter --------------------------------------
    if (this.sensorIdx.vel) {
      const { adr } = this.sensorIdx.vel;
      obs[off + 0] = sensordata[adr + 0];
      obs[off + 1] = sensordata[adr + 1];
      obs[off + 2] = sensordata[adr + 2];
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
   * corresponding 59 actuators. Caller is responsible for: (a) running
   * the policy forward pass, (b) ensuring actions are bounded
   * (the trained policy outputs roughly [-3, 3] mean for in-distribution
   * obs; ctrl ranges in MJCF clamp anyway, but we cap to ±1 here as a
   * safety net since OOD inputs can produce huge outputs).
   */
  applyTrainedWalkerActions(actions: Float32Array): void {
    if (actions.length !== 59) throw new Error(`actions must be 59-dim, got ${actions.length}`);
    const ctrl = this.data.ctrl as Float64Array;
    for (let i = 0; i < 59; i++) {
      const a = this.walkingActuatorIds[i];
      if (a < 0) continue;
      // Clamp to ±1 to absorb OOD spikes; the env's ctrl range
      // (often narrower) clamps further.
      const v = actions[i];
      ctrl[a] = v > 1 ? 1 : v < -1 ? -1 : v;
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
