// physics.ts — minimal MuJoCo runtime that gives the procedural fly a
// real rigid-body chassis. Honest framing: this is NOT flybody MJCF yet.
// It's a planar 3-DoF body (slide x, slide y, yaw) on a floor, driven by
// velocity actuators. Real gravity, real damping, real wall collision —
// enough to retire the kinematic JS integrator and prove the WASM bridge
// works. Swap in the real flybody MJCF once mesh loading is wired.

import loadMujoco from "@mujoco/mujoco";
import type { MainModule, MjModel, MjData } from "@mujoco/mujoco";

const FLY_ARENA_XML = `
<mujoco model="fly_arena">
  <option timestep="0.005" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="40 40 0.1" pos="0 0 0"/>
    <geom name="wall_n" type="box" size="40 0.2 1" pos="0  40 0.5"/>
    <geom name="wall_s" type="box" size="40 0.2 1" pos="0 -40 0.5"/>
    <geom name="wall_e" type="box" size="0.2 40 1" pos=" 40 0 0.5"/>
    <geom name="wall_w" type="box" size="0.2 40 1" pos="-40 0 0.5"/>
    <body name="fly" pos="0 0 0.4">
      <joint name="slide_x" type="slide" axis="1 0 0" damping="0.1"/>
      <joint name="slide_y" type="slide" axis="0 1 0" damping="0.1"/>
      <joint name="yaw"     type="hinge" axis="0 0 1" damping="0.05"/>
      <geom name="body" type="ellipsoid" size="0.6 0.7 0.3" mass="0.05"/>
    </body>
  </worldbody>
  <actuator>
    <velocity name="vx"   joint="slide_x" kv="2.5"/>
    <velocity name="vy"   joint="slide_y" kv="2.5"/>
    <velocity name="vyaw" joint="yaw"     kv="0.8"/>
  </actuator>
</mujoco>
`;

const FORWARD_SPEED_SCALE = 4.0;   // 1.0 cmd → 4 m/s body-frame forward
const TURN_RATE_SCALE     = 2.5;   // 1.0 cmd → 2.5 rad/s yaw

export interface PhysicsPose {
  x: number;
  y: number;
  yaw: number;
}

export class Physics {
  private mujoco!: MainModule;
  private model!: MjModel;
  private data!: MjData;

  static async create(): Promise<Physics> {
    const p = new Physics();
    p.mujoco = await loadMujoco();
    p.model = p.mujoco.MjModel.from_xml_string(FLY_ARENA_XML);
    p.data = new p.mujoco.MjData(p.model);
    return p;
  }

  /**
   * Drive body-frame forward + yaw rate, step physics by `substeps` × dt.
   * Returns the body's planar pose (world frame) for rendering.
   */
  step(forward: number, turn: number, substeps = 4): PhysicsPose {
    const qpos = this.data.qpos as Float64Array;
    const ctrl = this.data.ctrl as Float64Array;
    const yaw = qpos[2];
    // Body-frame forward maps to world via yaw rotation.
    // Body +y is "forward"; rotated by yaw around +z, world dx = -sin(yaw)*v, dy = cos(yaw)*v.
    // Body forward = local -y in MJ (so that the procedural fly's head,
    // which sits at TJ +z = MJ -y after the axis swap, leads the motion).
    // World velocity = R(yaw) · (0, -v, 0) = (sin(yaw)·v, -cos(yaw)·v).
    const v = forward * FORWARD_SPEED_SCALE;
    ctrl[0] =  Math.sin(yaw) * v;
    ctrl[1] = -Math.cos(yaw) * v;
    ctrl[2] = turn * TURN_RATE_SCALE;
    for (let s = 0; s < substeps; s++) {
      this.mujoco.mj_step(this.model, this.data);
    }
    return { x: qpos[0], y: qpos[1], yaw: qpos[2] };
  }

  /** Hard-reset pose + velocity to origin. */
  reset() {
    this.mujoco.mj_resetData(this.model, this.data);
  }

  dispose() {
    this.data.delete();
    this.model.delete();
  }
}
