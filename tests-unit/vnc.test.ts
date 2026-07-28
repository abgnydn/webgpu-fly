// vnc.test.ts — pure-CPU unit checks that run without a GPU, so CI has
// something that actually validates behaviour (the playwright suite needs
// WebGPU + ~1 GB of assets and cannot run on a hosted runner).
//
// Sits in tests-unit/ rather than tests/ so playwright's testDir ("./tests")
// does not try to collect it.
//
// Assertions are signs and structural invariants only. The LIF constants in
// vnc.ts are emergent dynamics, not a contract — asserting magnitudes would
// turn any legitimate retune into a red build.

import { test } from "node:test";
import assert from "node:assert/strict";

import { motorFromBrain, resetVnc, type MotorContext } from "../src/vnc.ts";
import { WALKING_OBS_TOTAL, obsOffset } from "../src/walking-policy.ts";

// Five DN inputs at fixed indices; everything else silent.
const ctxFor = (dn: string, visual?: { angle: number; area: number }) => {
  const rate = new Float32Array(5);
  const names = ["DNa01", "DNa02", "DNb01", "DNg13", "DNp01"];
  const i = names.indexOf(dn);
  if (i >= 0) rate[i] = 1;
  const ctx: MotorContext = {
    famousDns: { DNa01: [0], DNa02: [1], DNb01: [2], DNg13: [3], DNp01: [4] },
    dnLeft: [],
    dnRight: [],
    visual,
  };
  return { rate, ctx };
};

// sharedVnc is module-level mutable LIF state — without resetVnc() every
// case inherits the previous one's membrane potentials.
const drive = (dn: string, visual?: { angle: number; area: number }) => {
  resetVnc();
  const { rate, ctx } = ctxFor(dn, visual);
  let out = motorFromBrain(rate, ctx);
  for (let i = 0; i < 50; i++) out = motorFromBrain(rate, ctx);
  return out;
};

test("DNa01 drive walks forward", () => {
  assert.ok(drive("DNa01").fwd > 0);
});

test("DNb01 drive walks backward", () => {
  assert.ok(drive("DNb01").fwd < 0);
});

test("visual target to the left turns left", () => {
  assert.ok(drive("", { angle: 0.5, area: 0.01 }).turn > 0);
});

test("visual target to the right turns right", () => {
  assert.ok(drive("", { angle: -0.5, area: 0.01 }).turn < 0);
});

test("walking observation layout matches the trained policy's input", () => {
  assert.equal(WALKING_OBS_TOTAL, 741);
  assert.equal(obsOffset("world_zaxis"), 738);
});
