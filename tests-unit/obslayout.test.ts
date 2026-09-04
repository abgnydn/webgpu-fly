// obslayout.test.ts — pure-CPU checks for WALKING_OBS_LAYOUT contract.
// Node's native TypeScript stripping handles the .ts file.
//
// Sits in tests-unit/ so playwright's testDir ("./tests") does not collect it.

import { test } from "node:test";
import assert from "node:assert/strict";

import { WALKING_OBS_LAYOUT, WALKING_OBS_TOTAL, obsOffset } from "../src/walking-policy.ts";

const names = WALKING_OBS_LAYOUT.map((o) => o.name);

function shapeProduct(shape: readonly number[]): number {
  return shape.reduce((p, s) => p * s, 1);
}

test("every entry's dim equals its shape product", () => {
  for (const o of WALKING_OBS_LAYOUT) {
    assert.equal(
      o.dim,
      shapeProduct(o.shape),
      `${o.name}: dim ${o.dim} !== shape product ${shapeProduct(o.shape)}`,
    );
  }
});

test("all observable names are unique", () => {
  const unique = new Set(names);
  assert.equal(unique.size, names.length);
});

test("observable names are sorted alphabetically", () => {
  assert.deepEqual(names, [...names].sort());
});

test("WALKING_OBS_TOTAL equals 741 and matches manual layout sum", () => {
  const manualSum = WALKING_OBS_LAYOUT.reduce((s, o) => s + o.dim, 0);
  assert.equal(WALKING_OBS_TOTAL, 741);
  assert.equal(manualSum, 741);
  assert.equal(WALKING_OBS_TOTAL, manualSum);
});

test("obsOffset spot checks and cumulative consistency", () => {
  assert.equal(obsOffset("accelerometer"), 0);
  assert.equal(obsOffset("world_zaxis"), 738);

  let cumulative = 0;
  for (const o of WALKING_OBS_LAYOUT) {
    assert.equal(
      obsOffset(o.name),
      cumulative,
      `${o.name}: offset ${obsOffset(o.name)} !== cumulative ${cumulative}`,
    );
    cumulative += o.dim;
  }
  assert.equal(cumulative, WALKING_OBS_TOTAL);
});

test("obsOffset throws for an unknown observable name", () => {
  assert.throws(() => obsOffset("not_an_observable"), /unknown observable/);
});
