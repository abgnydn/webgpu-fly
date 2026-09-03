// vncmeta.test.ts — pure-CPU shape validation for vnc.meta.json.
//
// Mirrors sim.test.ts: imports only from the pure module src/vncMeta.ts so
// Node can run it without DOM or WebGPU dependencies.

import { test } from "node:test";
import assert from "node:assert/strict";

import { assertVncMetaShape, type VncMeta } from "../src/vncMeta.ts";

function validMeta(): VncMeta {
  return {
    dn_inputs: { DNa01: [0, 1], DNb01: [2] },
    motor: {
      "L-F": { all: [10, 11], ti_extensor: [10], ti_flexor: [11] },
      "R-F": { all: [20, 21], ti_extensor: [20], ti_flexor: [21] },
    },
    motor_by_subclass: { ti_extensor: [10, 20], ti_flexor: [11, 21] },
    motor_by_target: { swing: [10, 20], stance: [11, 21] },
    num_neurons: 100,
    num_edges: 200,
  };
}

test("valid meta passes", () => {
  assert.doesNotThrow(() => assertVncMetaShape(validMeta()));
});

test("missing dn_inputs throws /missing/", () => {
  const meta = { ...validMeta(), dn_inputs: undefined };
  assert.throws(() => assertVncMetaShape(meta), /missing dn_inputs/);
});

test("missing motor throws", () => {
  const meta = { ...validMeta(), motor: undefined };
  assert.throws(() => assertVncMetaShape(meta), /missing motor/);
});

test("null input throws", () => {
  assert.throws(() => assertVncMetaShape(null), /missing root object/);
});

test("array input throws", () => {
  assert.throws(() => assertVncMetaShape([]), /missing root object/);
});

test("string input throws", () => {
  assert.throws(() => assertVncMetaShape("not meta"), /missing root object/);
});

test("motor_by_subclass missing throws", () => {
  const meta = { ...validMeta(), motor_by_subclass: undefined };
  assert.throws(() => assertVncMetaShape(meta), /missing motor_by_subclass/);
});

test("motor group that is not an object throws", () => {
  const meta = {
    ...validMeta(),
    motor: { "L-F": { all: [10] }, "R-F": "bad" },
  };
  assert.throws(() => assertVncMetaShape(meta), /motor\.R-F is not an object/);
});
