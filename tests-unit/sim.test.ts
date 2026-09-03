// sim.test.ts — pure-CPU checks for simulation parameter validation.
//
// These tests import from src/simParams.ts rather than src/sim.ts because
// sim.ts pulls in .wgsl?raw shader imports that Node cannot resolve.

import { test } from "node:test";
import assert from "node:assert/strict";

import { DEFAULT_PARAMS, assertValidDt } from "../src/simParams.ts";

test("DEFAULT_PARAMS.dtMs === 1 passes the guard", () => {
  assert.equal(DEFAULT_PARAMS.dtMs, 1.0);
  assert.doesNotThrow(() => assertValidDt(DEFAULT_PARAMS));
});

test("dtMs = 0.5 throws", () => {
  assert.throws(
    () => assertValidDt({ ...DEFAULT_PARAMS, dtMs: 0.5 }),
    /lif\.wgsl A_SYN is compiled for dt=1ms/
  );
});

test("dtMs = 2.0 throws", () => {
  assert.throws(
    () => assertValidDt({ ...DEFAULT_PARAMS, dtMs: 2.0 }),
    /lif\.wgsl A_SYN is compiled for dt=1ms/
  );
});
