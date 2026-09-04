// cache.test.ts — pure-CPU unit checks for progressText without a GPU or
// browser runtime. Node's native TypeScript stripping handles the .ts file.
//
// Sits in tests-unit/ so playwright's testDir ("./tests") does not collect it.

import { test } from "node:test";
import assert from "node:assert/strict";

import { progressText } from "../src/cache.ts";

test("progressText with unknown total shows only megabytes", () => {
  assert.equal(progressText(0, 0), "0.0 MB");
  assert.equal(progressText(42e6, 0), "42.0 MB");
});

test("progressText with known total shows fraction and percentage", () => {
  assert.equal(progressText(50e6, 100e6), "50.0 / 100.0 MB (50%)");
  assert.equal(progressText(1e6, 3e6), "1.0 / 3.0 MB (33%)");
});

test("progressText is unclamped when got exceeds total", () => {
  // The helper reports the raw ratio; callers are expected to drive it with
  // monotonically increasing byte counts that stop at Content-Length.
  assert.equal(progressText(125e6, 100e6), "125.0 / 100.0 MB (125%)");
});
