// obsnorm.test.ts — pure-CPU unit checks for parseObsNorm without a GPU or
// browser runtime. Node's native TypeScript stripping handles the .ts file.
//
// Sits in tests-unit/ so playwright's testDir ("./tests") does not collect it.
//
// We build synthetic ArrayBuffers rather than mocking fetch, because
// parseObsNorm is the real contract: headerless `u32 n, u32 reserved,
// f32[n] mean, f32[n] std`, validated only by exact-size check.

import { test } from "node:test";
import assert from "node:assert/strict";

import { parseObsNorm, type ObsNormStats } from "../src/walking-policy.ts";

function makeObsNormBuffer(n: number, mean: number[], std: number[]): ArrayBuffer {
  // Exact layout: u32 n, u32 reserved, f32[n] mean, f32[n] std.
  const total = 8 + 8 * n;
  assert.equal(mean.length, n, "mean length must match n");
  assert.equal(std.length, n, "std length must match n");

  const buf = new ArrayBuffer(total);
  const dv = new DataView(buf);

  dv.setUint32(0, n, true);
  dv.setUint32(4, 0, true); // reserved

  let off = 8;
  for (const m of mean) {
    dv.setFloat32(off, m, true);
    off += 4;
  }
  for (const s of std) {
    dv.setFloat32(off, s, true);
    off += 4;
  }

  return buf;
}

function assertValidObsNorm2(s: ObsNormStats) {
  assert.equal(s.mean.length, 2);
  assert.equal(s.std.length, 2);
  assert.deepEqual(s.mean, new Float32Array([1.0, 2.0]));
  assert.deepEqual(s.std, new Float32Array([0.5, 1.5]));
}

test("parses a valid n=2 obs-norm buffer", () => {
  const buf = makeObsNormBuffer(2, [1.0, 2.0], [0.5, 1.5]);
  const s = parseObsNorm(buf);
  assertValidObsNorm2(s);
});

test("throws on truncated buffer", () => {
  // n claims 2 but file is missing the second std float.
  const buf = makeObsNormBuffer(2, [1.0, 2.0], [0.5, 1.5]).slice(0, 8 + 8 * 2 - 4);
  assert.throws(() => parseObsNorm(buf), /size mismatch/);
});

test("throws on wrong-size buffer", () => {
  // 9 bytes can never match 8 + 8*n for integer n.
  const buf = new ArrayBuffer(9);
  const dv = new DataView(buf);
  dv.setUint32(0, 0, true);
  dv.setUint32(4, 0, true);
  dv.setUint8(8, 0);
  assert.throws(() => parseObsNorm(buf), /size mismatch/);
});

test("n=0 returns empty mean and std arrays", () => {
  // Header-only file: 8 bytes total. The code views zero-length arrays
  // at offset 8 and returns them without throwing.
  const buf = makeObsNormBuffer(0, [], []);
  const s = parseObsNorm(buf);
  assert.equal(s.mean.length, 0);
  assert.equal(s.std.length, 0);
});
