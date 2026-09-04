// walkingref.test.ts — pure-CPU unit checks for parseWalkingRef without a
// GPU or browser runtime. Node's native TypeScript stripping handles the
// .ts file.
//
// Sits in tests-unit/ so playwright's testDir ("./tests") does not collect it.
//
// We build synthetic ArrayBuffers because parseWalkingRef is the real
// contract: header `char[8] magic, u32 version, u32 n_frames, f32 dt,
// u32[2] reserved` followed by f32[n_frames, 7] qpos.

import { test } from "node:test";
import assert from "node:assert/strict";

import { parseWalkingRef, type WalkingRef } from "../src/walking-ref.ts";

function makeWalkingRefBuffer(n: number, qpos: number[], dt = 0.5): ArrayBuffer {
  // Exact layout: 8 magic + 4 version + 4 n_frames + 4 dt + 12 reserved
  // + f32[n, 7] qpos.
  const total = 32 + n * 7 * 4;
  assert.equal(qpos.length, n * 7, "qpos length must match n*7");

  const buf = new ArrayBuffer(total);
  const dv = new DataView(buf);

  for (let i = 0; i < 8; i++) {
    dv.setUint8(i, "WGFLYREF".charCodeAt(i));
  }
  dv.setUint32(8, 1, true); // version
  dv.setUint32(12, n, true); // n_frames
  dv.setFloat32(16, dt, true); // dt
  dv.setUint32(20, 0, true); // reserved
  dv.setUint32(24, 0, true); // reserved
  dv.setUint32(28, 0, true); // reserved

  let off = 32;
  for (const v of qpos) {
    dv.setFloat32(off, v, true);
    off += 4;
  }

  return buf;
}

function assertValidWalkingRef2(r: WalkingRef) {
  assert.equal(r.numFrames, 2);
  assert.equal(r.dt, 0.5);
  assert.equal(r.qpos.length, 14);
  assert.deepEqual(
    r.qpos,
    new Float32Array([
      1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0,
      4.0, 5.0, 6.0, 0.70710677, 0.0, 0.70710677, 0.0,
    ]),
  );
}

test("parses a valid n=2 walking-ref buffer", () => {
  const buf = makeWalkingRefBuffer(2, [
    1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0,
    4.0, 5.0, 6.0, 0.70710677, 0.0, 0.70710677, 0.0,
  ]);
  const r = parseWalkingRef(buf);
  assertValidWalkingRef2(r);
});

test("throws on bad magic", () => {
  const buf = makeWalkingRefBuffer(0, []);
  const dv = new DataView(buf);
  dv.setUint8(0, "X".charCodeAt(0));
  assert.throws(() => parseWalkingRef(buf), /bad magic/);
});

test("throws on unsupported version", () => {
  const buf = makeWalkingRefBuffer(0, []);
  const dv = new DataView(buf);
  dv.setUint32(8, 2, true);
  assert.throws(() => parseWalkingRef(buf), /unsupported walking-ref version 2/);
});

test("throws on truncated buffer", () => {
  // n claims 2 but file is missing the last float.
  const buf = makeWalkingRefBuffer(2, [
    1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0,
    4.0, 5.0, 6.0, 0.70710677, 0.0, 0.70710677, 0.0,
  ]).slice(0, 32 + 2 * 7 * 4 - 4);
  assert.throws(() => parseWalkingRef(buf), /size mismatch/);
});

test("throws on wrong-size buffer", () => {
  // 33 bytes can never match 32 + n*7*4 for integer n.
  const buf = new ArrayBuffer(33);
  const dv = new DataView(buf);
  for (let i = 0; i < 8; i++) {
    dv.setUint8(i, "WGFLYREF".charCodeAt(i));
  }
  dv.setUint32(8, 1, true);
  dv.setUint32(12, 0, true);
  dv.setFloat32(16, 0.5, true);
  dv.setUint8(32, 0);
  assert.throws(() => parseWalkingRef(buf), /size mismatch/);
});

test("n=0 returns empty qpos array", () => {
  // Header-only file: 32 bytes total. The code views a zero-length array
  // at offset 32 and returns it without throwing.
  const buf = makeWalkingRefBuffer(0, []);
  const r = parseWalkingRef(buf);
  assert.equal(r.numFrames, 0);
  assert.equal(r.dt, 0.5);
  assert.equal(r.qpos.length, 0);
});
