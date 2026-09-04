// evolution.test.ts — pure-CPU checks for gait-policy decoding.
//
// These tests import from src/evolutionParams.ts rather than src/evolution.ts
// because evolution.ts pulls in .wgsl?raw shader imports that Node cannot
// resolve.

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import { POLICY_DIM, decode } from "../src/evolutionParams.ts";

test("POLICY_DIM is 8", () => {
  assert.equal(POLICY_DIM, 8);
});

test("decode zeros input returns the seeded baseline", () => {
  const gait = decode(new Float32Array(POLICY_DIM));
  assert.deepEqual(gait, {
    freq: 5.0,
    coxaAmp: 0,
    femurAmp: 0,
    tibiaAmp: 0,
    swingRatio: 0.1,
    liftOffset: 0,
    strideGain: 0,
    dragCoeff: 0,
  });
});

test("decode saturates large positive inputs", () => {
  const gait = decode(new Float32Array(POLICY_DIM).fill(10));
  assert.deepEqual(gait, {
    freq: 5.0 * Math.exp(1.5),
    coxaAmp: 1.5,
    femurAmp: 1.5,
    tibiaAmp: 1.5,
    swingRatio: 0.9,
    liftOffset: 1,
    strideGain: 2,
    dragCoeff: 1,
  });
});

test("decode clamps large negative inputs", () => {
  const gait = decode(new Float32Array(POLICY_DIM).fill(-10));
  assert.deepEqual(gait, {
    freq: 5.0 * Math.exp(-1.5),
    coxaAmp: 0,
    femurAmp: 0,
    tibiaAmp: 0,
    swingRatio: 0.1,
    liftOffset: 0,
    strideGain: 0,
    dragCoeff: 0,
  });
});

test("POLICY_DIM matches src/shaders/evolve.wgsl:13", () => {
  // npm run test:unit runs from the repo root, so the relative URL resolves
  // to src/shaders/evolve.wgsl.
  const source = readFileSync(
    new URL("../src/shaders/evolve.wgsl", import.meta.url),
    "utf8",
  );
  const match = source.match(/const POLICY_DIM\s*:\s*u32\s*=\s*(\d+)u/);
  assert.ok(match, "POLICY_DIM constant not found in evolve.wgsl");
  assert.equal(Number(match[1]), POLICY_DIM);
});
