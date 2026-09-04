// mjcfpatch.test.ts — pure-string unit checks for patchActuatorFilters.
// No MuJoCo WASM or browser runtime required.

import { test } from "node:test";
import assert from "node:assert/strict";

import { patchActuatorFilters } from "../src/mjcfPatch.ts";

function makeXml(opts: { nJoints: number; includeAdhesion: boolean }): string {
  const joints = Array.from(
    { length: opts.nJoints },
    (_, i) => `  <general name="joint_${i}" joint="joint_${i}" gear="1 0 0 0 0 0"/>`,
  );
  const adhesion = opts.includeAdhesion
    ? '  <general dyntype="none" dynprm="1"/>'
    : '';
  return [
    '<mujoco>',
    '  <actuator>',
    ...joints,
    adhesion,
    '  </actuator>',
    '</mujoco>',
  ].join('\n');
}

function countOccurrences(haystack: string, needle: string): number {
  let count = 0;
  let pos = 0;
  while ((pos = haystack.indexOf(needle, pos)) !== -1) {
    count++;
    pos += needle.length;
  }
  return count;
}

test("patches 70 joint actuators and the adhesion default", () => {
  const xml = makeXml({ nJoints: 70, includeAdhesion: true });
  const patched = patchActuatorFilters(xml);

  assert.equal(countOccurrences(patched, 'dyntype="filter" dynprm="0.01"'), 70);
  assert.equal(countOccurrences(patched, 'dyntype="filter" dynprm="0.007"'), 1);
  assert.equal(countOccurrences(patched, 'dyntype="none"'), 0);
});

test("throws on wrong joint actuator count", () => {
  const xml = makeXml({ nJoints: 69, includeAdhesion: true });
  assert.throws(() => patchActuatorFilters(xml), /70|joint actuators/);
});

test("throws on missing adhesion default", () => {
  const xml = makeXml({ nJoints: 70, includeAdhesion: false });
  assert.throws(() => patchActuatorFilters(xml), /70|joint actuators/);
});
