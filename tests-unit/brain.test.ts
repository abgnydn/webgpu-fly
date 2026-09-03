// brain.test.ts — pure-CPU unit checks for parseBrain without a GPU or
// browser runtime. Node's native TypeScript stripping handles the .ts file.
//
// Sits in tests-unit/ so playwright's testDir ("./tests") does not collect it.
//
// We build synthetic ArrayBuffers rather than mocking fetch, because
// parseBrain is the real contract: magic/version/header, neurons, CSR.
// Signs and structural invariants are asserted; float values are whatever
// the helper writes, so the tests stay stable if default dynamics change.

import { test } from "node:test";
import assert from "node:assert/strict";

import {
  parseBrain,
  MAGIC,
  VNC_MAGIC,
  HEADER_BYTES,
  NEURON_BYTES,
  type Brain,
} from "../src/brain.ts";

function writeString(dv: DataView, off: number, s: string) {
  for (let i = 0; i < s.length; i++) dv.setUint8(off + i, s.charCodeAt(i));
}

interface NeuronSpec {
  pos: [number, number, number];
  sign: number;
  cellType: number;
  superClass: number;
  flags: number;
  ntConf: number;
}

function makeBrainBuffer(
  magic: string,
  version: number,
  neurons: NeuronSpec[],
  rowPtr: number[],
  colIdx: number[],
  weights: number[],
  trailing = 0,
): ArrayBuffer {
  const N = neurons.length;
  const E = colIdx.length;
  assert.equal(rowPtr.length, N + 1, "rowPtr must have N+1 entries");
  assert.equal(colIdx.length, E, "colIdx length must match E");
  assert.equal(weights.length, E, "weights length must match E");

  const total = HEADER_BYTES + N * NEURON_BYTES + (N + 1 + E) * 4 + E * 4 + trailing;
  const buf = new ArrayBuffer(total);
  const dv = new DataView(buf);
  const u8 = new Uint8Array(buf);

  // Header
  writeString(dv, 0, magic);
  dv.setUint32(8, version, true);
  dv.setUint32(12, N, true);
  dv.setUint32(16, E, true);
  dv.setUint32(20, 0, true); // flags
  dv.setFloat32(24, 4.0, true);
  dv.setFloat32(28, 4.0, true);
  dv.setFloat32(32, 40.0, true);
  // reserved bytes 36..63 stay zero

  // Neurons
  let off = HEADER_BYTES;
  for (const n of neurons) {
    dv.setFloat32(off + 0, n.pos[0], true);
    dv.setFloat32(off + 4, n.pos[1], true);
    dv.setFloat32(off + 8, n.pos[2], true);
    dv.setFloat32(off + 12, n.sign, true);
    dv.setUint32(off + 16, n.cellType, true);
    dv.setUint32(off + 20, n.superClass, true);
    dv.setUint32(off + 24, n.flags, true);
    dv.setFloat32(off + 28, n.ntConf, true);
    off += NEURON_BYTES;
  }

  // CSR row_ptr
  for (const r of rowPtr) {
    dv.setUint32(off, r, true);
    off += 4;
  }

  // CSR col_idx
  for (const c of colIdx) {
    dv.setUint32(off, c, true);
    off += 4;
  }

  // CSR weights
  for (const w of weights) {
    dv.setFloat32(off, w, true);
    off += 4;
  }

  return buf;
}

const twoNeurons: NeuronSpec[] = [
  { pos: [100, 200, 300], sign: 1, cellType: 1, superClass: 10, flags: 0, ntConf: 0.9 },
  { pos: [400, 500, 600], sign: -1, cellType: 2, superClass: 20, flags: 1, ntConf: 0.7 },
];

function validBrain(extraTrailing = 0): ArrayBuffer {
  // 2 neurons, 1 edge from neuron 0 -> neuron 1.
  return makeBrainBuffer(
    MAGIC,
    1,
    twoNeurons,
    [0, 0, 1], // rowPtr: neuron 0 has 0 incoming; neuron 1 has 1 incoming at offset 0
    [0],       // colIdx: the single edge comes from neuron 0
    [3.5],     // weight
    extraTrailing,
  );
}

function assertValidBrain(b: Brain) {
  assert.equal(b.header.version, 1);
  assert.equal(b.header.numNeurons, 2);
  assert.equal(b.header.numEdges, 1);
  assert.equal(b.header.flags, 0);
  assert.deepEqual(b.header.voxelToNm, [4.0, 4.0, 40.0]);

  assert.equal(b.neurons.pos.length, 6);
  assert.equal(b.neurons.sign.length, 2);
  assert.equal(b.neurons.cellType.length, 2);
  assert.equal(b.neurons.superClass.length, 2);
  assert.equal(b.neurons.flags.length, 2);
  assert.equal(b.neurons.ntConf.length, 2);

  assert.deepEqual(b.neurons.pos, new Float32Array([100, 200, 300, 400, 500, 600]));
  assert.deepEqual(b.neurons.sign, new Float32Array([1, -1]));
  assert.deepEqual(b.neurons.cellType, new Uint32Array([1, 2]));
  assert.deepEqual(b.neurons.superClass, new Uint32Array([10, 20]));
  assert.deepEqual(b.neurons.flags, new Uint32Array([0, 1]));
  assert.deepEqual(b.neurons.ntConf, new Float32Array([0.9, 0.7]));

  assert.deepEqual(b.rowPtr, new Uint32Array([0, 0, 1]));
  assert.deepEqual(b.colIdx, new Uint32Array([0]));
  assert.deepEqual(b.weight, new Float32Array([3.5]));
}

test("parses a valid 2-neuron 1-edge brain buffer", () => {
  const b = parseBrain(validBrain());
  assertValidBrain(b);
});

test("accepts VNC magic WGFLYVNC", () => {
  // The VNC share the same binary layout; only the magic differs.
  const buf = makeBrainBuffer(
    VNC_MAGIC,
    1,
    twoNeurons,
    [0, 0, 1],
    [0],
    [3.5],
  );
  const b = parseBrain(buf);
  assert.equal(b.header.version, 1);
  assert.equal(b.header.numNeurons, 2);
  assert.equal(b.header.numEdges, 1);
});

test("throws on bad magic", () => {
  const buf = makeBrainBuffer("BADMAGIC", 1, twoNeurons, [0, 0, 1], [0], [3.5]);
  assert.throws(() => parseBrain(buf), /bad magic/);
});

test("throws on unsupported version", () => {
  const buf = makeBrainBuffer(MAGIC, 999, twoNeurons, [0, 0, 1], [0], [3.5]);
  assert.throws(() => parseBrain(buf), /unsupported brain version 999/);
});

test("handles trailing bytes without throwing", () => {
  // Real build artifacts may carry padding or appended metadata; the parser
  // should warn but still return the parsed structure.
  const b = parseBrain(validBrain(8));
  assertValidBrain(b);
});
