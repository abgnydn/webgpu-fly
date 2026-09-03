// brain.ts — parses public/brain.bin into typed views.
// Format spec: tools/build_csr.py docstring (single source of truth).

export const MAGIC = "WGFLYBRN";
export const VNC_MAGIC = "WGFLYVNC";
export const HEADER_BYTES = 64;
export const NEURON_BYTES = 32;

export interface BrainHeader {
  version: number;
  numNeurons: number;
  numEdges: number;
  flags: number;
  voxelToNm: [number, number, number];
}

export interface Neurons {
  pos: Float32Array;        // length 3*N, [x0,y0,z0,x1,...] in nm
  sign: Float32Array;       // length N
  cellType: Uint32Array;    // length N
  superClass: Uint32Array;  // length N
  flags: Uint32Array;       // length N
  ntConf: Float32Array;     // length N
}

export interface Brain {
  header: BrainHeader;
  neurons: Neurons;
  rowPtr: Uint32Array;      // length N+1
  colIdx: Uint32Array;      // length E
  weight: Float32Array;     // length E
}

export async function loadBrain(
  url: string = "/brain.bin",
  onProgress?: (got: number, total: number) => void,
): Promise<Brain> {
  // Route through IDB cache so reloads in the same browser session
  // skip the ~125 MB network fetch (~30s on the dev server). Cache
  // key is the full URL including ?v=<sha> from assets.json, so a
  // new build naturally invalidates the cache (different URL = new
  // entry), and idbPut deletes the previous generation of the same asset.
  const { getOrFetch } = await import("./cache");
  const buf = await getOrFetch(url, url, onProgress);
  return parseBrain(buf);
}

export function parseBrain(buf: ArrayBuffer): Brain {
  const bytes = new Uint8Array(buf);
  const dv = new DataView(buf);

  // --- Header ---
  const magic = String.fromCharCode(...bytes.subarray(0, 8));
  if (magic !== MAGIC && magic !== VNC_MAGIC) {
    throw new Error(`bad magic: got "${magic}", expected "${MAGIC}" or "${VNC_MAGIC}"`);
  }
  const version = dv.getUint32(8, true);
  if (version !== 1) throw new Error(`unsupported brain version ${version}`);
  const numNeurons = dv.getUint32(12, true);
  const numEdges = dv.getUint32(16, true);
  const flags = dv.getUint32(20, true);
  const voxelToNm: [number, number, number] = [
    dv.getFloat32(24, true),
    dv.getFloat32(28, true),
    dv.getFloat32(32, true),
  ];

  let off = HEADER_BYTES;

  // --- Neurons (struct of arrays for GPU friendliness) ---
  const pos = new Float32Array(numNeurons * 3);
  const sign = new Float32Array(numNeurons);
  const cellType = new Uint32Array(numNeurons);
  const superClass = new Uint32Array(numNeurons);
  const flagsArr = new Uint32Array(numNeurons);
  const ntConf = new Float32Array(numNeurons);

  for (let i = 0; i < numNeurons; i++) {
    const base = off + i * NEURON_BYTES;
    pos[3 * i + 0] = dv.getFloat32(base + 0, true);
    pos[3 * i + 1] = dv.getFloat32(base + 4, true);
    pos[3 * i + 2] = dv.getFloat32(base + 8, true);
    sign[i] = dv.getFloat32(base + 12, true);
    cellType[i] = dv.getUint32(base + 16, true);
    superClass[i] = dv.getUint32(base + 20, true);
    flagsArr[i] = dv.getUint32(base + 24, true);
    ntConf[i] = dv.getFloat32(base + 28, true);
  }
  off += numNeurons * NEURON_BYTES;

  // --- CSR row_ptr (N+1 u32) ---
  const rowPtr = new Uint32Array(buf, off, numNeurons + 1).slice();
  off += (numNeurons + 1) * 4;

  // --- CSR col_idx (E u32) ---
  const colIdx = new Uint32Array(buf, off, numEdges).slice();
  off += numEdges * 4;

  // --- CSR weight (E f32) ---
  const weight = new Float32Array(buf, off, numEdges).slice();
  off += numEdges * 4;

  if (off !== buf.byteLength) {
    console.warn(`brain.bin: ${buf.byteLength - off} trailing bytes ignored`);
  }

  return {
    header: { version, numNeurons, numEdges, flags, voxelToNm },
    neurons: { pos, sign, cellType, superClass, flags: flagsArr, ntConf },
    rowPtr,
    colIdx,
    weight,
  };
}
