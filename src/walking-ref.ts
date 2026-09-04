// walking-ref.ts — load the baked real-fly walking reference trajectory
// from public/walking-ref.bin (produced by tools/bake_walking_ref.py
// from the Vaxenburg 2025 mocap dataset). Replaces the synthetic
// straight-line ref currently fed to the trained walker policy.
//
// Format spec lives in tools/bake_walking_ref.py docstring. Header is
// 32 B, data is N × 7 × f32 (root_qpos: x, y, z, qw, qx, qy, qz).

const MAGIC = "WGFLYREF";

export interface WalkingRef {
  /** root qpos, length N×7. Layout per frame: [x, y, z, qw, qx, qy, qz]. */
  qpos: Float32Array;
  /** Frames in the trajectory. */
  numFrames: number;
  /** Per-frame timestep in seconds. */
  dt: number;
}

/**
 * Parse a baked walking-ref binary buffer. Format spec lives in
 * tools/bake_walking_ref.py docstring.
 */
export function parseWalkingRef(buf: ArrayBuffer): WalkingRef {
  const dv = new DataView(buf);
  const magic = String.fromCharCode(...new Uint8Array(buf, 0, 8));
  if (magic !== MAGIC) throw new Error(`bad magic: got "${magic}", expected "${MAGIC}"`);
  const version = dv.getUint32(8, true);
  if (version !== 1) throw new Error(`unsupported walking-ref version ${version}`);
  const numFrames = dv.getUint32(12, true);
  const dt = dv.getFloat32(16, true);
  // header is 32 B (8 magic + 4 version + 4 frames + 4 dt + 12 reserved)
  const HEADER_BYTES = 32;
  const expected = HEADER_BYTES + numFrames * 7 * 4;
  if (buf.byteLength !== expected) {
    throw new Error(`walking-ref size mismatch: header N=${numFrames}, file ${buf.byteLength} bytes (expected ${expected})`);
  }
  const qpos = new Float32Array(buf, HEADER_BYTES, numFrames * 7);
  return { qpos, numFrames, dt };
}

export async function loadWalkingRef(
  url: string = (import.meta.env.VITE_WALKING_REF_URL ?? "/walking-ref.bin"),
): Promise<WalkingRef> {
  const { getOrFetch } = await import("./cache");
  const buf = await getOrFetch(url, url);
  return parseWalkingRef(buf);
}
