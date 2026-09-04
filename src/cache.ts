// cache.ts — IndexedDB-backed asset cache so the 149 MB of flybody OBJs
// only download once per machine. Survives hard refreshes (vite's
// no-cache header otherwise re-downloads on Cmd+Shift+R).
//
// Single object store keyed by the full asset URL. Stores an
// `{etag, size, bytes}` blob; on hit we serve the cached bytes directly.
// Invalidation comes from the ?v=<sha> in the key — a new build asks for
// a different key, and idbPut drops the previous generation.

const DB_NAME = "webgpu-fly-cache";
const DB_VERSION = 1;
const STORE = "flybody";

interface Entry {
  etag: string | null;
  size: number;
  bytes: ArrayBuffer;
}

let dbPromise: Promise<IDBDatabase> | null = null;

function openDB(): Promise<IDBDatabase> {
  if (dbPromise) return dbPromise;
  dbPromise = new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE)) db.createObjectStore(STORE);
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
  return dbPromise;
}

async function idbGet(key: string): Promise<Entry | undefined> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, "readonly");
    const req = tx.objectStore(STORE).get(key);
    req.onsuccess = () => resolve(req.result as Entry | undefined);
    req.onerror = () => reject(req.error);
  });
}

async function idbPut(key: string, value: Entry): Promise<void> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, "readwrite");
    const store = tx.objectStore(STORE);
    store.put(value, key);
    // Drop older generations of the same asset — the key is the versioned
    // URL, so a rebuild would otherwise orphan the previous ~140 MB entry
    // forever. IDB evicts whole origins, never individual records.
    const base = key.split("?")[0];
    store.getAllKeys().onsuccess = (e) => {
      for (const k of (e.target as IDBRequest<IDBValidKey[]>).result) {
        if (typeof k === "string" && k !== key && k.split("?")[0] === base) store.delete(k);
      }
    };
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

/** Emit progress at most once per MB — see readWithProgress. */
const PROGRESS_STEP = 1_000_000;

/** "42.0 / 125.7 MB (33%)", or just "42.0 MB" when the length is unknown. */
export function progressText(got: number, total: number): string {
  const mb = (b: number) => (b / 1e6).toFixed(1);
  return total > 0
    ? `${mb(got)} / ${mb(total)} MB (${Math.round((100 * got) / total)}%)`
    : `${mb(got)} MB`;
}

/**
 * Drain `r.body` chunk by chunk so the caller can paint a progress line.
 * Chunks are kept as-is and copied once into a pre-sized buffer — growing
 * a concatenated array per chunk would be quadratic on a 140 MB bundle.
 * The callback is throttled to one call per MB: at chunk granularity it
 * fires thousands of times a second and the DOM writes on the other end
 * measurably slow the download.
 */
async function readWithProgress(
  r: Response,
  onProgress: (got: number, total: number) => void,
): Promise<ArrayBuffer> {
  if (!r.body) throw new Error("Response body is null");
  const len = Number(r.headers.get("Content-Length"));
  // Absent / non-numeric / 0 → report bytes only, never "/ 0 MB".
  const total = Number.isFinite(len) && len > 0 ? len : 0;
  const reader = r.body!.getReader();
  const chunks: Uint8Array[] = [];
  let got = 0;
  let nextEmit = PROGRESS_STEP;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.byteLength;
    if (got >= nextEmit) {
      onProgress(got, total);
      nextEmit = got + PROGRESS_STEP;
    }
  }
  const out = new Uint8Array(got);
  let off = 0;
  for (const c of chunks) {
    out.set(c, off);
    off += c.byteLength;
  }
  onProgress(got, total);
  return out.buffer;
}

/**
 * Get bytes for `key` (the cache key) by fetching `url` if not cached.
 * If `key` is already in IDB, return its bytes immediately. Otherwise
 * fetch over network, store, return. Network errors propagate.
 * `onProgress` (bytes so far, total or 0) streams the download instead of
 * buffering it whole; without it the response is read in one shot.
 */
export async function getOrFetch(
  key: string,
  url: string,
  onProgress?: (got: number, total: number) => void,
): Promise<ArrayBuffer> {
  try {
    const hit = await idbGet(key);
    if (hit) return hit.bytes;
  } catch {
    // IDB unavailable (private mode, quota exceeded mid-write, etc.) —
    // fall back to plain fetch each call.
  }
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  const bytes = onProgress && r.body
    ? await readWithProgress(r, onProgress)
    : await r.arrayBuffer();
  // Fire-and-forget the IDB write — we have the bytes in memory and
  // the caller doesn't need to block on the cache populating. For
  // 125 MB blobs the IDB write is ~30 s and was dominating cold-load
  // wall time.
  idbPut(key, {
    etag: r.headers.get("ETag"),
    size: bytes.byteLength,
    bytes,
  }).catch(() => {
    // Ignore quota / write errors — we already have the bytes.
  });
  return bytes;
}

/** How many entries (and total bytes) are cached. Useful for status logs. */
export async function cacheStats(): Promise<{ count: number; bytes: number }> {
  try {
    const db = await openDB();
    return new Promise((resolve) => {
      const tx = db.transaction(STORE, "readonly");
      const store = tx.objectStore(STORE);
      let count = 0, bytes = 0;
      store.openCursor().onsuccess = (e) => {
        const cur = (e.target as IDBRequest<IDBCursorWithValue>).result;
        if (cur) {
          count++;
          bytes += (cur.value as Entry).size ?? 0;
          cur.continue();
        } else {
          resolve({ count, bytes });
        }
      };
    });
  } catch {
    return { count: 0, bytes: 0 };
  }
}
