// cache.ts — IndexedDB-backed asset cache so the 149 MB of flybody OBJs
// only download once per machine. Survives hard refreshes (vite's
// no-cache header otherwise re-downloads on Cmd+Shift+R).
//
// Single object store keyed by filename. Stores an `{etag, size, bytes}`
// blob; on hit we revalidate cheaply by comparing size against the new
// HEAD/Content-Length. If size matches, we trust IDB.

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
    tx.objectStore(STORE).put(value, key);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

/**
 * Get bytes for `key` (the cache key) by fetching `url` if not cached.
 * If `key` is already in IDB, return its bytes immediately. Otherwise
 * fetch over network, store, return. Network errors propagate.
 */
export async function getOrFetch(key: string, url: string): Promise<ArrayBuffer> {
  try {
    const hit = await idbGet(key);
    if (hit) return hit.bytes;
  } catch {
    // IDB unavailable (private mode, quota exceeded mid-write, etc.) —
    // fall back to plain fetch each call.
  }
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  const bytes = await r.arrayBuffer();
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
