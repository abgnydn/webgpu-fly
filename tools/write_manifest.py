#!/usr/bin/env python3
"""write_manifest.py — hash every asset under public/ that the runtime
loads, write public/assets.json mapping asset path → 12-char sha256.

The runtime fetches assets.json first (always no-cache) then appends
`?v=<sha>` to every binary URL. The browser cache key is the full URL
including query string, so:

  - same build  → same sha → cache hit → no network for big binaries
  - new build   → new sha  → fresh fetch (cache transparently invalidated)

Combined with `Cache-Control: public, max-age=31536000, immutable` on the
binaries (set by tools/upload_to_r2.sh), this gives permanent browser
caching with automatic invalidation on rebuild — no IDB, no service worker.

Run after any of: build_csr.py, build_vnc.py, extract_walking_policy.py,
dump_flybody_spec.py. Cheap (a few sha256s over ~100 MB), idempotent.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PUBLIC = ROOT / "public"
OUT = PUBLIC / "assets.json"

# Files whose URLs should be version-stamped at runtime. Anything not
# listed here loads with no cache busting (acceptable for tiny static
# assets like index.html itself, served by the host with its own
# cache headers).
TRACKED = [
    "brain.bin",
    "brain.meta.json",
    "vnc.bin",
    "vnc.meta.json",
    "walking-policy.bin",
    "walking-obs-norm.bin",
    "walking-ref.bin",
]


def sha12(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def main() -> int:
    manifest: dict[str, str] = {}
    missing: list[str] = []
    for name in TRACKED:
        p = PUBLIC / name
        if not p.exists():
            missing.append(name)
            continue
        manifest[name] = sha12(p)
        print(f"  {name:30}  {manifest[name]}  ({p.stat().st_size / 1e6:.1f} MB)")

    OUT.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {OUT.relative_to(ROOT)} ({len(manifest)} entries)")
    if missing:
        print(f"  skipped {len(missing)} missing: {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
