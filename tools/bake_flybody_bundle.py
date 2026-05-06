#!/usr/bin/env python3
"""bake_flybody_bundle.py — concat all flybody assets (XMLs + 85 OBJs)
into a single binary so the runtime can fetch + cache + parse all of
them with ONE IndexedDB transaction.

Why: the per-file IDB cache pattern in src/cache.ts works correctly
but each transaction has ~300-1000 ms of overhead — instrumented test
runs show 85 OBJ files taking ~34 s of cumulative IDB time even when
every read is a cache HIT. Going through one bundle file collapses
that to a single ~500 ms transaction (a single big ArrayBuffer is
much faster than 85 small ones).

Format (little-endian):
  Header 16 B:
    magic     char[8]  = "WGFLYBND"
    version   u32      = 1
    n_files   u32

  Manifest n_files × 16 B:
    name_off  u32      offset into names section
    name_len  u32      bytes
    data_off  u32      offset into data section
    data_len  u32      bytes

  Names section: concatenated UTF-8 names, no separators (use name_len)
  Data section : concatenated file bytes (4-byte aligned)

Output: public/flybody.bundle.bin (~149 MB).
"""
from __future__ import annotations

import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "public" / "flybody"
OUT = ROOT / "public" / "flybody.bundle.bin"

MAGIC = b"WGFLYBND"


def main() -> int:
    if not SRC.is_dir():
        raise SystemExit(f"missing {SRC}; populate from TuragaLab/flybody first")

    # Collect XML + OBJ in stable order so the bundle is deterministic.
    xmls = sorted(SRC.glob("*.xml"))
    objs = sorted(SRC.glob("*.obj"))
    files = xmls + objs
    if not files:
        raise SystemExit(f"no XML/OBJ files in {SRC}")

    print(f"  bundling {len(xmls)} XMLs + {len(objs)} OBJs")

    # Build names + data buffers.
    names = bytearray()
    data = bytearray()
    name_offs: list[tuple[int, int, int, int]] = []
    for p in files:
        name_b = p.name.encode("utf-8")
        # 4-byte align data starts (some downstream parsers prefer it).
        pad = (-len(data)) & 3
        if pad:
            data.extend(b"\x00" * pad)
        n_off, n_len = len(names), len(name_b)
        d_off = len(data)
        bytes_ = p.read_bytes()
        d_len = len(bytes_)
        names.extend(name_b)
        data.extend(bytes_)
        name_offs.append((n_off, n_len, d_off, d_len))

    n_files = len(files)
    HEADER_BYTES = 16
    MANIFEST_BYTES = n_files * 16
    NAMES_OFFSET = HEADER_BYTES + MANIFEST_BYTES
    NAMES_BYTES = len(names)
    # 4-byte align names section end.
    NAMES_BYTES_PADDED = (NAMES_BYTES + 3) & ~3
    DATA_SECTION_OFFSET = NAMES_OFFSET + NAMES_BYTES_PADDED

    # Now rewrite per-entry data_off so it's an ABSOLUTE file offset.
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<II", 1, n_files))
        assert f.tell() == HEADER_BYTES
        for (n_off, n_len, d_off, d_len) in name_offs:
            # name_off is relative to names section start; data_off is
            # absolute file offset (NAMES_OFFSET + name_off → name).
            f.write(struct.pack("<IIII", n_off, n_len,
                                DATA_SECTION_OFFSET + d_off, d_len))
        assert f.tell() == NAMES_OFFSET
        f.write(bytes(names))
        if NAMES_BYTES_PADDED > NAMES_BYTES:
            f.write(b"\x00" * (NAMES_BYTES_PADDED - NAMES_BYTES))
        assert f.tell() == DATA_SECTION_OFFSET
        f.write(bytes(data))

    size_mb = OUT.stat().st_size / 1e6
    print(f"\nwrote {OUT.relative_to(ROOT)}  ({size_mb:.1f} MB)")
    print(f"  header={HEADER_BYTES} B, manifest={MANIFEST_BYTES} B, "
          f"names={NAMES_BYTES} B, data={len(data)} B")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
