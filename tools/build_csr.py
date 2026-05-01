#!/usr/bin/env python3
"""
build_csr.py — FlyWire feather + annotations TSV → binary CSR brain blob.

Reads
  data/raw/proofread_connections_783.feather   (aggregated pre,post,weight)
  data/raw/proofread_root_ids_783.npy          (verified neuron list)
  data/raw/flywire_annotations/supplemental_files/Supplemental_file1_neuron_annotations.tsv

Writes
  public/brain.bin       (binary, little-endian)
  public/brain.meta.json (header echo + stats for the runtime to display)

Binary format (authoritative; the WGSL kernel and TS loader must agree):

  [ Header — 64 B, 16-byte aligned ]
    magic         char[8]  = "WGFLYBRN"
    version       u32      = 1
    num_neurons   u32      = N
    num_edges     u32      = E
    flags         u32      bit 0: weights are pre-signed by presynaptic NT
    voxel_to_nm   f32[3]   FAFB14 voxel size in nm (4, 4, 40)
    reserved      u32[5]

  [ Neurons — N × 32 B ]
    pos_x         f32      soma_x or pos_x, in nm
    pos_y         f32
    pos_z         f32
    sign          f32      -1 / 0 / +1 (informational; weights already signed)
    cell_type     u32      packed enum (see CELL_TYPE_TABLE in this file)
    super_class   u32      packed enum (SUPER_CLASS_TABLE)
    flags         u32      bit 0: is_descending  bit 1: is_sensory
    nt_conf       f32      NT prediction confidence in [0, 1]

  [ CSR row_ptr — (N+1) × u32 ]
    row_ptr[i]   = offset into col_idx where neuron i's INCOMING edges start
    row_ptr[N]   = E

  [ CSR col_idx — E × u32 ]
    presynaptic neuron index (0..N-1) for each incoming edge

  [ CSR weight — E × f32 ]
    sign(pre_nt) × synapse_count
    if pre_nt confidence < CONF_THRESHOLD or pre_nt is modulatory, weight = 0

Total ≈ 64 + 32N + 4(N+1) + 8E bytes. For N=140k, E=5M ≈ 45 MB.
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT_BIN = ROOT / "public" / "brain.bin"
OUT_META = ROOT / "public" / "brain.meta.json"

# FAFB14 voxel size in nm
VOXEL_NM = (4.0, 4.0, 40.0)

# Below this NT-prediction confidence we drop the synapse to weight=0
CONF_THRESHOLD = 0.5

NT_SIGN = {
    "acetylcholine": +1.0,  # nAChR — excitatory
    "gaba":          -1.0,  # GABA-A/B — inhibitory
    "glutamate":     -1.0,  # GluCl predominantly inhibitory in fly
    "dopamine":       0.0,  # modulatory — out of scope v1
    "serotonin":      0.0,
    "octopamine":     0.0,
}

SUPER_CLASS_TABLE = [
    "unknown", "sensory", "ascending", "intrinsic", "central",
    "descending", "motor", "endocrine", "visual_centrifugal",
    "visual_projection", "optic",
]
SUPER_CLASS_IDX = {name: i for i, name in enumerate(SUPER_CLASS_TABLE)}

# Cell type enum is large in FlyWire — for v1 we hash the label to a stable u32
# (top 24 bits) and reserve the bottom 8 bits for a few well-known "hero" types
# we want the runtime to highlight cheaply.
HERO_CELL_TYPES = {
    "kenyon": 1,    # mushroom body intrinsic
    "mbon":   2,    # mushroom body output
    "lhn":    3,    # lateral horn
    "pn":     4,    # antennal lobe projection
    "orn":    5,    # olfactory receptor
    "gf":     6,    # giant fiber (escape)
    "dn":     7,    # generic descending
}


def pack_cell_type(label: str | float) -> int:
    if not isinstance(label, str) or not label:
        return 0
    lo = label.lower()
    hero = 0
    for key, idx in HERO_CELL_TYPES.items():
        if key in lo:
            hero = idx
            break
    # FNV-1a on the label, top 24 bits
    h = 2166136261
    for c in label.encode("utf-8"):
        h ^= c
        h = (h * 16777619) & 0xFFFFFFFF
    return ((h & 0xFFFFFF00) | hero) & 0xFFFFFFFF


def main() -> int:
    feather = RAW / "proofread_connections_783.feather"
    root_ids_npy = RAW / "proofread_root_ids_783.npy"
    annotations = RAW / "flywire_annotations" / "supplemental_files" / "Supplemental_file1_neuron_annotations.tsv"

    for p in (feather, root_ids_npy, annotations):
        if not p.exists():
            print(f"missing: {p}", file=sys.stderr)
            print("run: bash tools/download_data.sh", file=sys.stderr)
            return 1

    print(f"[1/5] loading verified root ids from {root_ids_npy.name}")
    root_ids = np.load(root_ids_npy)
    n_neurons = len(root_ids)
    print(f"      {n_neurons:,} verified neurons")

    # neuron_id (u64) -> dense index (u32)
    id_to_idx = {int(r): i for i, r in enumerate(root_ids)}

    print(f"[2/5] loading annotations from {annotations.name}")
    ann = pd.read_csv(annotations, sep="\t", low_memory=False)
    ann = ann.set_index("root_id")
    n_ann_match = sum(1 for r in root_ids if int(r) in ann.index)
    print(f"      {n_ann_match:,} / {n_neurons:,} root_ids matched in annotations")

    print("[3/5] building neuron table")
    neurons = np.zeros(n_neurons, dtype=[
        ("pos_x", "<f4"), ("pos_y", "<f4"), ("pos_z", "<f4"),
        ("sign",  "<f4"),
        ("cell_type",   "<u4"),
        ("super_class", "<u4"),
        ("flags",       "<u4"),
        ("nt_conf",     "<f4"),
    ])

    pre_sign = np.zeros(n_neurons, dtype=np.float32)  # for pre-signing weights

    n_with_soma = n_with_pos = n_with_nt = 0

    for i, root in enumerate(root_ids):
        row = ann.loc[int(root)] if int(root) in ann.index else None
        if row is None:
            continue
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]

        # Position: prefer soma, fall back to synapse-cloud centroid
        sx, sy, sz = row.get("soma_x"), row.get("soma_y"), row.get("soma_z")
        if pd.notna(sx) and pd.notna(sy) and pd.notna(sz):
            neurons[i]["pos_x"] = float(sx) * VOXEL_NM[0]
            neurons[i]["pos_y"] = float(sy) * VOXEL_NM[1]
            neurons[i]["pos_z"] = float(sz) * VOXEL_NM[2]
            n_with_soma += 1
        else:
            px, py, pz = row.get("pos_x"), row.get("pos_y"), row.get("pos_z")
            if pd.notna(px) and pd.notna(py) and pd.notna(pz):
                neurons[i]["pos_x"] = float(px) * VOXEL_NM[0]
                neurons[i]["pos_y"] = float(py) * VOXEL_NM[1]
                neurons[i]["pos_z"] = float(pz) * VOXEL_NM[2]
                n_with_pos += 1

        # Neurotransmitter sign
        nt = row.get("top_nt")
        conf = row.get("top_nt_conf")
        if isinstance(nt, str) and pd.notna(conf) and float(conf) >= CONF_THRESHOLD:
            sign = NT_SIGN.get(nt.lower(), 0.0)
            neurons[i]["sign"] = sign
            neurons[i]["nt_conf"] = float(conf)
            pre_sign[i] = sign
            n_with_nt += 1

        # Class enums
        sc = row.get("super_class")
        if isinstance(sc, str):
            neurons[i]["super_class"] = SUPER_CLASS_IDX.get(sc.lower(), 0)

        ct = row.get("cell_type")
        neurons[i]["cell_type"] = pack_cell_type(ct)

        flags = 0
        if isinstance(sc, str) and sc.lower() == "descending":
            flags |= 1
        if isinstance(sc, str) and sc.lower() in ("sensory", "optic"):
            flags |= 2
        neurons[i]["flags"] = flags

    print(f"      soma pos:  {n_with_soma:,}")
    print(f"      synapse centroid pos: {n_with_pos:,}")
    print(f"      NT-signed: {n_with_nt:,} ({100*n_with_nt/n_neurons:.1f}%)")

    print(f"[4/5] loading edges from {feather.name}")
    edges = pd.read_feather(feather)
    print(f"      raw rows: {len(edges):,}")
    print(f"      columns: {list(edges.columns)}")

    # Column names vary by Zenodo release. Standardise.
    rename_map = {}
    for src in ("pre_pt_root_id", "pre_root_id", "pre"):
        if src in edges.columns:
            rename_map[src] = "pre"
            break
    for src in ("post_pt_root_id", "post_root_id", "post"):
        if src in edges.columns:
            rename_map[src] = "post"
            break
    for src in ("syn_count", "n_syn", "weight", "count"):
        if src in edges.columns:
            rename_map[src] = "weight"
            break
    edges = edges.rename(columns=rename_map)
    if not {"pre", "post", "weight"}.issubset(edges.columns):
        print(f"      could not find pre/post/weight columns, got {list(edges.columns)}", file=sys.stderr)
        return 2

    # If neuropil-resolved (has a per-region row per pair), aggregate.
    if "neuropil" in edges.columns or len(edges) > 5_000_000:
        print("      aggregating edges by (pre, post)")
        edges = edges.groupby(["pre", "post"], as_index=False, sort=False)["weight"].sum()
        print(f"      after aggregation: {len(edges):,} unique pairs")

    # Map ids → dense indices, drop edges referencing unverified neurons
    pre_ids = edges["pre"].to_numpy()
    post_ids = edges["post"].to_numpy()
    weights = edges["weight"].to_numpy(dtype=np.float32)

    pre_idx = np.fromiter((id_to_idx.get(int(p), -1) for p in pre_ids), dtype=np.int64, count=len(pre_ids))
    post_idx = np.fromiter((id_to_idx.get(int(p), -1) for p in post_ids), dtype=np.int64, count=len(post_ids))
    keep = (pre_idx >= 0) & (post_idx >= 0)
    pre_idx = pre_idx[keep].astype(np.uint32)
    post_idx = post_idx[keep].astype(np.uint32)
    weights = weights[keep]
    print(f"      edges with both endpoints verified: {len(pre_idx):,}")

    # Pre-sign: weight ← sign(pre_nt) × synapse_count
    weights_signed = (weights * pre_sign[pre_idx]).astype(np.float32)
    n_silenced = int((weights_signed == 0).sum())
    print(f"      silenced (NT unknown / modulatory / low-conf): {n_silenced:,}")

    # Build CSR by post (incoming): row[i] = list of pre indices feeding i
    print("[5/5] building incoming-edge CSR")
    order = np.argsort(post_idx, kind="stable")
    post_sorted = post_idx[order]
    col_idx = pre_idx[order]
    weight_arr = weights_signed[order]

    row_ptr = np.zeros(n_neurons + 1, dtype=np.uint32)
    counts = np.bincount(post_sorted, minlength=n_neurons).astype(np.uint32)
    np.cumsum(counts, out=row_ptr[1:])
    n_edges = int(row_ptr[-1])
    assert n_edges == len(col_idx)

    OUT_BIN.parent.mkdir(parents=True, exist_ok=True)
    print(f"      writing {OUT_BIN}")

    with OUT_BIN.open("wb") as f:
        # Header
        f.write(b"WGFLYBRN")
        f.write(struct.pack("<III", 1, n_neurons, n_edges))
        f.write(struct.pack("<I", 0x1))  # bit 0: pre-signed weights
        f.write(struct.pack("<fff", *VOXEL_NM))
        f.write(b"\x00" * (5 * 4))  # reserved
        # Neurons
        f.write(neurons.tobytes())
        # CSR
        f.write(row_ptr.tobytes())
        f.write(col_idx.tobytes())
        f.write(weight_arr.tobytes())

    size_bytes = OUT_BIN.stat().st_size
    print(f"      {size_bytes / 1e6:.1f} MB written")

    meta = {
        "version": 1,
        "num_neurons": n_neurons,
        "num_edges": n_edges,
        "voxel_to_nm": list(VOXEL_NM),
        "stats": {
            "with_soma": n_with_soma,
            "with_centroid_only": n_with_pos,
            "with_nt": n_with_nt,
            "silenced_edges": n_silenced,
        },
        "super_class_table": SUPER_CLASS_TABLE,
        "hero_cell_types": HERO_CELL_TYPES,
    }
    OUT_META.write_text(json.dumps(meta, indent=2))
    print(f"      {OUT_META.name} written")
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
