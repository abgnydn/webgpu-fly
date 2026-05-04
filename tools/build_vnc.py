#!/usr/bin/env python3
"""
build_vnc.py — Janelia MANC v1.0 → binary CSR spine blob (vnc.bin).

Reads
  data/manc/traced-neurons.csv         (bodyId, type, instance)
  data/manc/traced-connections.csv     (bodyId_pre, bodyId_post, weight)
  data/manc/neuron-properties.feather  (full metadata: class, predictedNt, etc.)

Writes
  public/vnc.bin       (binary CSR, ~70 MB)
  public/vnc.meta.json (DN input map + motor neuron groups by leg)

Format mirrors public/brain.bin so the same WGSL LIF kernel can read it
without modification — only the magic string differs.

  [ Header — 64 B ]
    magic         char[8]  = "WGFLYVNC"
    version       u32      = 1
    num_neurons   u32      = N
    num_edges     u32      = E
    flags         u32      bit 0: weights are pre-signed by presynaptic NT
    reserved      u32[12]  pad to 64 B

  [ Neurons — N × 32 B ]
    pos_x         f32      soma x in nm (from somaLocation)
    pos_y         f32
    pos_z         f32
    sign          f32      -1 / 0 / +1 (informational; weights already signed)
    cell_class    u32      0 unknown, 1 DN_INPUT, 2 MOTOR, 3 SENSORY, 4 INTRINSIC
    leg_segment   u32      packed: bits 0..3 leg id (1..6), bits 4..7 side (0=L, 1=R)
                           0 if not a leg motor neuron
    flags         u32      reserved
    nt_conf       f32      neurotransmitter prediction confidence

  [ CSR row_ptr — (N+1) × u32 ]   incoming edges
  [ CSR col_idx — E × u32 ]       presynaptic neuron index
  [ CSR weight  — E × f32 ]       sign(pre_nt) × weight

vnc.meta.json keys:
  num_neurons, num_edges
  dn_inputs : { "DNa01": [vnc_idx, vnc_idx, ...], "DNp01": [...], ... }
              — by-cell-type lookup of DN axon terminals so the brain
              can drive them via captureRollingRate output.
  motor : {
    "T1_left":  { "coxa": [...], "femur": [...], "tibia": [...] },
    "T1_right": ...,
    ...
  }              — motor neurons grouped by leg + side, ready for the
                  body-driver to read average rates from.
  cell_classes : counts per class
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as pa_feather

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "manc"
OUT_BIN = ROOT / "public" / "vnc.bin"
OUT_META = ROOT / "public" / "vnc.meta.json"

# Cell-class enum (kept stable across versions; runtime reads as u32).
CLASS_UNKNOWN, CLASS_DN_INPUT, CLASS_MOTOR, CLASS_SENSORY, CLASS_INTRINSIC = 0, 1, 2, 3, 4

# Neurotransmitter sign (same convention as build_csr.py).
NT_SIGN = {
    "acetylcholine": +1,
    "gaba": -1,
    "glutamate": -1,           # mostly inhibitory in fly via GluCl
    "dopamine": 0,
    "serotonin": 0,
    "octopamine": 0,
    "unclear": 0,
    "unknown": 0,
}
NT_CONF_MIN = 0.5


def load_neurons() -> pd.DataFrame:
    print("loading neuron-properties.feather …")
    df = pa_feather.read_feather(RAW / "neuron-properties.feather")
    print(f"  {len(df):,} neurons total")

    # Reduce to bodyIds present in traced-neurons.csv (the LIF circuit).
    traced = pd.read_csv(RAW / "traced-neurons.csv")
    df = df[df["bodyId"].isin(traced["bodyId"])].copy()
    print(f"  {len(df):,} traced neurons retained")
    return df.reset_index(drop=True)


def load_connections(body2idx: dict[int, int]) -> pd.DataFrame:
    print("loading traced-connections.csv …")
    df = pd.read_csv(RAW / "traced-connections.csv")
    print(f"  {len(df):,} edges total")

    # Map bodyIds → dense indices; drop any edge whose endpoints aren't
    # in the traced neuron list.
    df["pre_idx"] = df["bodyId_pre"].map(body2idx)
    df["post_idx"] = df["bodyId_post"].map(body2idx)
    n_before = len(df)
    df = df.dropna(subset=["pre_idx", "post_idx"]).copy()
    df["pre_idx"] = df["pre_idx"].astype(np.uint32)
    df["post_idx"] = df["post_idx"].astype(np.uint32)
    print(f"  {len(df):,} edges retained (dropped {n_before - len(df):,} cross-references)")
    return df


def classify(row) -> int:
    """Map MANC class string to our cell_class enum."""
    raw = row.get("class")
    cls = "" if raw is None or (isinstance(raw, float) and np.isnan(raw)) else str(raw).lower()
    if "descending" in cls:
        return CLASS_DN_INPUT
    if "motor" in cls:
        return CLASS_MOTOR
    if "sensory" in cls:
        return CLASS_SENSORY
    if "intrinsic" in cls:
        return CLASS_INTRINSIC
    return CLASS_UNKNOWN


def parse_position(val) -> tuple[float, float, float]:
    """somaLocation / position is a list-string '[x, y, z]' or list. Returns
    (x, y, z) in nm; (0,0,0) if missing."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return 0.0, 0.0, 0.0
    if isinstance(val, str):
        s = val.strip("[]() ").replace(",", " ").split()
        if len(s) < 3:
            return 0.0, 0.0, 0.0
        return float(s[0]), float(s[1]), float(s[2])
    if isinstance(val, (list, tuple, np.ndarray)) and len(val) >= 3:
        return float(val[0]), float(val[1]), float(val[2])
    return 0.0, 0.0, 0.0


def neuron_sign(predicted_nt: str | None) -> tuple[int, float]:
    if predicted_nt is None or (isinstance(predicted_nt, float) and np.isnan(predicted_nt)):
        return 0, 0.0
    nt = str(predicted_nt).lower()
    return NT_SIGN.get(nt, 0), 1.0    # confidence not exposed cleanly per row; use 1.0


def leg_segment_packed(row) -> int:
    """Pack T1/T2/T3 + L/R for motor neurons. 0 if not a leg motor."""
    if classify(row) != CLASS_MOTOR:
        return 0
    raw_nm = row.get("somaNeuromere")
    raw_sd = row.get("somaSide")
    nm = "" if raw_nm is None or (isinstance(raw_nm, float) and np.isnan(raw_nm)) else str(raw_nm).upper()
    side = "" if raw_sd is None or (isinstance(raw_sd, float) and np.isnan(raw_sd)) else str(raw_sd).upper()
    leg_id = 0
    if "T1" in nm: leg_id = 1
    elif "T2" in nm: leg_id = 2
    elif "T3" in nm: leg_id = 3
    if leg_id == 0:
        return 0
    side_bit = 1 if side == "RHS" else 0
    return leg_id | (side_bit << 4)


def build_meta(df: pd.DataFrame) -> dict:
    """Build the runtime-side index map. Motor neurons are split by:
      - body region (subclass: fl/ml/hl=front/middle/hind leg, wm=wing,
        nm=neck, hm=haltere, ad=abdomen-dorsal),
      - per-leg + side (T1/T2/T3 × left/right, biomechanical key),
      - per-muscle target group (ti_flexor, ti_extensor, tr_flexor,
        tr_extensor, fe_reductor, etc.) for muscle-level readout.
    """
    dn_inputs: dict[str, list[int]] = {}
    motor: dict[str, dict[str, list[int]]] = {}
    motor_by_subclass: dict[str, list[int]] = {}
    motor_by_target: dict[str, list[int]] = {}

    def normalise_target(raw) -> str | None:
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            return None
        s = str(raw).strip().lower()
        # Collapse e.g. "Ti flexor" / "Acc. ti flexor" → "ti_flexor"
        s = s.replace(".", "").replace("/", "_").replace("-", "_")
        s = "_".join(s.split())
        return s or None

    for idx, row in df.iterrows():
        cls = classify(row)
        ty = str(row.get("type") or "")
        if cls == CLASS_DN_INPUT and ty:
            dn_inputs.setdefault(ty, []).append(int(idx))
        elif cls == CLASS_MOTOR:
            sub = row.get("subclass")
            if isinstance(sub, str):
                motor_by_subclass.setdefault(sub, []).append(int(idx))
            tgt = normalise_target(row.get("target"))
            if tgt:
                motor_by_target.setdefault(tgt, []).append(int(idx))

            # Per-leg + side bucketing (only for fl/ml/hl subclass —
            # avoids dragging wing/neck/abdomen motors into "leg group").
            sub_str = str(sub) if isinstance(sub, str) else ""
            if sub_str in ("fl", "ml", "hl"):
                leg_label = {"fl": "T1", "ml": "T2", "hl": "T3"}[sub_str]
                raw_sd = row.get("somaSide")
                side = "" if raw_sd is None or (isinstance(raw_sd, float) and np.isnan(raw_sd)) else str(raw_sd).upper()
                side_label = "left" if side == "LHS" else "right" if side == "RHS" else "mid"
                key = f"{leg_label}_{side_label}"
                bucket = motor.setdefault(key, {"all": []})
                bucket["all"].append(int(idx))
                if tgt:
                    bucket.setdefault(tgt, []).append(int(idx))

    cell_class_counts = df.apply(classify, axis=1).value_counts().to_dict()
    return {
        "dn_inputs": dn_inputs,
        "motor": motor,
        "motor_by_subclass": motor_by_subclass,
        "motor_by_target": motor_by_target,
        "cell_classes": {
            "unknown": int(cell_class_counts.get(CLASS_UNKNOWN, 0)),
            "dn_input": int(cell_class_counts.get(CLASS_DN_INPUT, 0)),
            "motor": int(cell_class_counts.get(CLASS_MOTOR, 0)),
            "sensory": int(cell_class_counts.get(CLASS_SENSORY, 0)),
            "intrinsic": int(cell_class_counts.get(CLASS_INTRINSIC, 0)),
        },
    }


def main() -> None:
    df = load_neurons()
    body_ids = df["bodyId"].astype(np.int64).to_numpy()
    body2idx = {int(b): i for i, b in enumerate(body_ids)}
    N = len(df)

    edges = load_connections(body2idx)

    # Sort by post then pre so CSR row_ptr building is a single pass.
    edges.sort_values(by=["post_idx", "pre_idx"], inplace=True, kind="stable")
    edges.reset_index(drop=True, inplace=True)

    # Pre-sign weights by presynaptic NT (same convention as brain.bin).
    print("signing weights by presynaptic neurotransmitter …")
    pre_nt_by_idx = df["predictedNt"].fillna("unknown").astype(str).str.lower().to_numpy()
    sign_arr = np.zeros(N, dtype=np.float32)
    for i, nt in enumerate(pre_nt_by_idx):
        sign_arr[i] = NT_SIGN.get(nt, 0)

    pre_idx = edges["pre_idx"].to_numpy(dtype=np.uint32)
    post_idx = edges["post_idx"].to_numpy(dtype=np.uint32)
    weights = edges["weight"].to_numpy(dtype=np.float32) * sign_arr[pre_idx]

    E = len(edges)
    print(f"  {N:,} neurons, {E:,} edges, "
          f"{(weights > 0).sum():,} excitatory, "
          f"{(weights < 0).sum():,} inhibitory, "
          f"{(weights == 0).sum():,} silent")

    # Build CSR row_ptr.
    print("building CSR row_ptr …")
    row_ptr = np.zeros(N + 1, dtype=np.uint32)
    for p in post_idx:
        row_ptr[p + 1] += 1
    np.cumsum(row_ptr, out=row_ptr)

    # Per-neuron records.
    print("packing neuron records …")
    pos = np.zeros((N, 3), dtype=np.float32)
    cell_class = np.zeros(N, dtype=np.uint32)
    leg_seg = np.zeros(N, dtype=np.uint32)
    nt_conf = np.zeros(N, dtype=np.float32)

    for i, row in df.iterrows():
        soma = row.get("somaLocation")
        if soma is None or (isinstance(soma, float) and np.isnan(soma)):
            soma = row.get("position")
        x, y, z = parse_position(soma)
        pos[i] = (x, y, z)
        cell_class[i] = classify(row)
        leg_seg[i] = leg_segment_packed(row)
        # predictedNtProb gives float confidence; default 0
        try:
            nt_conf[i] = float(row.get("predictedNtProb") or 0)
        except Exception:
            nt_conf[i] = 0.0

    # Write binary
    OUT_BIN.parent.mkdir(parents=True, exist_ok=True)
    print(f"writing {OUT_BIN} …")
    with open(OUT_BIN, "wb") as f:
        # Header — 64 B
        f.write(b"WGFLYVNC")                 # magic 8B
        f.write(struct.pack("<III", 1, N, E))  # version, N, E
        f.write(struct.pack("<I", 1))          # flags: bit 0 = pre-signed
        f.write(b"\x00" * (64 - 8 - 12 - 4))   # pad to 64 B (12*4 bytes left)
        # Neurons N × 32 B
        for i in range(N):
            x, y, z = pos[i]
            sign = float(np.sign(sign_arr[i]))
            f.write(struct.pack("<ffffIIIf",
                                x, y, z, sign,
                                int(cell_class[i]),
                                int(leg_seg[i]),
                                0,                                 # flags reserved
                                float(nt_conf[i])))
        # CSR row_ptr (N+1) × u32
        f.write(row_ptr.tobytes())
        # CSR col_idx E × u32
        f.write(pre_idx.tobytes())
        # CSR weight E × f32
        f.write(weights.tobytes())

    bytes_written = OUT_BIN.stat().st_size
    print(f"  wrote {bytes_written / 1e6:.1f} MB")

    print(f"writing {OUT_META} …")
    meta = build_meta(df)
    meta["num_neurons"] = N
    meta["num_edges"] = int(E)
    OUT_META.write_text(json.dumps(meta, indent=2))

    # Summary
    print()
    print("VNC summary:")
    print(f"  neurons   : {N:,}")
    print(f"  edges     : {E:,}")
    print(f"  DN inputs : {meta['cell_classes']['dn_input']:,}")
    print(f"  motor     : {meta['cell_classes']['motor']:,}")
    print(f"  sensory   : {meta['cell_classes']['sensory']:,}")
    print(f"  intrinsic : {meta['cell_classes']['intrinsic']:,}")
    print(f"  unknown   : {meta['cell_classes']['unknown']:,}")
    print(f"  DN names  : {len(meta['dn_inputs'])} unique types")
    print(f"  motor legs: {sorted(meta['motor'].keys())}")
    print(f"  subclass  : {dict((k, len(v)) for k, v in sorted(meta['motor_by_subclass'].items()))}")
    print(f"  muscles   : {len(meta['motor_by_target'])} distinct target groups")


if __name__ == "__main__":
    main()
