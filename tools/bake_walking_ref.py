#!/usr/bin/env python3
"""bake_walking_ref.py — extract one (longest) walking trajectory from
the Vaxenburg 2025 mocap dataset, subsample 10× to 50 Hz, and emit a
small binary that the runtime can replay as the trained walker's
reference trajectory.

Replaces the synthetic straight-line ref currently fed to the policy
in src/physics.ts:buildWalkingObservation. Real fly walking has body
oscillations (lateral sway, vertical bob, yaw wobble) at the gait
frequency that the policy was trained on; with a real ref the policy
runs closer to its training distribution.

Source: data/raw/flybody-mocap/walking-dataset-small_female-only_*.hdf5
        (Janelia Figshare 25309105 → datasets_walking-imitation.zip,
         small subset: 100 snippets, ≥0.5 s each).

Output: public/walking-ref.bin

Format (little-endian):
  Header 32 B:
    magic    char[8] = "WGFLYREF"
    version  u32     = 1
    n_frames u32                  (after subsampling)
    timestep f32     = 0.02       (50 Hz, env's _WALK_CONTROL_TIMESTEP × 10)
    reserved u32[2]
  Data:
    root_qpos f32 [n_frames, 7]   (x, y, z, qw, qx, qy, qz)
                                  (root_qpos[:, :2] -= root_qpos[0, :2]
                                   so trajectory starts at origin)

The single trajectory will be looped at runtime. ~600 frames at 50 Hz
≈ 12 s of walking, sub-2 KB on disk.
"""
from __future__ import annotations

import struct
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
HDF5 = ROOT / "data/raw/flybody-mocap/walking-dataset-small_female-only_snippets-100_min-len-0.5s_trk-files-0-9.hdf5"
OUT = ROOT / "public/walking-ref.bin"

SUBSAMPLE = 10  # 500 Hz → 50 Hz


def main() -> int:
    if not HDF5.exists():
        raise SystemExit(f"missing {HDF5}; extract from datasets_walking-imitation.zip first")

    with h5py.File(HDF5, "r") as f:
        n_traj = len(f["trajectories"])
        lens = f["trajectory_lengths"][:]
        timestep = float(f["timestep_seconds"][()])
        print(f"  source: {n_traj} trajectories, dt={timestep*1e3:.1f} ms")

        # Score trajectories by forward distance × length so we get a
        # long forward-walking clip (the dataset has plenty of turners
        # and backwalkers we want to skip — the longest trajectory
        # turns out to walk backward at -0.8 cm/s).
        n_zeros = len(str(n_traj))
        best_idx, best_score = -1, -1e9
        for idx in range(n_traj):
            key = str(idx).zfill(n_zeros)
            rq = f["trajectories"][key]["root_qpos"][:]
            T = rq.shape[0]
            fwd = float(rq[-1, 0] - rq[0, 0])
            # Project fwd onto initial heading: rotate the (dx,dy) world
            # vector by the inverse of the initial yaw so positive = the
            # fly's own forward.
            qw, qx, qy, qz = rq[0, 3], rq[0, 4], rq[0, 5], rq[0, 6]
            hx = 1 - 2 * (qy * qy + qz * qz)
            hy = 2 * (qx * qy + qw * qz)
            dx, dy = float(rq[-1, 0] - rq[0, 0]), float(rq[-1, 1] - rq[0, 1])
            body_fwd = dx * hx + dy * hy
            # Score: body-frame forward × duration. Skip clips that drift
            # backward, encourage long, straight walks.
            score = body_fwd * T if body_fwd > 0 else -1
            if score > best_score:
                best_score, best_idx = score, idx

        traj_idx = best_idx
        key = str(traj_idx).zfill(n_zeros)
        snippet = f["trajectories"][key]
        root_qpos = snippet["root_qpos"][:]   # (T, 7)
        T = root_qpos.shape[0]
        print(f"  picked trajectory {traj_idx}: {T} frames "
              f"({T*timestep:.2f} s at source rate, score={best_score:.2f})")

    # Subsample to env rate
    sub = root_qpos[::SUBSAMPLE]
    T_sub = sub.shape[0]
    out_dt = timestep * SUBSAMPLE
    print(f"  subsampled {SUBSAMPLE}× → {T_sub} frames at {1/out_dt:.0f} Hz "
          f"({T_sub*out_dt:.2f} s)")

    # Anchor at origin (XY only — keep Z so we don't lose the body height)
    sub = sub.astype(np.float32, copy=True)
    sub[:, 0] -= sub[0, 0]
    sub[:, 1] -= sub[0, 1]
    print(f"  anchored: start xyz=({sub[0,0]:.3f},{sub[0,1]:.3f},{sub[0,2]:.3f}) "
          f"end xyz=({sub[-1,0]:.3f},{sub[-1,1]:.3f},{sub[-1,2]:.3f})")
    fwd = float(sub[-1, 0] - sub[0, 0])
    print(f"  forward distance: {fwd:.3f} cm  →  avg speed {fwd/(T_sub*out_dt):.3f} cm/s")

    # Write binary
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("wb") as g:
        # Header — 32 B total
        g.write(b"WGFLYREF")                          # 8
        g.write(struct.pack("<I", 1))                 # version
        g.write(struct.pack("<I", T_sub))             # n_frames
        g.write(struct.pack("<f", float(out_dt)))     # timestep
        g.write(b"\x00" * 12)                         # reserved (3 × u32)
        assert g.tell() == 32
        # Data: T × 7 floats
        g.write(sub.tobytes())

    size_kb = OUT.stat().st_size / 1024
    print(f"\nwrote {OUT.relative_to(ROOT)}  ({size_kb:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
