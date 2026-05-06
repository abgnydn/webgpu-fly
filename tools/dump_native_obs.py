#!/usr/bin/env python3
"""dump_native_obs.py — capture flybody Walking env's full 741-dim obs
at frames 0, 1, 5, 20 with zero actions. Used to align our wasm
buildWalkingObservation() against the gold standard, channel by
channel.

Output: data/native-obs.json (per-frame per-channel statistics +
the actual flat 741 vec).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/native-obs.json"


def main():
    from flybody.fly_envs import walk_imitation
    env = walk_imitation()
    n_act = env.action_spec().shape[0]

    captures = {}
    capture_at = {0, 1, 5, 20}
    max_steps = max(capture_at) + 1

    ts = env.reset()
    for s in range(max_steps + 1):
        if s in capture_at:
            keys = sorted(ts.observation.keys())
            arrays = {}
            for k in keys:
                arrays[k] = np.asarray(ts.observation[k]).reshape(-1).tolist()
            phys = env.physics
            captures[str(s)] = {
                "obs": arrays,
                "qpos": phys.data.qpos.tolist(),
                "qvel": phys.data.qvel.tolist(),
                "ctrl": phys.data.ctrl.tolist(),
                "act": list(phys.data.act) if phys.data.act is not None else [],
            }
        if s == max_steps:
            break
        ts = env.step(np.zeros(n_act, dtype=np.float32))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(captures, indent=2))
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"frames captured: {sorted(int(s) for s in captures)}")
    # Quick per-channel stats summary at frame 0
    print("\nFrame 0 channel summary:")
    for k, v in captures["0"]["obs"].items():
        a = np.array(v)
        print(f"  {k:40s}  dim={a.size:4d}  mean={a.mean():+.4f}  std={a.std():.4f}  range=[{a.min():+.4f}, {a.max():+.4f}]")


if __name__ == "__main__":
    main()
