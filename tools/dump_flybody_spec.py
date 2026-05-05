"""Instantiate flybody's Walking task and dump everything we need to align
our browser sim with the trained policy:

  - actuator names in declaration order (must match walkingActuatorIds[])
  - joint names + qpos/qvel sizes (must match our 85-slot slicing)
  - body names for appendages_pos source (7 sites in egocentric frame)
  - observation_spec dims per stream (must match WALKING_OBS_LAYOUT)
  - action_spec range (already confirmed [0,1])

Outputs:
  data/flybody-spec.json — human-readable spec
  data/walking-obs-norm.bin — per-channel mean/std over a random rollout
                              (286 dims; ref_* is synthetic + identity in
                              our runtime so we don't normalize those)

Run from repo root:
  .venv-flygym/bin/python tools/dump_flybody_spec.py
"""
import json
import os

import numpy as np
from dm_control import composer
from dm_control.locomotion.arenas import floors

from flybody.fruitfly.fruitfly import FruitFly
from flybody.tasks.base import Walking


class WalkingStub(Walking):
    def get_reward_factors(self, physics):
        return {}


def main() -> None:
    arena = floors.Floor()
    task = WalkingStub(
        walker=FruitFly,
        arena=arena,
        time_limit=2.0,
        joint_filter=0.01,
        disable_wings=True,
    )
    env = composer.Environment(task)
    physics = env.physics

    actuator_names = [physics.model.id2name(i, "actuator") for i in range(physics.model.nu)]
    joint_names = [physics.model.id2name(i, "joint") for i in range(physics.model.njnt)]
    body_names = [physics.model.id2name(i, "body") for i in range(physics.model.nbody)]
    site_names = [physics.model.id2name(i, "site") for i in range(physics.model.nsite)]
    sensor_names = [physics.model.id2name(i, "sensor") for i in range(physics.model.nsensor)]

    spec = env.observation_spec()
    obs_streams = []
    for k in sorted(spec.keys()):
        s = spec[k]
        flat = int(np.prod(s.shape))
        obs_streams.append({"name": k, "shape": list(s.shape), "flat_dim": flat, "dtype": str(s.dtype)})

    a_spec = env.action_spec()

    out = {
        "nq": int(physics.model.nq),
        "nv": int(physics.model.nv),
        "nu": int(physics.model.nu),
        "njnt": int(physics.model.njnt),
        "actuators": actuator_names,
        "joints": joint_names,
        "bodies": body_names,
        "sites": site_names,
        "sensors": sensor_names,
        "observation_spec": obs_streams,
        "observation_total_flat": int(sum(s["flat_dim"] for s in obs_streams)),
        "action_spec": {
            "shape": list(a_spec.shape),
            "minimum": a_spec.minimum.tolist(),
            "maximum": a_spec.maximum.tolist(),
            "dtype": str(a_spec.dtype),
        },
    }

    os.makedirs("data", exist_ok=True)
    with open("data/flybody-spec.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote data/flybody-spec.json — nu={out['nu']}  nq={out['nq']}  obs_flat={out['observation_total_flat']}")
    print(f"  first 5 actuators: {actuator_names[:5]}")
    print(f"  first 5 joints:    {joint_names[:5]}")

    # ---- normalization rollout ---------------------------------------
    # Uniformly random actions blow the body around — observations end
    # up far from anything the trained policy actually sees. Use a
    # mid-control + small noise schedule instead: each control held
    # near its midpoint with a 10% gaussian perturbation. The fly mostly
    # stands and gently shifts, which is closer to the policy's
    # deployment distribution than a flailing fly.
    n_steps = 600
    obs_buf = []
    ts = env.reset()
    a_lo = a_spec.minimum.astype(np.float32)
    a_hi = a_spec.maximum.astype(np.float32)
    a_mid = 0.5 * (a_lo + a_hi)
    a_half = 0.5 * (a_hi - a_lo)
    rng = np.random.default_rng(0)
    for _ in range(n_steps):
        noise = rng.normal(0.0, 0.1, size=a_spec.shape).astype(np.float32)
        a = a_mid + noise * a_half
        a = np.clip(a, a_lo, a_hi)
        ts = env.step(a)
        if ts.last():
            ts = env.reset()
            continue
        flat = []
        for k in sorted(ts.observation.keys()):
            flat.append(np.asarray(ts.observation[k]).reshape(-1))
        obs_buf.append(np.concatenate(flat))
    obs = np.stack(obs_buf, axis=0).astype(np.float32)
    print(f"obs rollout shape: {obs.shape}")

    mean = obs.mean(axis=0)
    std = obs.std(axis=0)
    # Floor std at 1e-2 — a too-small std blows up normalization at
    # near-constant channels (e.g. world_zaxis when the fly is upright).
    std = np.maximum(std, 1e-2)

    os.makedirs("public", exist_ok=True)
    # Binary layout: u32 nDims, u32 reserved, then float32[nDims] mean,
    # then float32[nDims] std. Total 8 + 8*nDims bytes.
    n = mean.shape[0]
    with open("public/walking-obs-norm.bin", "wb") as f:
        f.write(np.array([n, 0], dtype=np.uint32).tobytes())
        f.write(mean.astype(np.float32).tobytes())
        f.write(std.astype(np.float32).tobytes())
    print(f"wrote public/walking-obs-norm.bin — n={n}  mean[:3]={mean[:3]}  std[:3]={std[:3]}")


if __name__ == "__main__":
    main()
