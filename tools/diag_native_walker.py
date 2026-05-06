#!/usr/bin/env python3
"""diag_native_walker.py — run flybody's Walking env in native MuJoCo
(via dm_control) for a fixed number of control ticks, applying a
known action sequence, and log body trajectory + leg sensor readings
each tick. Output is the gold-standard reference our browser
mujoco_wasm reproduction is compared against.

Two modes (pick via CLI arg):

  zero     constant 0 across all 59 actuator dims. Body should stand
           still under adhesion + gravity. Useful to verify that
           ground contact / mass / gravity match.

  rl       fully load the trained RL policy and run it for the
           requested duration. Compares the headline behavior we
           care about (walking forward).

Usage:
  .venv-flygym/bin/python tools/diag_native_walker.py zero    --steps 200
  .venv-flygym/bin/python tools/diag_native_walker.py rl      --steps 200

Output: data/native-walker-<mode>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def setup_env():
    """Build the flybody walk-imitation env in inference mode (no
    mocap dataset required); same configuration the trained policy
    was deployed against per flybody/fly_envs.py."""
    from flybody.fly_envs import walk_imitation
    return walk_imitation()


def init_state(env):
    """Reset env and return initial body trajectory entry."""
    ts = env.reset()
    physics = env.physics
    qpos = physics.data.qpos[:7].copy()  # x,y,z, qw,qx,qy,qz
    return ts, qpos


def step_log(env, action):
    """Step env once with the given action, log body root position."""
    ts = env.step(action)
    physics = env.physics
    qpos = physics.data.qpos[:7].copy()
    qvel = physics.data.qvel[:6].copy()
    return ts, qpos, qvel, ts.reward, ts.last()


def run(mode: str, steps: int):
    env = setup_env()
    n_act = env.action_spec().shape[0]
    print(f"  env: action dim = {n_act}, steps = {steps}")

    if mode == "zero":
        actions = np.zeros((steps, n_act), dtype=np.float32)
    elif mode == "rl":
        # Load our extracted policy (numpy weights from
        # tools/extract_walking_policy.py) and run forward passes.
        import struct
        bin_path = ROOT / "public/walking-policy.bin"
        if not bin_path.exists():
            raise SystemExit(f"missing {bin_path}; run extract_walking_policy.py first")
        # Tiny in-memory loader matching src/walking-policy.ts.
        buf = bin_path.read_bytes()
        magic = buf[:8].decode()
        assert magic == "WGFLYWLK", magic
        version, obs_dim, hidden, act_dim, n_layers = struct.unpack("<IIIII", buf[8:28])
        off = 32
        ln_scale = np.frombuffer(buf, dtype=np.float32, count=hidden, offset=off); off += hidden * 4
        ln_bias = np.frombuffer(buf, dtype=np.float32, count=hidden, offset=off); off += hidden * 4
        dense_w, dense_b = [], []
        # Layer 1: obs → hidden
        dense_w.append(np.frombuffer(buf, dtype=np.float32, count=obs_dim*hidden, offset=off).reshape(obs_dim, hidden)); off += obs_dim*hidden*4
        dense_b.append(np.frombuffer(buf, dtype=np.float32, count=hidden, offset=off)); off += hidden * 4
        for _ in range(n_layers - 1):
            dense_w.append(np.frombuffer(buf, dtype=np.float32, count=hidden*hidden, offset=off).reshape(hidden, hidden)); off += hidden*hidden*4
            dense_b.append(np.frombuffer(buf, dtype=np.float32, count=hidden, offset=off)); off += hidden * 4
        head_w = np.frombuffer(buf, dtype=np.float32, count=hidden*act_dim, offset=off).reshape(hidden, act_dim); off += hidden*act_dim*4
        head_b = np.frombuffer(buf, dtype=np.float32, count=act_dim, offset=off); off += act_dim * 4
        print(f"  policy: obs={obs_dim} hidden={hidden} act={act_dim} layers={n_layers}")

        def layer_norm(x, scale, bias, eps=1e-5):
            mean = x.mean()
            std = np.sqrt(x.var() + eps)
            return (x - mean) / std * scale + bias

        def policy(obs):
            h = obs @ dense_w[0] + dense_b[0]
            h = layer_norm(h, ln_scale, ln_bias)
            h = np.tanh(h)
            for l in range(1, n_layers):
                h = h @ dense_w[l] + dense_b[l]
                h = np.where(h > 0, h, np.expm1(h))  # ELU
            return h @ head_w + head_b

        actions = None
    else:
        raise SystemExit(f"unknown mode: {mode}")

    ts, qpos0 = init_state(env)
    print(f"  init qpos[:3] = {qpos0[:3]}")

    log = {
        "mode": mode,
        "steps": steps,
        "control_dt": float(env.control_timestep()),
        "init_qpos": qpos0.tolist(),
        "trajectory": [],   # per-step [step, qpos[:3], qvel[:3], reward]
        "actions": [],      # per-step action vec (only for zero/fixed; rl logs first 5 dims)
    }

    obs = ts.observation
    action_stats = {"max": 0.0, "mean": 0.0, "n": 0}
    for s in range(steps):
        if mode == "rl":
            # Build the 741-dim flat obs vec in alphabetical key order.
            keys = sorted(k for k in obs.keys() if not k.startswith("walker/"))
            # Actually flybody prefixes obs with "walker/"; iterate them.
            keys = sorted(k for k in obs.keys() if k.startswith("walker/"))
            flat = np.concatenate([np.asarray(obs[k]).reshape(-1) for k in keys])
            assert flat.size == obs_dim, f"obs size {flat.size} != {obs_dim}"
            action = policy(flat.astype(np.float32))
            am = float(np.abs(action).max())
            ame = float(np.abs(action).mean())
            action_stats["max"] = max(action_stats["max"], am)
            action_stats["mean"] = (action_stats["mean"] * action_stats["n"] + ame) / (action_stats["n"] + 1)
            action_stats["n"] += 1
            # CanonicalSpec inverse: action ∈ [-1,1] → ctrl
            action_clipped = np.clip(action, -1, 1)
        else:
            action_clipped = actions[s]

        ts, qpos, qvel, reward, done = step_log(env, action_clipped)
        obs = ts.observation
        log["trajectory"].append({
            "step": s,
            "qpos": qpos.tolist(),
            "qvel": qvel.tolist(),
            "reward": float(reward) if reward is not None else 0.0,
        })
        if mode != "rl":
            log["actions"].append(action_clipped.tolist())
        if (s + 1) % 50 == 0:
            print(f"    step {s+1}/{steps}  qpos[:3]={qpos[:3].round(3)}  done={done}")
        if done:
            print(f"  episode terminated at step {s}")
            break

    out = ROOT / f"data/native-walker-{mode}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(log, indent=2))
    final = log["trajectory"][-1]["qpos"]
    init = log["init_qpos"]
    dx = final[0] - init[0]
    dy = final[1] - init[1]
    dz = final[2] - init[2]
    print(f"\n  final dx={dx:.3f}  dy={dy:.3f}  dz={dz:.3f}  cm")
    if mode == "rl":
        print(f"  policy actions:  |max|={action_stats['max']:.3f}  |mean|={action_stats['mean']:.3f}")
    print(f"  wrote {out.relative_to(ROOT)}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["zero", "rl"])
    ap.add_argument("--steps", type=int, default=200)
    args = ap.parse_args()
    sys.exit(run(args.mode, args.steps))


if __name__ == "__main__":
    main()
