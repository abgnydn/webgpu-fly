#!/usr/bin/env python3
"""
verify_walking_policy.py — numpy-only forward pass over the same
weights walking-policy.bin packs, used as ground truth for the TS
implementation in src/walking-policy.ts.

For two test inputs (all zeros + a synthetic ramp), computes the
deterministic action and writes it to tests/walking-policy-fixtures.json.
The e2e test then loads this fixture, runs the TS forward pass on the
same input, and checks every output matches within float tolerance.

Run after every change to extract_walking_policy.py.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "public" / "walking-policy.bin"
OUT = ROOT / "tests" / "walking-policy-fixtures.json"


def load_policy(path: Path):
    buf = path.read_bytes()
    assert buf[:8] == b"WGFLYWLK", "bad magic"
    version, obs_dim, hidden, act_dim, n_layers, _pad = struct.unpack("<IIIIII", buf[8:32])
    assert version == 1
    off = 32

    def read(n):
        nonlocal off
        a = np.frombuffer(buf, dtype=np.float32, count=n, offset=off)
        off += n * 4
        return a

    ln_scale = read(hidden).copy()
    ln_bias = read(hidden).copy()
    weights = []
    biases = []
    # Layer 1: obs → hidden
    weights.append(read(obs_dim * hidden).reshape(obs_dim, hidden).copy())
    biases.append(read(hidden).copy())
    # Layers 2..n: hidden → hidden
    for _ in range(1, n_layers):
        weights.append(read(hidden * hidden).reshape(hidden, hidden).copy())
        biases.append(read(hidden).copy())
    # Head: hidden → act_dim
    head_w = read(hidden * act_dim).reshape(hidden, act_dim).copy()
    head_b = read(act_dim).copy()

    return {
        "obs_dim": obs_dim, "hidden": hidden, "act_dim": act_dim, "n_layers": n_layers,
        "ln_scale": ln_scale, "ln_bias": ln_bias,
        "weights": weights, "biases": biases,
        "head_w": head_w, "head_b": head_b,
    }


def layernorm(x, scale, bias, eps=1e-5):
    m = x.mean()
    v = x.var()
    return (x - m) / np.sqrt(v + eps) * scale + bias


def forward(p, obs):
    h = obs @ p["weights"][0] + p["biases"][0]
    h = layernorm(h, p["ln_scale"], p["ln_bias"])
    h = np.tanh(h)
    for l in range(1, p["n_layers"]):
        h = h @ p["weights"][l] + p["biases"][l]
        # ELU (Acme LayerNormMLP default)
        h = np.where(h > 0, h, np.expm1(h))
    a = h @ p["head_w"] + p["head_b"]
    return a


def main():
    p = load_policy(BIN)
    print(f"obs_dim={p['obs_dim']} hidden={p['hidden']} act_dim={p['act_dim']} n_layers={p['n_layers']}")

    fixtures = []
    # Case 1: all zeros (rest pose).
    obs0 = np.zeros(p["obs_dim"], dtype=np.float32)
    a0 = forward(p, obs0)
    fixtures.append({"name": "zero", "obs": obs0.tolist(), "action": a0.tolist()})
    print(f"zero  obs → max action {np.abs(a0).max():.3f}, first 5: {a0[:5]}")

    # Case 2: ramp (i % 7) * 0.01 — same as the e2e test.
    obs1 = np.zeros(p["obs_dim"], dtype=np.float32)
    for i in range(p["obs_dim"]):
        obs1[i] = (i % 7) * 0.01
    a1 = forward(p, obs1)
    fixtures.append({"name": "ramp", "obs": obs1.tolist(), "action": a1.tolist()})
    print(f"ramp  obs → max action {np.abs(a1).max():.3f}, first 5: {a1[:5]}")

    OUT.write_text(json.dumps(fixtures))
    print(f"\nwrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
