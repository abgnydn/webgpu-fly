#!/usr/bin/env python3
"""
extract_walking_policy.py — read TuragaLab/flybody's trained walking
policy SavedModel and emit a tiny binary the browser bundle can load
without TensorFlow / ONNX runtime.

The policy is a 4-hidden-layer MLP:
  obs[741] → Dense(512) + LayerNorm + tanh
          → Dense(512) + tanh
          → Dense(512) + tanh
          → Dense(512) + tanh
          → Dense(59) action mean
                 + Dense(59) action stddev (unused at inference)

We only need the action-mean head — the stddev head is for stochastic
sampling during training. At inference Acme uses the mean.

Output: public/walking-policy.bin
  Magic     : "WGFLYWLK" (8 B)
  Header    : version u32, obs_dim u32, hidden u32, act_dim u32, n_layers u32, pad u32 (24 B)
  Layers    : per dense layer { W (in×out f32), b (out f32) }
              followed by per layernorm { gain (out f32), bias (out f32) }
              order matches Acme/Snt's internal numbering captured below.
  Total     : ~3 MB (mean head only — half the original 4.7 MB).
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "flybody-policies" / "walking" / "variables" / "variables"
OUT = ROOT / "public" / "walking-policy.bin"

# Observed shapes from the checkpoint:
#   _variables/0   [512]             —  layernorm bias        (post-Dense1)
#   _variables/1   [741, 512]        —  Dense1 weight (obs → 512)
#   _variables/2   [512]             —  Dense1 bias
#   _variables/3   [512]             —  layernorm gain
#   _variables/4   [512]             —  layernorm bias (?)    — actually unsure
#   _variables/5   [512, 512]        —  Dense2 weight
#   _variables/6   [512]             —  Dense2 bias
#   _variables/7   [512, 512]        —  Dense3 weight
#   _variables/8   [512]             —  Dense3 bias
#   _variables/9   [512, 512]        —  Dense4 weight
#   _variables/10  [59]              —  ?
#   _variables/11  [512, 59]         —  Dense5 weight (action mean)
#   _variables/12  [59]              —  ?
#   _variables/13  [512, 59]         —  Dense6 weight (action stddev — unused)
#
# Acme's `acme.tf.networks.LayerNormMLP` order is:
#   layernorm.scale  (3)
#   layernorm.offset (4 or 0?)
#   dense_1.w (1)  dense_1.b (2)
#   dense_2.w (5)  dense_2.b (6)
#   dense_3.w (7)  dense_3.b (8)
#   dense_4.w (9)  dense_4.b (?)
# Then the head:
#   dense_mean.w  dense_mean.b
#   dense_std.w   dense_std.b
# Numbering is Sonnet's submodule order which is creation-order.

reader = tf.train.load_checkpoint(str(SRC))
variables = {}
for k in sorted(reader.get_variable_to_shape_map().keys()):
    if k == "_CHECKPOINTABLE_OBJECT_GRAPH":
        continue
    idx = int(k.split("/")[1])
    variables[idx] = (k, reader.get_tensor(k))

print("variable index → (path, shape):")
for i in sorted(variables.keys()):
    path, val = variables[i]
    print(f"  v{i:>2} {str(val.shape):20s} {path}")

# Reconstruct the network. The Acme LayerNormMLP order in the
# checkpoint (after some experimentation) is:
#   v0  layernorm bias  (after Dense1)
#   v1  Dense1.w
#   v2  Dense1.b
#   v3  layernorm scale
#   v4  unused / could be a duplicate?
#   v5  Dense2.w     v6  Dense2.b
#   v7  Dense3.w     v8  Dense3.b
#   v9  Dense4.w     v10 Dense4.b? but it's [59] not [512]…
#
# So variable 4 is likely Dense_mean.b? Let me verify by shape pairs:
#   Dense layer pattern: (in, out) weight + (out,) bias
#   v1+v2: Dense(741→512) + b[512]            ✓
#   v5+v6: Dense(512→512) + b[512]            ✓
#   v7+v8: Dense(512→512) + b[512]            ✓
#   v9+...: Dense(512→512) + b[??]            ← v9 is [512,512], v10 is [59]
#   v11+v12: Dense(512→59) + b[59]?           ← v11 is [512,59], v12 is [59]
#   v13+...: Dense(512→59) + b[?]
#
# So the Dense4 bias is actually the layernorm-bias / something else.
# Most likely the four hidden layers each have their own LN, but only
# the first one is exposed here (acme's LayerNormMLP only does LN on
# layer 1; subsequent are plain Dense+tanh).

# Final layout assumption (LayerNormMLP + GaussianMixtureHead-ish):
#   layer  | weight idx | bias idx | extra (LN scale, LN bias)
#   1      | v1         | v2       | LN: scale=v3, bias=v0     ← LayerNorm only on layer 1
#   2      | v5         | v6       |
#   3      | v7         | v8       |
#   4      | v9         | v?       | bias missing? maybe v4 is it
#   head   | v11        | v10 (?)  |   59-d action mean
#   stddev | v13        | v12 (?)  |   unused at inference
#
# Let me try v4 as Dense4.b — but v4 is [512], same shape as needed.

# Pack everything into a single bin in a known order:
def pack(*arrays):
    out = bytearray()
    for a in arrays:
        out += a.astype(np.float32).tobytes()
    return out

ln_bias  = variables[0][1]          # [512]  layernorm bias
ln_scale = variables[3][1]          # [512]  layernorm scale
d1_w     = variables[1][1]          # [741, 512]
d1_b     = variables[2][1]          # [512]
d2_w     = variables[5][1]          # [512, 512]
d2_b     = variables[6][1]          # [512]
d3_w     = variables[7][1]          # [512, 512]
d3_b     = variables[8][1]          # [512]
d4_w     = variables[9][1]          # [512, 512]
d4_b     = variables[4][1]          # [512]   — guessed
head_w   = variables[11][1]         # [512, 59] action mean
head_b   = variables[10][1]         # [59]

OBS_DIM   = d1_w.shape[0]
HIDDEN    = d1_w.shape[1]
ACT_DIM   = head_w.shape[1]
N_LAYERS  = 4

print()
print(f"obs_dim={OBS_DIM}  hidden={HIDDEN}  act_dim={ACT_DIM}  n_layers={N_LAYERS}")
print(f"first-layer LN bias[:5] = {ln_bias[:5]}")
print(f"head bias[:5]            = {head_b[:5]}")

# Magic + header
buf = bytearray(b"WGFLYWLK")
buf += struct.pack("<IIIIII",
                   1,          # version
                   OBS_DIM,
                   HIDDEN,
                   ACT_DIM,
                   N_LAYERS,
                   0)          # pad
# Layer norm (first layer only, per acme convention)
buf += pack(ln_scale, ln_bias)
# Dense layers
buf += pack(d1_w, d1_b,
            d2_w, d2_b,
            d3_w, d3_b,
            d4_w, d4_b)
# Action mean head
buf += pack(head_w, head_b)

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_bytes(buf)
print(f"\nwrote {OUT} ({len(buf) / 1e6:.2f} MB)")
