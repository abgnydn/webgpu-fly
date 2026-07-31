#!/usr/bin/env python3
"""
vnc_rhythm.py — offline test of whether a walking rhythm exists in the
shipped MANC connectome (public/vnc.bin + public/vnc.meta.json).

Read-only. Touches no repo file. Run with:
  uv run --with numpy --with scipy python3 vnc_rhythm.py struct
  uv run --with numpy --with scipy python3 vnc_rhythm.py sweep

Dynamics replicate src/shaders/lif.wgsl exactly, with the constants the
VNC sim is actually created with (src/main.ts:594 -> DEFAULT_PARAMS in
src/sim.ts:31-40):
  dt = 1 ms, tau_m = 20 ms  -> alpha = exp(-1/20)
  v_thresh = -45, v_reset = v_rest = -52
  refractory 2.2 ms -> 2 steps
  ext_gain = 2.0, w_syn = 0.005
  alpha synapse A_SYN = 0.81873 (lif.wgsl:52)
"""
from __future__ import annotations

import json
import struct
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/Users/ahmetbarisgunaydin/dev/webgpu-fly")
BIN = ROOT / "public" / "vnc.bin"
META = ROOT / "public" / "vnc.meta.json"

CLASS_UNKNOWN, CLASS_DN_INPUT, CLASS_MOTOR, CLASS_SENSORY, CLASS_INTRINSIC = 0, 1, 2, 3, 4

# --- LIF constants (see docstring) ---
ALPHA = float(np.exp(-1.0 / 20.0))
V_TH, V_RESET, V_REST = -45.0, -52.0, -52.0
REFRAC_STEPS = 2
EXT_GAIN = 2.0
W_SYN = 0.005
A_SYN = 0.81873

# src/vnc.ts:51-54. LEG_KEYS order [T1L,T1R,T2L,T2R,T3L,T3R] = idx 0..5;
# tripod A = idx 0,3,4 ; tripod B = idx 1,2,5.
LEG_KEYS = ["T1_left", "T1_right", "T2_left", "T2_right", "T3_left", "T3_right"]
TRIPOD_A = ["T1_left", "T2_right", "T3_left"]
TRIPOD_B = ["T1_right", "T2_left", "T3_right"]

# src/main.ts:386-390
ANTAGONISTS = {
    "tibia": (["ti_extensor"], ["ti_flexor", "acc_ti_flexor"]),
    "femur": (["tr_extensor", "sternotrochanter"], ["tr_flexor", "acc_tr_flexor"]),
    "coxa_twist": (["sternal_anterior_rotator"], ["sternal_posterior_rotator"]),
}


# ---------------------------------------------------------------- loading
def load_vnc():
    raw = np.memmap(BIN, dtype=np.uint8, mode="r")
    magic = bytes(raw[:8])
    assert magic == b"WGFLYVNC", magic
    version, N, E, flags = struct.unpack("<IIII", bytes(raw[8:24]))
    off = 64
    nrec = np.frombuffer(bytes(raw[off:off + N * 32]), dtype=np.uint8).reshape(N, 32)
    pos = nrec[:, 0:12].copy().view(np.float32).reshape(N, 3)
    sign = nrec[:, 12:16].copy().view(np.float32).ravel()
    cell_class = nrec[:, 16:20].copy().view(np.uint32).ravel()
    leg_seg = nrec[:, 20:24].copy().view(np.uint32).ravel()
    nt_conf = nrec[:, 28:32].copy().view(np.float32).ravel()
    off += N * 32
    row_ptr = np.frombuffer(bytes(raw[off:off + (N + 1) * 4]), dtype=np.uint32).astype(np.int64)
    off += (N + 1) * 4
    col_idx = np.frombuffer(bytes(raw[off:off + E * 4]), dtype=np.uint32).astype(np.int64)
    off += E * 4
    weight = np.frombuffer(bytes(raw[off:off + E * 4]), dtype=np.float32).copy()
    meta = json.loads(META.read_text())
    return dict(version=version, N=N, E=E, flags=flags, pos=pos, sign=sign,
                cell_class=cell_class, leg_seg=leg_seg, nt_conf=nt_conf,
                row_ptr=row_ptr, col_idx=col_idx, weight=weight, meta=meta)


def post_of_edges(row_ptr, E):
    """Postsynaptic index for every edge (CSR is incoming-indexed)."""
    post = np.zeros(E, dtype=np.int64)
    counts = np.diff(row_ptr)
    return np.repeat(np.arange(len(counts), dtype=np.int64), counts)


def glut_fix_weights(weight, col_idx, cell_class):
    """Counterfactual: every motor-presynaptic edge that is currently
    negative becomes positive. Upper bound on DEFECT 1's reach — the
    shipped bin conflates GABA and glutamate, so this over-counts by
    however many motor neurons are truly GABAergic."""
    w = weight.copy()
    pre_is_motor = cell_class[col_idx] == CLASS_MOTOR
    flip = pre_is_motor & (w < 0)
    w[flip] = -w[flip]
    return w, int(flip.sum())


# ------------------------------------------------------- structural (1,2)
def premotor_pools(d, include_motor: bool):
    """P_L = presynaptic partners of leg L's motor pool."""
    meta = d["meta"]
    row_ptr, col_idx, cc = d["row_ptr"], d["col_idx"], d["cell_class"]
    pools = {}
    for leg in LEG_KEYS:
        mot = np.array(meta["motor"][leg]["all"], dtype=np.int64)
        pres = []
        for m in mot:
            pres.append(col_idx[row_ptr[m]:row_ptr[m + 1]])
        pres = np.unique(np.concatenate(pres))
        if not include_motor:
            pres = pres[cc[pres] != CLASS_MOTOR]
        pools[leg] = pres
    return pools


def signed_matrix(d, pools, weight, targets=None):
    """M[a][b] = summed signed weight of edges pre in pools[b] -> post in
    targets[a] (default targets = pools). Returns (raw, per-pair-density)."""
    row_ptr, col_idx = d["row_ptr"], d["col_idx"]
    if targets is None:
        targets = pools
    n = len(LEG_KEYS)
    raw = np.zeros((n, n))
    dens = np.zeros((n, n))
    memb = {}
    for j, leg in enumerate(LEG_KEYS):
        m = np.zeros(d["N"], dtype=bool)
        m[pools[leg]] = True
        memb[j] = m
    for a, la in enumerate(LEG_KEYS):
        tgt = targets[la]
        # concatenate all incoming edges of the target set
        segs_c, segs_w = [], []
        for t in tgt:
            s, e = row_ptr[t], row_ptr[t + 1]
            segs_c.append(col_idx[s:e])
            segs_w.append(weight[s:e])
        c = np.concatenate(segs_c) if segs_c else np.zeros(0, np.int64)
        w = np.concatenate(segs_w) if segs_w else np.zeros(0, np.float32)
        for b in range(len(LEG_KEYS)):
            sel = memb[b][c]
            raw[a, b] = w[sel].sum()
            dens[a, b] = raw[a, b] / (len(tgt) * len(pools[LEG_KEYS[b]]))
    return raw, dens


def tripod_split(M, groupA=None):
    """Return (within-tripod off-diagonal mean, across-tripod mean)."""
    if groupA is None:
        groupA = TRIPOD_A
    tri = {k: ("A" if k in groupA else "B") for k in LEG_KEYS}
    within, across = [], []
    for a, la in enumerate(LEG_KEYS):
        for b, lb in enumerate(LEG_KEYS):
            if a == b:
                continue
            (within if tri[la] == tri[lb] else across).append(M[a, b])
    return float(np.mean(within)), float(np.mean(across)), within, across


def tripod_permutation_control(M):
    """All 10 distinct 3/3 splits of the six legs. Where does the true
    tripod rank on (across - within)?"""
    from itertools import combinations
    seen, out = set(), []
    for c in combinations(LEG_KEYS, 3):
        key = frozenset(c)
        comp = frozenset(set(LEG_KEYS) - key)
        if comp in seen:
            continue
        seen.add(key)
        wi, ac, _, _ = tripod_split(M, groupA=list(c))
        out.append((ac - wi, tuple(sorted(c))))
    out.sort(reverse=True)
    return out


def fmt_matrix(M, scale=1.0):
    hdr = "            " + "".join(f"{k:>12}" for k in LEG_KEYS)
    lines = [hdr]
    for a, la in enumerate(LEG_KEYS):
        lines.append(f"{la:>12}" + "".join(f"{M[a,b]*scale:12.4g}" for b in range(6)))
    return "\n".join(lines)


def stage_struct(d):
    cc = d["cell_class"]
    print(f"vnc.bin  N={d['N']:,}  E={d['E']:,}  flags={d['flags']}  version={d['version']}")
    print("cell_class counts:", {int(k): int(v) for k, v in
                                 zip(*np.unique(cc, return_counts=True))})
    w0 = d["weight"]
    wf, nflip = glut_fix_weights(w0, d["col_idx"], cc)
    print(f"glutamate-fix counterfactual flips {nflip:,} edges "
          f"({100*nflip/d['E']:.3f}% of E), |w| moved = {2*np.abs(w0[w0<0][:0]).sum() if False else np.abs(w0[(cc[d['col_idx']]==CLASS_MOTOR)&(w0<0)]).sum():,.0f} "
          f"of {np.abs(w0).sum():,.0f} total |w| ({100*np.abs(w0[(cc[d['col_idx']]==CLASS_MOTOR)&(w0<0)]).sum()/np.abs(w0).sum():.3f}%)")

    for include_motor in (False, True):
        tag = "premotor INCLUDES motor neurons" if include_motor else "premotor EXCLUDES motor neurons"
        pools = premotor_pools(d, include_motor)
        sizes = {k: len(v) for k, v in pools.items()}
        allp = np.concatenate([pools[k] for k in LEG_KEYS])
        uniq = len(np.unique(allp))
        print(f"\n===== {tag} =====")
        print("pool sizes:", sizes, f" union={uniq:,} (sum={len(allp):,}, overlap factor {len(allp)/uniq:.2f}x)")

        for label, wts in (("SHIPPED", w0), ("GLUT-FIXED", wf)):
            raw, dens = signed_matrix(d, pools, wts)
            npos = int((raw > 0).sum())
            wi, ac, wl, al = tripod_split(dens)
            wi_r, ac_r, _, _ = tripod_split(raw)
            print(f"\n--- {label} premotor<-premotor signed net weight (raw sums) ---")
            print(fmt_matrix(raw))
            print(f"  positive entries: {npos}/36   negative: {36-npos}/36   "
                  f"min={raw.min():.4g} max={raw.max():.4g}")
            print(f"  per-pair density (raw / |Pa|*|Pb|), within-tripod mean={wi:.6g}  "
                  f"across-tripod mean={ac:.6g}   across-minus-within={ac-wi:+.6g}")
            print(f"  raw-sum within={wi_r:.6g} across={ac_r:.6g} across-minus-within={ac_r-wi_r:+.6g}")

            # direct: premotor -> that leg's motor pool
            targets = {k: np.array(d["meta"]["motor"][k]["all"], dtype=np.int64) for k in LEG_KEYS}
            rawD, densD = signed_matrix(d, pools, wts, targets=targets)
            nposD = int((rawD > 0).sum())
            wiD, acD, _, _ = tripod_split(densD)
            print(f"--- {label} motor<-premotor (leg a's MOTOR pool <- leg b's premotor) ---")
            print(fmt_matrix(rawD))
            print(f"  positive entries: {nposD}/36   min={rawD.min():.4g} max={rawD.max():.4g}")
            print(f"  density within-tripod={wiD:.6g} across-tripod={acD:.6g} "
                  f"across-minus-within={acD-wiD:+.6g}")

            # motor <- motor directly
            rawM, densM = signed_matrix(d, targets, wts, targets=targets)
            nposM = int((rawM > 0).sum())
            wiM, acM, _, _ = tripod_split(densM)
            print(f"--- {label} motor<-motor ---")
            print(fmt_matrix(rawM))
            print(f"  positive entries: {nposM}/36  density within={wiM:.6g} across={acM:.6g} "
                  f"across-minus-within={acM-wiM:+.6g}")

            perm = tripod_permutation_control(dens)
            rank = [i for i, (_, c) in enumerate(perm)
                    if c == tuple(sorted(TRIPOD_A))][0]
            print(f"  tripod permutation control on premotor<-premotor density "
                  f"(across-minus-within, all 10 splits):")
            for i, (v, c) in enumerate(perm):
                mark = "  <== TRUE TRIPOD" if c == tuple(sorted(TRIPOD_A)) else ""
                print(f"    {i+1:2d}. {v:+.6g}  {'/'.join(c)}{mark}")
            print(f"  true tripod rank {rank+1}/10")


# ----------------------------------------------------------- LIF sweep (3,4)
def build_outgoing(d):
    """Outgoing (CSC-like) view: for each presynaptic neuron, its edges."""
    N, E = d["N"], d["E"]
    pre = d["col_idx"]
    post = post_of_edges(d["row_ptr"], E)
    order = np.argsort(pre, kind="stable")
    pre_s = pre[order]
    rows = post[order].astype(np.int32)
    eidx = order  # maps outgoing slot -> original edge index
    indptr = np.zeros(N + 1, dtype=np.int64)
    np.add.at(indptr, pre_s + 1, 1)
    np.cumsum(indptr, out=indptr)
    return indptr, rows, eidx


def simulate(indptr, rows, w_out, ext, N, n_steps, rec_idx, csr=None,
             noise=0.0, seed=0):
    """csr: optional scipy CSR of the same signed weights in INCOMING form.
    Used instead of the sparse gather when the network is densely active —
    identical arithmetic, just a faster route for the saturated regime."""
    vm = np.full(N, V_REST, np.float32)
    gx = np.zeros(N, np.float32)
    gy = np.zeros(N, np.float32)
    refrac = np.zeros(N, np.int32)
    spikes = np.zeros(N, bool)
    ext_in = (ext * EXT_GAIN).astype(np.float32)
    rec = np.zeros((n_steps, len(rec_idx)), np.uint8)
    pop = np.zeros(n_steps, np.int32)
    arange_N = np.arange(N)
    sv = np.zeros(N, np.float32)
    rng = np.random.default_rng(seed)
    for t in range(n_steps):
        act = np.flatnonzero(spikes)
        if csr is not None and act.size > 0.12 * N:
            sv[:] = spikes
            delta = csr.dot(sv).astype(np.float32)
        elif act.size:
            s = indptr[act]
            c = indptr[act + 1] - s
            tot = int(c.sum())
            if tot:
                csum = np.cumsum(c)
                base = np.repeat(s - (csum - c), c)
                slots = base + np.arange(tot, dtype=np.int64)
                delta = np.bincount(rows[slots], weights=w_out[slots],
                                    minlength=N).astype(np.float32)
            else:
                delta = np.zeros(N, np.float32)
        else:
            delta = np.zeros(N, np.float32)
        gy = gy * A_SYN + gx
        gx = gx * A_SYN + delta
        i_in = gy * W_SYN + ext_in
        if noise > 0:
            i_in = i_in + rng.normal(0.0, noise, N).astype(np.float32)
        r = refrac > 0
        vm[r] = V_RESET
        refrac[r] -= 1
        nr = ~r
        idx_nr = arange_N[nr]
        v = np.float32(V_REST) + np.float32(ALPHA) * (vm[idx_nr] - np.float32(V_REST)) + i_in[idx_nr]
        sp = v >= V_TH
        v[sp] = V_RESET
        vm[idx_nr] = v
        fired = idx_nr[sp]
        spikes = np.zeros(N, bool)
        spikes[fired] = True
        refrac[fired] = REFRAC_STEPS
        rec[t] = spikes[rec_idx]
        pop[t] = len(fired)
    return rec, pop


def boxcar(x, k=20):
    """Sliding-window mean over axis 0, matching captureRollingRate(20)."""
    cs = np.cumsum(np.vstack([np.zeros((1,) + x.shape[1:]), x]), axis=0)
    return (cs[k:] - cs[:-k]) / k


def spectrum_stats(sig, fs=1000.0):
    """Return (band_fraction 3-25 Hz, peak_freq_Hz, peak_bin)."""
    x = np.asarray(sig, float)
    x = x - x.mean()
    if not np.any(np.abs(x) > 0):
        return 0.0, 0.0, 0
    T = len(x)
    win = np.hanning(T)
    X = np.fft.rfft(x * win)
    P = np.abs(X) ** 2
    f = np.fft.rfftfreq(T, d=1.0 / fs)
    tot = P[1:].sum()
    if tot <= 0:
        return 0.0, 0.0, 0
    band = P[(f >= 3.0) & (f <= 25.0)].sum()
    pk = int(np.argmax(P[1:]) + 1)
    return float(band / tot), float(f[pk]), pk


def pearson(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a - a.mean(); b = b - b.mean()
    da, db = np.sqrt((a * a).sum()), np.sqrt((b * b).sum())
    if da < 1e-12 or db < 1e-12:
        return np.nan
    return float((a * b).sum() / (da * db))


def stage_sweep(d, use_glut_fix=False, dn_set="walking", quick=False):
    meta = d["meta"]
    N = d["N"]
    indptr, rows, eidx = build_outgoing(d)
    w_base = (glut_fix_weights(d["weight"], d["col_idx"], d["cell_class"])[0]
              if use_glut_fix else d["weight"])[eidx].astype(np.float32)
    neg = w_base < 0

    # ---- DN drive targets
    brain_meta = json.loads((ROOT / "public" / "brain.meta.json").read_text())
    if dn_set == "walk":
        # forward/backward walking command DNs only (no escape DNs)
        names = ["DNa01", "DNa02", "DNb01", "DNg13", "MDN"]
    elif dn_set == "famous":
        names = list(brain_meta["famous_dns"].keys()) + list(brain_meta["walking_circuit_dns"].keys())
    else:
        names = None
    if names is None:
        dn_idx = np.flatnonzero(d["cell_class"] == CLASS_DN_INPUT)
        dn_label = f"ALL {len(dn_idx)} DN-input neurons"
    else:
        got = []
        used = []
        for n in names:
            if n in meta["dn_inputs"]:
                got.extend(meta["dn_inputs"][n]); used.append(n)
        dn_idx = np.unique(np.array(got, dtype=np.int64))
        dn_label = f"{len(dn_idx)} neurons from {len(used)} DN types: {','.join(sorted(used))}"

    # ---- recorded neurons: the 369 leg motor neurons
    leg_all = {k: np.array(meta["motor"][k]["all"], dtype=np.int64) for k in LEG_KEYS}
    rec_idx = np.unique(np.concatenate([leg_all[k] for k in LEG_KEYS]))
    pos_of = {int(g): i for i, g in enumerate(rec_idx)}
    leg_local = {k: np.array([pos_of[int(i)] for i in v]) for k, v in leg_all.items()}
    musc_local = {}
    for k in LEG_KEYS:
        musc_local[k] = {}
        for sub, idxs in meta["motor"][k].items():
            if sub == "all":
                continue
            musc_local[k][sub] = np.array([pos_of[int(i)] for i in idxs])

    import os
    W_SCALES = [0.1, 0.3, 1.0, 3.0, 10.0] if not quick else [1.0]
    if os.environ.get("VNC_WS"):
        W_SCALES = [float(x) for x in os.environ["VNC_WS"].split(",")]
    DRIVES = [0.1, 0.2, 0.35, 0.5, 1.0, 2.0] if not quick else [0.5]
    EI = [0.25, 0.5, 1.0, 2.0, 4.0] if not quick else [1.0]
    WARM, T = 500, 3000
    if quick:
        WARM, T = 200, 1000

    print(f"# sweep  glut_fix={use_glut_fix}  dn={dn_label}")
    print(f"# w_scale={W_SCALES}  drive={DRIVES}  ei={EI}  warm={WARM} T={T} (dt=1ms)")
    print("# LOOSE rhythm := band(3-25Hz)/total > 0.5 AND peak bin != 0 AND tripod corr < -0.3")
    print("# STRICT rhythm := same but peak freq must itself lie in [3,25] Hz")
    hdr = ("wS drive ei | popHz legHz popBand popPk | leg: corr band pkHz raw_band | "
           "ratio: joint corr band pkHz | extflexcorr | L_leg L_mus S_leg S_mus")
    print(hdr)
    results = []
    t0 = time.time()
    from scipy.sparse import csr_matrix
    w_in_base = (glut_fix_weights(d["weight"], d["col_idx"], d["cell_class"])[0]
                 if use_glut_fix else d["weight"]).astype(np.float32)
    neg_in = w_in_base < 0
    for ws in W_SCALES:
        w_run = (w_base * ws).astype(np.float32)
        w_in_ws = (w_in_base * ws).astype(np.float32)
        for ei in EI:
            w_ei = w_run.copy()
            w_ei[neg] *= ei
            w_in = w_in_ws.copy()
            w_in[neg_in] *= ei
            csr = csr_matrix((w_in, d["col_idx"], d["row_ptr"]), shape=(N, N))
            for dr in DRIVES:
                ext = np.zeros(N, np.float32)
                ext[dn_idx] = dr
                rec, pop = simulate(indptr, rows, w_ei, ext, N, WARM + T, rec_idx, csr=csr)
                rec = rec[WARM:]
                pop = pop[WARM:]
                rf = rec.astype(np.float32)
                r = boxcar(rf, 20)                              # (T-19, n_mn)
                pop_hz = pop.mean() / N * 1000.0
                pop_band, pop_pk, _ = spectrum_stats(boxcar(pop.astype(float)[:, None], 20)[:, 0])
                leg_tr = {k: r[:, leg_local[k]].mean(axis=1) for k in LEG_KEYS}
                leg_raw = {k: rf[:, leg_local[k]].mean(axis=1) for k in LEG_KEYS}
                leg_hz = np.mean([leg_tr[k].mean() for k in LEG_KEYS]) * 1000.0
                A = np.mean([leg_tr[k] for k in TRIPOD_A], axis=0)
                B = np.mean([leg_tr[k] for k in TRIPOD_B], axis=0)
                corr_leg = pearson(A, B)
                bands, pks, rawbands = [], [], []
                for k in LEG_KEYS:
                    b, f, pb = spectrum_stats(leg_tr[k])
                    bands.append(b); pks.append((f, pb))
                    rawbands.append(spectrum_stats(leg_raw[k])[0])
                band_leg = float(np.mean(bands))
                rawband_leg = float(np.mean(rawbands))
                pk_leg = float(np.median([p[0] for p in pks]))
                pkbin_ok_leg = all(p[1] > 0 for p in pks)
                pkfreq_ok_leg = all(3.0 <= p[0] <= 25.0 for p in pks)

                # ---- muscle-resolved
                def pool(k, subs):
                    ii = [musc_local[k][s] for s in subs if s in musc_local[k]]
                    if not ii:
                        return np.zeros(r.shape[0]), 0
                    ii = np.concatenate(ii)
                    return r[:, ii].mean(axis=1), len(ii)

                ratio_tr = {}
                extflex = []
                for k in LEG_KEYS:
                    for joint, (ex, fl) in ANTAGONISTS.items():
                        ae, ne = pool(k, ex)
                        af, nf = pool(k, fl)
                        ratio_tr[(k, joint)] = (ae - af) / (ae + af + 1e-9)
                        c = pearson(ae, af)
                        if not np.isnan(c):
                            extflex.append(c)
                best = None
                for joint in ANTAGONISTS:
                    RA = np.mean([ratio_tr[(k, joint)] for k in TRIPOD_A], axis=0)
                    RB = np.mean([ratio_tr[(k, joint)] for k in TRIPOD_B], axis=0)
                    c = pearson(RA, RB)
                    bb, ff, pb = [], [], []
                    for k in LEG_KEYS:
                        x, y, z = spectrum_stats(ratio_tr[(k, joint)])
                        bb.append(x); ff.append(y); pb.append(z)
                    entry = dict(joint=joint, corr=c, band=float(np.mean(bb)),
                                 pk=float(np.median(ff)),
                                 pkbin_ok=all(p > 0 for p in pb),
                                 pkfreq_ok=all(3.0 <= q <= 25.0 for q in ff))
                    loose = (not np.isnan(c)) and c < -0.3 and entry["band"] > 0.5 and entry["pkbin_ok"]
                    strict = loose and entry["pkfreq_ok"]
                    score = (strict, loose, -(c if not np.isnan(c) else 0))
                    if best is None or score > best[0]:
                        best = (score, entry)
                e = best[1]
                joint, corr_m, band_m, pk_m = e["joint"], e["corr"], e["band"], e["pk"]
                ef = float(np.mean(extflex)) if extflex else float("nan")

                L_leg = (not np.isnan(corr_leg)) and corr_leg < -0.3 and band_leg > 0.5 and pkbin_ok_leg
                S_leg = L_leg and pkfreq_ok_leg
                L_mus = best[0][1]
                S_mus = best[0][0]
                print(f"{ws:4g} {dr:5g} {ei:4g} | {pop_hz:6.2f} {leg_hz:6.2f} "
                      f"{pop_band:.3f} {pop_pk:6.2f} | "
                      f"{corr_leg:+.3f} {band_leg:.3f} {pk_leg:6.2f} {rawband_leg:.3f} | "
                      f"{joint:>10s} {corr_m:+.3f} {band_m:.3f} {pk_m:6.2f} | "
                      f"{ef:+.3f} | {int(L_leg)} {int(L_mus)} {int(S_leg)} {int(S_mus)}",
                      flush=True)
                results.append(dict(ws=ws, dr=dr, ei=ei, pop_hz=pop_hz, leg_hz=leg_hz,
                                    pop_band=pop_band, pop_pk=pop_pk,
                                    corr_leg=corr_leg, band_leg=band_leg, pk_leg=pk_leg,
                                    rawband_leg=rawband_leg,
                                    joint=joint, corr_m=corr_m, band_m=band_m, pk_m=pk_m,
                                    extflex=ef, L_leg=bool(L_leg), L_mus=bool(L_mus),
                                    S_leg=bool(S_leg), S_mus=bool(S_mus)))
    n = len(results)
    alive = [r for r in results if r["leg_hz"] > 0.05]
    cl = [r["corr_leg"] for r in alive if not np.isnan(r["corr_leg"])]
    cm = [r["corr_m"] for r in alive if not np.isnan(r["corr_m"])]
    ce = [r["extflex"] for r in alive if not np.isnan(r["extflex"])]
    print(f"\n#### {n} settings, {len(alive)} with non-silent leg motor pools "
          f"({100*len(alive)/n:.1f}%)")
    for key, lbl in (("L_leg", "LOOSE whole-leg"), ("L_mus", "LOOSE muscle-resolved"),
                     ("S_leg", "STRICT whole-leg"), ("S_mus", "STRICT muscle-resolved")):
        c = sum(r[key] for r in results)
        print(f"#### rhythm {lbl:24s}: {c}/{n} = {100*c/n:.2f}%")
    for key, lbl in (("corr_leg", "tripod corr whole-leg"), ("corr_m", "tripod corr muscle"),
                     ("extflex", "within-leg ext-vs-flex corr")):
        v = [r[key] for r in alive if not np.isnan(r[key])]
        if v:
            print(f"#### {lbl:28s} range [{min(v):+.3f}, {max(v):+.3f}] median {np.median(v):+.3f}")
    # component-wise pass rates: which criterion is the binding constraint?
    for lbl, f in (("corr_leg < -0.3", lambda r: r["corr_leg"] < -0.3),
                   ("corr_musc < -0.3", lambda r: r["corr_m"] < -0.3),
                   ("band_leg > 0.5", lambda r: r["band_leg"] > 0.5),
                   ("band_musc > 0.5", lambda r: r["band_m"] > 0.5),
                   ("rawband_leg > 0.5", lambda r: r["rawband_leg"] > 0.5),
                   ("pk_leg in [3,25]", lambda r: 3 <= r["pk_leg"] <= 25),
                   ("pk_musc in [3,25]", lambda r: 3 <= r["pk_m"] <= 25)):
        c = sum(1 for r in results if not np.isnan(r["corr_leg"]) and f(r))
        print(f"#### criterion {lbl:20s} passes {c}/{n}")
    print(f"#### wall clock {time.time()-t0:.1f}s")
    return results


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "struct"
    d = load_vnc()
    if stage == "struct":
        stage_struct(d)
    elif stage == "sweep":
        gf = "--glutfix" in sys.argv
        dn = "all" if "--all-dn" in sys.argv else ("walk" if "--walk" in sys.argv else "famous")
        stage_sweep(d, use_glut_fix=gf, dn_set=dn, quick="--quick" in sys.argv)
