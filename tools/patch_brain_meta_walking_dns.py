"""Patch public/brain.meta.json with additional famous neurons relevant
to the forward-walking circuit:

  RRN   — Roadrunner neurons, central-brain forward-walking command
          (cell_type CB0257 in FAFB/FlyWire; pair of neurons identified
          in Dallmann et al. 2026, bioRxiv 10.64898/2026.01.04.697356)
  DNp52 — leg-cluster DN downstream of RRN+BPN (Dallmann 2026)
  DNp09 — looming-evoked freezing/jump (von Reyn et al. 2014)
  MDN   — moonwalking (backward) command (Bidaye et al.)

Plus a `walking_circuit_dns` group dump (the 20 cell types Dallmann
identified as receiving net excitation from RRN+BPN — both leg-cluster
and lower-tectulum cluster) for any future UI / analysis.

These are not in the original 5-DN set baked into brain.bin's metadata.
Re-running tools/build_csr.py is the canonical regen path, but it
requires the full feather + annotations download. This script does a
targeted lookup against the same data sources (proofread_root_ids_783.npy
plus Supplemental_file1_neuron_annotations.tsv) to add only the
famous_dns / famous_dn_descriptions entries — no brain.bin touch.

Run after a fresh download:
  bash tools/download_data.sh
  .venv-flygym/bin/python tools/patch_brain_meta_walking_dns.py
"""
import csv
import json
from pathlib import Path

import numpy as np

ROOT_IDS_NPY = Path("data/raw/proofread_root_ids_783.npy")
ANN_TSV = Path("data/raw/flywire_annotations/supplemental_files/"
               "Supplemental_file1_neuron_annotations.tsv")
META_JSON = Path("public/brain.meta.json")

# Famous-DN buttons (UI). RRN is technically not a DN but is the command
# entry point of the Dallmann walking circuit, so it lives in the same
# button row.
NEW_DNS = {
    "RRN":   "forward walking — Roadrunner neurons (Dallmann 2026); cell_type CB0257",
    "DNp09": "looming-evoked freezing/jump (von Reyn et al. 2014)",
    "DNp52": "forward walking — Dallmann walking circuit (2026)",
    "MDN":   "moonwalking (backward) command — Bidaye et al.",
}

# Cell-types referenced for the new "RRN" button — RRN is annotated in
# FAFB/FlyWire under cell_type=CB0257.
DN_CELL_TYPE_OVERRIDES = {
    "RRN": "CB0257",
}

# 21 DN cell types Dallmann 2026 places in the leg/LTct clusters
# downstream of RRN+BPN. Stored under brain.meta.json["walking_circuit_dns"]
# for any future "fire the whole pathway" preset; not a button each.
WALKING_CIRCUIT_CELL_TYPES = (
    # Leg cluster (project to leg neuropils, ~71% of output)
    "DNp52", "DNg101", "DNg102", "DNp64", "DNge050", "DNd05",
    "DNge048", "DNa45", "DNge082", "DNpe020", "DNg44", "DNge103",
    # LTct cluster (project to lower tectulum, ~31% of output)
    "DNpe053", "DNp13", "DNp42", "DNge150", "DNp68", "DNp69",
    "DNpe042", "DNp45", "DNp55",
)


def main() -> None:
    if not ROOT_IDS_NPY.exists() or not ANN_TSV.exists():
        raise SystemExit(f"missing input data — run bash tools/download_data.sh first.")
    if not META_JSON.exists():
        raise SystemExit(f"{META_JSON} missing — run tools/build_csr.py first.")

    root_ids = np.load(ROOT_IDS_NPY)
    id_to_idx = {int(r): i for i, r in enumerate(root_ids)}

    # Build cell_type → label lookup. For RRN the FlyWire annotation
    # uses CB0257, not "RRN" — apply the override.
    label_to_search = {
        DN_CELL_TYPE_OVERRIDES.get(label, label): label for label in NEW_DNS
    }
    found = {label: [] for label in NEW_DNS}
    found_circuit = {ct: [] for ct in WALKING_CIRCUIT_CELL_TYPES}

    with open(ANN_TSV) as f:
        rd = csv.DictReader(f, delimiter="\t")
        for row in rd:
            ct = row.get("cell_type", "")
            rid = int(row["root_id"])
            idx = id_to_idx.get(rid)
            if idx is None:
                continue
            if ct in label_to_search:
                found[label_to_search[ct]].append(idx)
            if ct in found_circuit:
                found_circuit[ct].append(idx)

    meta = json.loads(META_JSON.read_text())
    meta.setdefault("famous_dns", {})
    meta.setdefault("famous_dn_descriptions", {})
    for label, idxs in found.items():
        if not idxs:
            print(f"  {label}: 0 (skipped — cell_type {DN_CELL_TYPE_OVERRIDES.get(label, label)} not in annotations)")
            continue
        meta["famous_dns"][label] = idxs
        meta["famous_dn_descriptions"][label] = NEW_DNS[label]
        print(f"  {label}: {len(idxs)} brain neurons → {NEW_DNS[label]}")

    # Walking-circuit DN catalog (Dallmann 2026 supp fig 2c). Stored as
    # cell_type → indices; not promoted to buttons. Kept for any future
    # whole-circuit stim preset or visualization.
    meta["walking_circuit_dns"] = {
        ct: idxs for ct, idxs in found_circuit.items() if idxs
    }
    n_circuit = sum(len(v) for v in meta["walking_circuit_dns"].values())
    n_types = len(meta["walking_circuit_dns"])
    print(f"\nwalking_circuit_dns: {n_types} cell types / {n_circuit} brain neurons "
          f"(Dallmann 2026 leg + LTct clusters)")

    META_JSON.write_text(json.dumps(meta, indent=2))
    print(f"\npatched {META_JSON}")
    print(f"famous_dns now: {sorted(meta['famous_dns'].keys())}")


if __name__ == "__main__":
    main()
