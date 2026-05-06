"""Patch public/brain.meta.json with additional famous neurons relevant
to the forward-walking circuit:

  RRN   — Roadrunner neurons, central-brain forward-walking command
          (cell_type CB0257 in FAFB/FlyWire; pair of neurons identified
          in Dallmann et al. 2026, bioRxiv 10.64898/2026.01.04.697356)
  BPN   — Bolt protocerebral neurons, the other excitatory forward-walking
          command (Bidaye et al. 2020). Curated by Dallmann 2026 as 32
          neurons spanning 4 sub-communities (BPN1-4) and multiple
          FAFB cell_types (CL210, SMP459, SMP460, SMP461, plus
          unannotated). Resolved by root_id from Dallmann's
          Supplementary Table 1.
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
DALLMANN_TABLE_1 = Path("data/raw/dallmann_2026/supplementary_table_1.xlsx")
META_JSON = Path("public/brain.meta.json")

# Famous-DN buttons (UI). RRN is technically not a DN but is the command
# entry point of the Dallmann walking circuit, so it lives in the same
# button row.
NEW_DNS = {
    "RRN":   "forward walking — Roadrunner neurons (Dallmann 2026); cell_type CB0257",
    "BPN":   "forward walking — Bolt protocerebral neurons (Bidaye 2020); 32 cells across BPN1-4",
    "DNp09": "looming-evoked freezing/jump (von Reyn et al. 2014)",
    "DNp52": "forward walking — Dallmann walking circuit (2026)",
    "MDN":   "moonwalking (backward) command — Bidaye et al.",
}

# Cell-types referenced for the new "RRN" button — RRN is annotated in
# FAFB/FlyWire under cell_type=CB0257.
DN_CELL_TYPE_OVERRIDES = {
    "RRN": "CB0257",
}

# Buttons whose neurons cannot be resolved from a single FlyWire cell_type
# and must be looked up by root_id from Dallmann 2026 Supp Table 1's
# `community_name` column.
COMMUNITY_NAME_LOOKUPS = {
    "BPN": ("BPN1", "BPN2", "BPN3", "BPN4"),  # Bidaye 2020 Bolt PNs
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


def _load_dallmann_table_1():
    """Returns list of (connectome_dataset, root_id, cell_type, community_name)
    tuples from Dallmann 2026 Supplementary Table 1, FlyWire rows only."""
    if not DALLMANN_TABLE_1.exists():
        return []
    import openpyxl
    wb = openpyxl.load_workbook(DALLMANN_TABLE_1, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    out = []
    for r in rows[1:]:
        ds, rid, ct, cn = r[0], r[1], r[2], r[3]
        if ds == "flywire_v783" and rid is not None:
            out.append((ds, int(rid), str(ct).strip() if ct else "", str(cn).strip() if cn else ""))
    return out


def main() -> None:
    if not ROOT_IDS_NPY.exists() or not ANN_TSV.exists():
        raise SystemExit(f"missing input data — run bash tools/download_data.sh first.")
    if not META_JSON.exists():
        raise SystemExit(f"{META_JSON} missing — run tools/build_csr.py first.")
    if COMMUNITY_NAME_LOOKUPS and not DALLMANN_TABLE_1.exists():
        raise SystemExit(
            f"{DALLMANN_TABLE_1} missing — required for community-name "
            f"lookups (e.g. BPN). Download Supplementary Table 1 from "
            f"bioRxiv 10.64898/2026.01.04.697356 and save as "
            f"{DALLMANN_TABLE_1}."
        )

    root_ids = np.load(ROOT_IDS_NPY)
    id_to_idx = {int(r): i for i, r in enumerate(root_ids)}

    # Build cell_type → label lookup. For RRN the FlyWire annotation
    # uses CB0257, not "RRN" — apply the override.
    cell_type_keys = {label for label in NEW_DNS if label not in COMMUNITY_NAME_LOOKUPS}
    label_to_search = {
        DN_CELL_TYPE_OVERRIDES.get(label, label): label for label in cell_type_keys
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

    # Resolve community-name buttons from Dallmann's Supp Table 1 by root_id.
    if COMMUNITY_NAME_LOOKUPS:
        table = _load_dallmann_table_1()
        for label, community_names in COMMUNITY_NAME_LOOKUPS.items():
            for _, rid, _, cn in table:
                if cn in community_names:
                    idx = id_to_idx.get(rid)
                    if idx is not None:
                        found[label].append(idx)

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
