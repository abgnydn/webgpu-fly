"""Patch public/brain.meta.json with three additional famous DNs:

  DNp52 — forward walking command (Dallmann walking circuit, 2026)
  DNp09 — looming-evoked freezing/jump (von Reyn et al. 2014)
  MDN   — moonwalking (backward) command (Bidaye et al.)

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

NEW_DNS = {
    "DNp09": "looming-evoked freezing/jump (von Reyn et al. 2014)",
    "DNp52": "forward walking — Dallmann walking circuit (2026)",
    "MDN":   "moonwalking (backward) command — Bidaye et al.",
}


def main() -> None:
    if not ROOT_IDS_NPY.exists() or not ANN_TSV.exists():
        raise SystemExit(f"missing input data — run bash tools/download_data.sh first.")
    if not META_JSON.exists():
        raise SystemExit(f"{META_JSON} missing — run tools/build_csr.py first.")

    root_ids = np.load(ROOT_IDS_NPY)
    id_to_idx = {int(r): i for i, r in enumerate(root_ids)}

    found = {ct: [] for ct in NEW_DNS}
    with open(ANN_TSV) as f:
        rd = csv.DictReader(f, delimiter="\t")
        for row in rd:
            ct = row.get("cell_type", "")
            if ct in NEW_DNS:
                rid = int(row["root_id"])
                idx = id_to_idx.get(rid)
                if idx is not None:
                    found[ct].append(idx)

    meta = json.loads(META_JSON.read_text())
    meta.setdefault("famous_dns", {})
    meta.setdefault("famous_dn_descriptions", {})
    for ct, idxs in found.items():
        if not idxs:
            print(f"  {ct}: 0 (skipped)")
            continue
        meta["famous_dns"][ct] = idxs
        meta["famous_dn_descriptions"][ct] = NEW_DNS[ct]
        print(f"  {ct}: {len(idxs)} brain neurons → {NEW_DNS[ct]}")

    META_JSON.write_text(json.dumps(meta, indent=2))
    print(f"\npatched {META_JSON}")
    print(f"famous_dns now: {sorted(meta['famous_dns'].keys())}")


if __name__ == "__main__":
    main()
