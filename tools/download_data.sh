#!/usr/bin/env bash
# Pull the v1 FlyWire data into ./data/raw/
# Total: ~855 MB
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p data/raw
cd data/raw

ZENODO="https://zenodo.org/records/10676866/files"

echo "[1/3] proofread_connections_783.feather (852 MB)"
if [ ! -f proofread_connections_783.feather ]; then
  curl -L --progress-bar -o proofread_connections_783.feather \
    "$ZENODO/proofread_connections_783.feather?download=1"
else
  echo "    already present, skipping"
fi

echo "[2/3] proofread_root_ids_783.npy (1.1 MB)"
if [ ! -f proofread_root_ids_783.npy ]; then
  curl -L --progress-bar -o proofread_root_ids_783.npy \
    "$ZENODO/proofread_root_ids_783.npy?download=1"
else
  echo "    already present, skipping"
fi

echo "[3/3] flywire_annotations TSV (cloning shallow)"
if [ ! -d flywire_annotations ]; then
  git clone --depth 1 https://github.com/flyconnectome/flywire_annotations.git
else
  echo "    already cloned, skipping"
fi

echo
echo "Done. Files in data/raw/:"
ls -lh

echo
echo "Next: python3 tools/build_csr.py"
