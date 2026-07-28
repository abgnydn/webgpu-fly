#!/usr/bin/env bash
# download_manc.sh — fetch the MANC v1.0 traced-neurons + traced-connections
# CSV exports from Janelia's public Google Cloud bucket. Together they're
# ~76 MB and contain everything needed to build the spine equivalent of
# brain.bin.
#
# The bucket is public, no auth required:
#   https://console.cloud.google.com/storage/browser/flyem-manc-exports
#
# After this finishes, run `tools/build_vnc.py` to produce public/vnc.bin.

set -euo pipefail

cd "$(dirname "$0")/.."

BUCKET="https://storage.googleapis.com/flyem-manc-exports/v1.0/manc-traced-adjacencies-v1.0"
PROPS_BUCKET="https://storage.googleapis.com/flyem-manc-exports/v1.0"

OUT="data/manc"
mkdir -p "$OUT"

echo "→ traced-neurons.csv (0.6 MB) — neuron metadata"
curl -L --progress-bar -o "$OUT/traced-neurons.csv" \
  "$BUCKET/traced-neurons.csv"

echo "→ traced-connections.csv (75 MB) — pre/post/weight aggregated edges"
curl -L --progress-bar -o "$OUT/traced-connections.csv" \
  "$BUCKET/traced-connections.csv"

echo "→ manc-v1.0-neuron-properties.feather (17 MB) — full neuron props"
curl -L --progress-bar -o "$OUT/neuron-properties.feather" \
  "$PROPS_BUCKET/manc-v1.0-neuron-properties.feather"

echo
echo "MANC files in $OUT:"
ls -lh "$OUT"
echo
echo "Next: python3 tools/build_vnc.py  →  public/vnc.bin"
