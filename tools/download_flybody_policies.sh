#!/usr/bin/env bash
# download_flybody_policies.sh — fetch the 6.5 MB trained policy bundle
# from the Vaxenburg et al. 2025 (Nature) Figshare deposit, then run
# tools/extract_walking_policy.py to emit public/walking-policy.bin.
#
# The dataset DOI corresponds to:
#   https://janelia.figshare.com/articles/dataset/MuJoCo_fruit_fly_body_model_datasets_supporting_Whole-body_simulation_of_realistic_fruit_fly_locomotion_with_deep_reinforcement_learning_/25309105
#
# Note: the figshare CDN sometimes 403's bare curl; route through
# janelia.figshare.com/ndownloader instead which signs the request.

set -euo pipefail

OUT="data/flybody-policies"
mkdir -p "$OUT"

URL="https://janelia.figshare.com/ndownloader/files/44815195"

echo "→ trained-fly-policies.zip (6.5 MB)"
curl -sL --output /tmp/trained-fly-policies.zip \
  -A "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36" \
  -H "Referer: https://janelia.figshare.com/" \
  "$URL"

echo "→ unzipping into $OUT/"
unzip -q -o /tmp/trained-fly-policies.zip -d "$OUT"
rm /tmp/trained-fly-policies.zip

echo
echo "Contents:"
ls -d $OUT/*/ 2>/dev/null
echo
echo "Next: .venv-tf/bin/python tools/extract_walking_policy.py"
echo "      → public/walking-policy.bin (4.8 MB)"
