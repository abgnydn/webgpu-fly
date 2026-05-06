#!/usr/bin/env bash
# upload_to_r2.sh — push the heavy assets (brain.bin, vnc.bin, meta
# files, all flybody OBJs + XMLs) to a Cloudflare R2 bucket so the
# Pages deploy can stay under the 25 MB / file limit.
#
# Prerequisite: a Cloudflare R2 bucket named "webgpu-fly-assets"
# (or override BUCKET below) with public access enabled. Wrangler
# must be authenticated via `wrangler login`.
#
# Run once on a fresh build of the assets, or whenever you regenerate
# brain.bin / vnc.bin via tools/build_csr.py / tools/build_vnc.py.
#
# Cache strategy: hashed-version-stamps via assets.json.
#   - assets.json itself: no-cache (must always be fresh — it is the
#     manifest the runtime consults to know which versions to load)
#   - all binaries: public, max-age=1y, immutable. The runtime appends
#     ?v=<sha> to their URLs from assets.json, so the browser cache
#     key is unique per build. New build → new sha → fresh fetch
#     transparently. Subsequent same-build loads skip the network.
#
# After this script, set the Pages project's environment variables:
#   VITE_BRAIN_URL       = https://<bucket>.<account>.r2.dev/brain.bin
#   VITE_BRAIN_META_URL  = https://<bucket>.<account>.r2.dev/brain.meta.json
#   VITE_VNC_URL         = https://<bucket>.<account>.r2.dev/vnc.bin
#   VITE_VNC_META_URL    = https://<bucket>.<account>.r2.dev/vnc.meta.json
#   VITE_FLYBODY_URL     = https://<bucket>.<account>.r2.dev/flybody

set -euo pipefail
# Force C locale so printf "%.1f" gets `.` for decimal regardless of
# the user's LANG (Turkish locale uses comma, breaks bc/printf parsing).
export LC_ALL=C

BUCKET="${R2_BUCKET:-webgpu-fly-assets}"

# Refresh the manifest first so the sha hashes match what we're about
# to upload. Cheap (a few sha256s over public/).
echo "regenerating public/assets.json …"
if [ -x .venv-flygym/bin/python ]; then
  .venv-flygym/bin/python tools/write_manifest.py
else
  python3 tools/write_manifest.py
fi
echo

CC_IMMUTABLE="public, max-age=31536000, immutable"
CC_NOCACHE="no-cache, must-revalidate"

put() {
  local path="$1" key="$2" ctype="$3" cc="${4:-$CC_IMMUTABLE}"
  if [ ! -f "$path" ]; then
    echo "  ! missing $path — skipping"
    return
  fi
  local size
  size=$(stat -f "%z" "$path" 2>/dev/null || stat -c "%s" "$path")
  printf "  → %-32s (%6.1f MB)  → r2://%s/%s\n" \
    "$key" "$(echo "scale=1; $size / 1048576" | bc)" "$BUCKET" "$key"
  npx --yes wrangler r2 object put "$BUCKET/$key" \
    --file "$path" \
    --content-type "$ctype" \
    --cache-control "$cc" \
    --remote >/dev/null
}

echo "uploading version manifest (no-cache) …"
put "public/assets.json" "assets.json" "application/json" "$CC_NOCACHE"

echo "uploading core LIF blobs (immutable) …"
put "public/brain.bin"            "brain.bin"            "application/octet-stream"
put "public/brain.meta.json"      "brain.meta.json"      "application/json"
put "public/vnc.bin"              "vnc.bin"              "application/octet-stream"
put "public/vnc.meta.json"        "vnc.meta.json"        "application/json"
put "public/walking-policy.bin"   "walking-policy.bin"   "application/octet-stream"
put "public/walking-obs-norm.bin" "walking-obs-norm.bin" "application/octet-stream"

echo "uploading flybody MJCF + meshes (immutable) …"
for f in public/flybody/*.xml; do
  [ -f "$f" ] || continue
  put "$f" "flybody/$(basename "$f")" "application/xml"
done
for f in public/flybody/*.obj; do
  [ -f "$f" ] || continue
  put "$f" "flybody/$(basename "$f")" "model/obj"
done

echo
echo "done. Set env vars in Cloudflare Pages → Settings → Environment Variables:"
echo "  VITE_BRAIN_URL      = https://<r2-public-url>/brain.bin"
echo "  VITE_BRAIN_META_URL = https://<r2-public-url>/brain.meta.json"
echo "  VITE_VNC_URL        = https://<r2-public-url>/vnc.bin"
echo "  VITE_VNC_META_URL   = https://<r2-public-url>/vnc.meta.json"
echo "  VITE_FLYBODY_URL    = https://<r2-public-url>/flybody"
echo "  VITE_ASSET_MANIFEST_URL = https://<r2-public-url>/assets.json"
