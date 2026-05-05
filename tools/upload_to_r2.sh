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
# After this script, set the Pages project's environment variables:
#   VITE_BRAIN_URL       = https://<bucket>.<account>.r2.dev/brain.bin
#   VITE_BRAIN_META_URL  = https://<bucket>.<account>.r2.dev/brain.meta.json
#   VITE_VNC_URL         = https://<bucket>.<account>.r2.dev/vnc.bin
#   VITE_VNC_META_URL    = https://<bucket>.<account>.r2.dev/vnc.meta.json
#   VITE_FLYBODY_URL     = https://<bucket>.<account>.r2.dev/flybody

set -euo pipefail

BUCKET="${R2_BUCKET:-webgpu-fly-assets}"

put() {
  local path="$1" key="$2" ctype="$3"
  if [ ! -f "$path" ]; then
    echo "  ! missing $path — skipping"
    return
  fi
  local size
  size=$(stat -f "%z" "$path" 2>/dev/null || stat -c "%s" "$path")
  printf "  → %-32s (%6.1f MB)  → r2://%s/%s\n" \
    "$key" "$(echo "scale=1; $size / 1048576" | bc)" "$BUCKET" "$key"
  wrangler r2 object put "$BUCKET/$key" \
    --file "$path" --content-type "$ctype" --remote >/dev/null
}

echo "uploading core LIF blobs …"
put "public/brain.bin"      "brain.bin"      "application/octet-stream"
put "public/brain.meta.json" "brain.meta.json" "application/json"
put "public/vnc.bin"        "vnc.bin"        "application/octet-stream"
put "public/vnc.meta.json"  "vnc.meta.json"  "application/json"
put "public/walking-policy.bin" "walking-policy.bin" "application/octet-stream"

echo "uploading flybody MJCF + meshes …"
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
