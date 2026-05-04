# Deploying webgpu-fly publicly

Asset budget:

| Asset | Size | Where it goes |
|---|---|---|
| `index.html` + `assets/index-*.js` | ~700 KB | Pages |
| `assets/mujoco-*.wasm` | 8.6 MB | Pages |
| `public/flybody/*.obj` (85 files) | 134 MB total, biggest 31 MB | R2 |
| `public/flybody/*.xml` (2 files) | <1 MB | R2 |
| `public/brain.bin` | 120 MB | R2 |
| `public/brain.meta.json` | 1 KB | R2 |
| `public/vnc.bin` | 43 MB | R2 |
| `public/vnc.meta.json` | 34 KB | R2 |

`head_red.obj` alone is 31 MB, over Cloudflare Pages's 25 MB per-file
cap, so we offload all heavy assets to **R2** and keep Pages slim.

## Path 1 — Cloudflare (recommended)

```bash
# one time
wrangler login
wrangler r2 bucket create webgpu-fly-assets
# enable public access on the bucket from the Cloudflare dashboard
# (or set up a custom domain; r2.dev URLs are rate-limited but free)

# every time you regenerate brain.bin / vnc.bin
npm run deploy:r2          # uploads big assets to R2 (~300 MB)

# every code push
npm run deploy             # builds + deploys to Cloudflare Pages
```

After the first `deploy:r2`, set these env vars in the Cloudflare Pages
project (Settings → Environment Variables, both Preview and Production):

```
VITE_BRAIN_URL       = https://<r2-public-host>/brain.bin
VITE_BRAIN_META_URL  = https://<r2-public-host>/brain.meta.json
VITE_VNC_URL         = https://<r2-public-host>/vnc.bin
VITE_VNC_META_URL    = https://<r2-public-host>/vnc.meta.json
VITE_FLYBODY_URL     = https://<r2-public-host>/flybody
```

`<r2-public-host>` is either your bucket's `r2.dev` subdomain or your
custom domain. Find it in Cloudflare dashboard → R2 → bucket →
Settings → Public access.

`public/_headers` already sets long immutable cache on the JS bundle
and WASM. The R2 bucket should also serve `Cache-Control:
public, max-age=31536000, immutable` — set this once via:

```bash
wrangler r2 bucket cors put webgpu-fly-assets --cors-rules '[
  { "AllowedOrigins": ["https://your-pages.pages.dev","https://your-domain.com"],
    "AllowedMethods": ["GET","HEAD"],
    "AllowedHeaders": ["*"],
    "MaxAgeSeconds": 86400 }
]'
```

## Path 2 — Vercel

`vercel.json` and the `deploy:vercel` script are still here as fallback.
Vercel Pro tolerates the 120 MB `brain.bin` directly; Hobby caps files
at 50 MB so you'd need the same R2-style external host (or a GitHub
Release).

## Cache strategy

Three layers cooperate:

1. **R2** serves with long-immutable `Cache-Control` so Cloudflare's
   edge fronts the bytes globally.
2. **Pages `_headers`** does the same for the JS and WASM.
3. **IndexedDB cache** (`src/cache.ts`) stores the flybody OBJs
   browser-side after first fetch, so a returning visitor skips the
   network entirely.
