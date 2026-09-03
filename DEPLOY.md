# Deploying webgpu-fly publicly

Asset budget:

| Asset | Size | Where it goes |
|---|---|---|
| `index.html` + `assets/index-*.js` | ~700 KB | Pages |
| `assets/mujoco-*.wasm` | 8.6 MB | Pages |
| `public/flybody/*.obj` (85 files) | 134 MB total, biggest 31 MB | R2 |
| `public/flybody/*.xml` (2 files) | <1 MB | R2 |
| `public/flybody.bundle.bin` | ~140 MB | R2 |
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
# Enable R2 in dash.cloudflare.com → R2 → Enable (requires payment
# method on file; won't be billed unless you exceed the 10 GB free tier)
wrangler r2 bucket create webgpu-fly-assets
wrangler r2 bucket dev-url enable webgpu-fly-assets   # captures the public URL
wrangler r2 bucket cors set webgpu-fly-assets --file r2-cors.json
wrangler pages project create webgpu-fly --production-branch=main

# every time you regenerate brain.bin / vnc.bin
npm run deploy:r2          # uploads big assets to R2 (~300 MB, ~10 min)

# every code push
npm run deploy             # builds (with R2 URLs from .env.production)
                           # then deploys to Cloudflare Pages
```

Vite reads `import.meta.env.VITE_*` at *build* time, not runtime, so the
URLs need to be in the local environment when `npm run build:slim`
runs. Easiest is a `.env.production` file (gitignored) containing:

```
VITE_BRAIN_URL=https://<r2-public-host>/brain.bin
VITE_BRAIN_META_URL=https://<r2-public-host>/brain.meta.json
VITE_VNC_URL=https://<r2-public-host>/vnc.bin
VITE_VNC_META_URL=https://<r2-public-host>/vnc.meta.json
VITE_FLYBODY_URL=https://<r2-public-host>/flybody
VITE_FLYBODY_BUNDLE_URL=https://<r2-public-host>/flybody.bundle.bin
VITE_WALKING_POLICY_URL=https://<r2-public-host>/walking-policy.bin
VITE_WALKING_OBS_NORM_URL=https://<r2-public-host>/walking-obs-norm.bin
VITE_WALKING_REF_URL=https://<r2-public-host>/walking-ref.bin
VITE_ASSET_MANIFEST_URL=https://<r2-public-host>/assets.json
```

`<r2-public-host>` is the bucket's r2.dev subdomain (printed by the
`dev-url enable` command), or your custom domain.

`VITE_FLYBODY_URL` is legacy — `src/physics.ts` loads only the baked
bundle, so `VITE_FLYBODY_BUNDLE_URL` is the one that must be set.

`r2-cors.json`:
```json
{
  "rules": [{
    "allowed": {
      "origins": ["*"],
      "methods": ["GET", "HEAD"],
      "headers": ["*"]
    },
    "exposeHeaders": ["Content-Length", "Content-Type"],
    "maxAgeSeconds": 86400
  }]
}
```

`public/_headers` already sets long immutable cache on the JS bundle
and WASM. The R2 objects get `Cache-Control: public, max-age=31536000,
immutable` from `tools/upload_to_r2.sh` at upload time
(`--cache-control`), and CORS comes from `r2-cors.json` via the
`wrangler r2 bucket cors set` step above — nothing further to configure.

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
