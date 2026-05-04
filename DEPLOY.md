# Deploying webgpu-fly publicly

The build outputs to `dist/`. Three categories of assets:

| Asset | Size | Notes |
|---|---|---|
| JS bundle + index.html | ~700 KB | tiny |
| mujoco WASM | ~8.6 MB | fits anywhere |
| `public/flybody/*.obj` (85 files) | 134 MB | largest single ~10 MB |
| `public/brain.bin` | 120 MB | one big blob |

## Path 1 — Vercel Pro (simplest)

```bash
vercel login
npm run deploy
```

Pro plan tolerates the 120 MB `brain.bin` directly. `vercel.json` already
sets immutable cache headers on the heavy assets.

## Path 2 — Vercel Hobby + external assets (free)

Hobby caps individual files at 50 MB, so `brain.bin` has to live somewhere
else (GitHub Release, Cloudflare R2, S3). Same for any flybody mesh that
crosses the 50 MB line — most are < 5 MB so usually only `brain.bin`
matters.

1. Upload `public/brain.bin` and `public/brain.meta.json` to a GitHub
   Release on this repo:

   ```bash
   gh release create v0.1.0 \
     public/brain.bin \
     public/brain.meta.json
   ```

   Note the public download URLs.

2. Optional: upload the flybody mesh dir too if you want to slim the
   deploy further (~134 MB).

   ```bash
   tar -czf flybody.tar.gz -C public flybody
   gh release upload v0.1.0 flybody.tar.gz
   ```

   You'd then need to extract on a CDN that serves them at a known URL.

3. Set env vars in your Vercel project (Settings → Environment Variables):

   - `VITE_BRAIN_URL` = `https://github.com/<you>/webgpu-fly/releases/download/v0.1.0/brain.bin`
   - `VITE_BRAIN_META_URL` = `https://.../brain.meta.json`
   - `VITE_FLYBODY_URL` = `https://your-cdn.example.com/flybody` (only if you offloaded flybody too)

4. Deploy:

   ```bash
   vercel --prod
   ```

## Path 3 — GitHub Pages

Pages caps individual files at 100 MB and total repo at 1 GB. `brain.bin`
at 120 MB doesn't fit. Use Path 2's release-host trick + push only `dist/`
without the heavy assets to a `gh-pages` branch.

## CORS

GitHub Releases serve with permissive CORS so cross-origin fetches from
your Vercel domain work without configuration. Cloudflare R2 needs a
CORS rule allowing your domain.

## Cache strategy

`vercel.json` tags `brain.bin`, `brain.meta.json`, the flybody dir, and
the WASM blob with `Cache-Control: public, max-age=31536000, immutable`.
The runtime also keeps an IndexedDB copy (`src/cache.ts`) so a returning
visitor pays zero network bandwidth on subsequent loads.
