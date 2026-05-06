// fixtures.ts — persistent Chromium context fixture.
//
// Default playwright spawns a fresh BrowserContext per test → fresh
// IDB → every test re-fetches the ~170 MB of binaries (~30s each).
// 25 tests × 30s ≈ 12 min wasted on cold IDB.
//
// This fixture launches ONE persistent Chromium with a fixed
// user-data-dir for the whole worker. IDB persists across tests AND
// across `npm run test:e2e` invocations. After the first run on a
// machine, every subsequent run hits the IDB cache instantly for
// brain.bin / vnc.bin / walking-policy.bin / walking-obs-norm.bin /
// walking-ref.bin / flybody OBJs.
//
// Cache invalidation: the cache key is the full URL including the
// ?v=<sha> stamp from /assets.json — a new build naturally produces
// new entries and old ones are eventually evicted.
//
// Usage: import `test` from this file instead of @playwright/test.

import {
  chromium,
  test as base,
  type BrowserContext,
  type Page,
} from "@playwright/test";
import { mkdirSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

const PROFILE_DIR = process.env.PLAYWRIGHT_PROFILE_DIR
  ?? join(homedir(), ".cache", "webgpu-fly-e2e-chromium");

mkdirSync(PROFILE_DIR, { recursive: true });

interface WorkerFixtures {
  persistentContext: BrowserContext;
}

interface TestFixtures {
  page: Page;
}

export const test = base.extend<TestFixtures, WorkerFixtures>({
  persistentContext: [async ({}, use) => {
    const ctx = await chromium.launchPersistentContext(PROFILE_DIR, {
      headless: false,
      args: [
        "--enable-unsafe-webgpu",
        "--enable-features=Vulkan,WebGPU",
        "--disable-dawn-features=disallow_unsafe_apis",
      ],
      viewport: { width: 1280, height: 720 },
    });
    await use(ctx);
    await ctx.close();
  }, { scope: "worker" }],

  page: async ({ persistentContext }, use) => {
    const page = await persistentContext.newPage();
    await use(page);
    await page.close();
  },
});

export { expect } from "@playwright/test";
export type { Page } from "@playwright/test";
