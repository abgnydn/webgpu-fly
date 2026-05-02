// Playwright config — runs e2e smoke tests against the live dev server
// at :8766 with WebGPU enabled. WebGPU only works in a real GPU context,
// so we launch Chromium with the headed flag and the unsafe-webgpu flag
// (Chromium needs explicit opt-in for WebGPU outside of stable Chrome).
//
// To run: npm run test:e2e (boots vite + executes tests).

import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  timeout: 240_000,        // brain compile + flybody MJCF can take ~60s
  expect: { timeout: 30_000 },
  fullyParallel: false,    // GPU pipeline doesn't like sharing
  workers: 1,
  reporter: [["list"], ["html", { open: "never" }]],
  use: {
    baseURL: "http://127.0.0.1:8766",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    video: "retain-on-failure",
    headless: false,        // WebGPU needs a real surface
    launchOptions: {
      args: [
        "--enable-unsafe-webgpu",
        "--enable-features=Vulkan,WebGPU",
        "--disable-dawn-features=disallow_unsafe_apis",
      ],
    },
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
  webServer: {
    command: "npm run dev",
    url: "http://127.0.0.1:8766",
    reuseExistingServer: true,
    timeout: 60_000,
  },
});
