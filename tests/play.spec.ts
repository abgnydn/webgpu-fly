// play.spec.ts — end-to-end smoke for the game-first /play.html page.
//
// Mirrors the game.spec.ts boot check: verify the Game HUD builds and the
// play-page extras are visible after the brain + body finish loading.

import { test, expect } from "./fixtures";

test.describe("play page", () => {
  test.setTimeout(180_000);

  test("boot + Game HUD + play extras render", async ({ page }) => {
    await page.goto("/play.html");

    // Wait for play-mode readiness signal in the hidden log sink.
    await page.waitForFunction(
      () => /play mode: ready/.test(
        document.querySelector("#out")?.textContent ?? "",
      ),
      null,
      { timeout: 120_000 },
    );

    await expect(page.locator("#game-hud")).toBeVisible();
    await expect(page.locator("#game-keys")).toBeVisible();
    await expect(page.locator("#game-overlay")).toBeVisible();
    await expect(page.locator("#play-heading")).toBeVisible();
    await expect(page.locator("#play-best")).toBeVisible();
    await expect(page.locator("#retina-mini")).toBeVisible();

    await page.screenshot({ path: "test-results/play-boot.png" });
  });
});
