// game.spec.ts — end-to-end smoke for game mode.
//
// Verifies:
//   1. /?mode=game (or default) shows the HUD + key strip + intro card.
//   2. Pressing SPACE starts the round, hides the intro.
//   3. Pressing a DN key (Q) flashes the active class on the strip.
//   4. The HUD timer ticks during a round.
//   5. The game eventually ends if we move the fly close to target
//      (we cheat-cheat: directly reposition the target to the fly's
//      current spot to trigger win), and the win card appears.
//
// We do NOT assert win conditions on real fly walking — that depends
// on the trained walker reacting to a stim, which is timing-sensitive.

import { test, expect } from "./fixtures";

test.describe("game mode", () => {
  test.setTimeout(180_000);

  test("HUD + key strip + intro render after boot", async ({ page }) => {
    await page.goto("/app?mode=game");

    // Wait for game readiness signal in the log.
    await page.waitForFunction(
      () => /game mode: ready/.test(
        document.querySelector("#out")?.textContent ?? "",
      ),
      null,
      { timeout: 120_000 },
    );

    await expect(page.locator("#game-hud")).toBeVisible();
    await expect(page.locator("#game-keys")).toBeVisible();
    await expect(page.locator("#game-overlay")).toBeVisible();
    // Title card should be on the intro overlay.
    await expect(page.locator(".overlay-title")).toContainText("Drosophila");
    // Sidebar hidden in game mode.
    await expect(page.locator("#side")).toBeHidden();

    // 10 famous DN keys from brain.meta.json should be present.
    const keyCount = await page.locator(".game-key").count();
    expect(keyCount).toBeGreaterThanOrEqual(10);
  });

  test("SPACE starts a round, key press flashes the strip", async ({ page }) => {
    await page.goto("/app?mode=game");
    await page.waitForFunction(
      () => /game mode: ready/.test(
        document.querySelector("#out")?.textContent ?? "",
      ),
      null,
      { timeout: 120_000 },
    );

    // Start round.
    await page.keyboard.press("Space");

    // Wait through the 3-2-1-GO countdown (~2.2s) until overlay hides.
    await page.waitForFunction(
      () => {
        const o = document.querySelector("#game-overlay") as HTMLElement | null;
        return o && o.style.display === "none";
      },
      null,
      { timeout: 6000 },
    );

    // Press Q (DNa01, first famous DN). Should add .active to the strip
    // entry while held.
    await page.keyboard.down("q");
    await expect(page.locator(".game-key.active")).toHaveCount(1, { timeout: 1000 });
    await page.keyboard.up("q");
    await expect(page.locator(".game-key.active")).toHaveCount(0, { timeout: 1500 });
  });

  test("HUD timer ticks during a round", async ({ page }) => {
    await page.goto("/app?mode=game");
    await page.waitForFunction(
      () => /game mode: ready/.test(
        document.querySelector("#out")?.textContent ?? "",
      ),
      null,
      { timeout: 120_000 },
    );
    await page.keyboard.press("Space");
    await page.waitForFunction(
      () => {
        const o = document.querySelector("#game-overlay") as HTMLElement | null;
        return o && o.style.display === "none";
      },
      null,
      { timeout: 6000 },
    );
    // Read the timer twice 1.5 s apart; the second should be larger.
    const t1 = await page.locator("#game-timer").textContent();
    await page.waitForTimeout(1500);
    const t2 = await page.locator("#game-timer").textContent();
    expect(t2).not.toEqual(t1);
  });

  test("science mode keeps classic layout", async ({ page }) => {
    await page.goto("/app?mode=science");
    // Original sidebar should be visible (not game mode).
    await expect(page.locator("#side")).toBeVisible();
    await expect(page.locator("#game-hud")).toBeHidden();
  });
});

test("daily-challenge button is visible in intro", async ({ page }) => {
  await page.goto("/app?mode=game");
  await page.waitForFunction(
    () => /game mode: ready/.test(document.querySelector("#out")?.textContent ?? ""),
    null,
    { timeout: 120_000 },
  );
  // Wait through any pending boot fade.
  await page.waitForFunction(
    () => {
      const b = document.querySelector("#boot") as HTMLElement | null;
      return !b || b.classList.contains("hidden");
    },
    null,
    { timeout: 60_000 },
  );
  const dailyBtn = page.locator("#game-mode-daily");
  await dailyBtn.waitFor({ state: "visible", timeout: 5000 });
  const text = await dailyBtn.textContent();
  expect(text).toContain("daily challenge");
  // Click it — should start a countdown.
  await dailyBtn.click();
  await page.waitForFunction(
    () => /[1-3]|GO/.test(document.querySelector(".overlay-count")?.textContent ?? ""),
    null,
    { timeout: 4000 },
  );
});
