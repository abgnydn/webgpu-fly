// game-screenshot.spec.ts — diagnostic: capture three states of the
// game (intro, mid-round, win) so we can eyeball the layout.

import { test } from "./fixtures";

test("capture game screenshots", async ({ page }) => {
  test.setTimeout(180_000);

  await page.goto("/?mode=game");
  await page.waitForFunction(
    () => /game mode: ready/.test(document.querySelector("#out")?.textContent ?? ""),
    null,
    { timeout: 120_000 },
  );

  // Wait for boot overlay to fade.
  await page.waitForFunction(
    () => {
      const b = document.querySelector("#boot") as HTMLElement | null;
      return !b || b.classList.contains("hidden");
    },
    null,
    { timeout: 60_000 },
  );

  await page.screenshot({ path: "test-results/game-intro.png", fullPage: false });
  await page.keyboard.press("Space");
  await page.waitForFunction(
    () => {
      const o = document.querySelector("#game-overlay") as HTMLElement | null;
      return o && o.style.display === "none";
    },
    null,
    { timeout: 6000 },
  );
  await page.waitForTimeout(800);
  await page.screenshot({ path: "test-results/game-playing-pre.png" });

  await page.keyboard.down("q");
  await page.waitForTimeout(2000);
  await page.screenshot({ path: "test-results/game-playing-mid-q.png" });
  await page.keyboard.up("q");

  await page.waitForTimeout(500);
  // Force a win for screenshot purposes by moving target onto the fly.
  await page.evaluate(() => {
    const phys = (window as unknown as {
      __physicsForTest?: { data: { qpos: Float64Array } };
    }).__physicsForTest;
    const room = (window as unknown as { __game?: { ctx: { room: { setTargetPos: (x: number, y: number) => void } } } }).__game;
    if (phys && room) {
      const x = phys.data.qpos[0];
      const y = phys.data.qpos[1];
      room.ctx.room.setTargetPos(x, y);
    }
  });
  await page.waitForTimeout(1500);
  await page.screenshot({ path: "test-results/game-win.png" });
});
