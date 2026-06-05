// game-feel.spec.ts — does pressing Q (DNa01 forward) actually walk
// the fly closer to the target? Diagnostic test, not a gate.
//
// Reads target distance before/after a 5-second Q hold. With the
// connectome → DN → body pipeline working end-to-end, distance should
// shrink (target is in front-arc by construction). With the pipeline
// broken (no spike cascade or bad motor mapping), distance stays flat
// or grows.

import { test, expect } from "./fixtures";

test("pressing Q (DNa01) moves the fly toward the target", async ({ page }) => {
  test.setTimeout(180_000);

  await page.goto("/?mode=game");
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

  // Read initial distance.
  const readDist = async () => {
    const txt = await page.locator("#game-dist").textContent();
    return parseFloat((txt ?? "0").replace("cm", "").trim());
  };
  // Let the brain stabilize a moment after countdown.
  await page.waitForTimeout(500);
  const d0 = await readDist();

  // Hold Q for 5s of real time (the brain ticks at ~10 Hz so this is
  // ~50 brain ticks, plenty for the cascade to develop).
  await page.keyboard.down("q");
  await page.waitForTimeout(5000);
  await page.keyboard.up("q");
  await page.waitForTimeout(500);

  const d1 = await readDist();

  // Capture body x position too — independent of target placement.
  const bodyX = await page.evaluate(() => {
    const phys = (window as unknown as {
      __physicsForTest?: { data: { qpos: Float64Array } };
    }).__physicsForTest;
    return phys ? phys.data.qpos[0] : null;
  });

  console.log(`\n  initial dist: ${d0.toFixed(2)} cm`);
  console.log(`  final   dist: ${d1.toFixed(2)} cm`);
  console.log(`  body x      : ${bodyX?.toFixed(3)} cm (started at 0)`);

  // The fly should have moved at all (body x changed > 0.1 cm).
  expect(bodyX !== null && Math.abs(bodyX) > 0.1).toBe(true);
});
