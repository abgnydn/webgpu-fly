// game-replay.spec.ts — verify the replay URL captures keystrokes and
// reproducibly replays them with the same target seed.

import { test, expect } from "./fixtures";

async function waitGameReady(page: import("@playwright/test").Page) {
  await page.waitForFunction(
    () => /game mode: ready/.test(document.querySelector("#out")?.textContent ?? ""),
    null,
    { timeout: 120_000 },
  );
}

async function startRound(page: import("@playwright/test").Page) {
  await page.keyboard.press("Space");
  await page.waitForFunction(
    () => {
      const o = document.querySelector("#game-overlay") as HTMLElement | null;
      return o && o.style.display === "none";
    },
    null,
    { timeout: 6000 },
  );
}

test("replay URL roundtrips events + target seed", async ({ page }) => {
  test.setTimeout(180_000);
  await page.goto("/?mode=game");
  await waitGameReady(page);
  await startRound(page);

  // Seeded target position — the byte-exact, GPU-free determinism check.
  // Captured before the forced win below, which moves the target.
  const t0 = await page.evaluate(() => {
    const g = (window as unknown as {
      __game?: { ctx: { room: { targetPos: [number, number, number] } } };
    }).__game!;
    return [g.ctx.room.targetPos[0], g.ctx.room.targetPos[1]];
  });

  // Press Q for 1.2s, then E for 0.8s.
  await page.keyboard.down("q");
  await page.waitForTimeout(1200);
  await page.keyboard.up("q");
  await page.waitForTimeout(50);
  await page.keyboard.down("e");
  await page.waitForTimeout(800);
  await page.keyboard.up("e");
  await page.waitForTimeout(200);

  // Force a win: move target onto the fly's current xy.
  await page.evaluate(() => {
    const phys = (window as unknown as {
      __physicsForTest?: { data: { qpos: Float64Array } };
    }).__physicsForTest;
    const game = (window as unknown as {
      __game?: { ctx: { room: { setTargetPos: (x: number, y: number) => void } } };
    }).__game;
    if (phys && game) {
      game.ctx.room.setTargetPos(phys.data.qpos[0], phys.data.qpos[1]);
    }
  });
  await page.waitForFunction(
    () => {
      const w = document.querySelector("#game-win") as HTMLElement | null;
      return w && w.style.display === "flex";
    },
    null,
    { timeout: 10_000 },
  );

  // Pull the replay URL out via the share button's clipboard write —
  // we patch navigator.clipboard.writeText to expose it.
  const replayUrl = await page.evaluate(() => {
    return new Promise<string>((resolve) => {
      const realWrite = navigator.clipboard.writeText.bind(navigator.clipboard);
      navigator.clipboard.writeText = (text: string) => {
        navigator.clipboard.writeText = realWrite;
        resolve(text);
        return Promise.resolve();
      };
      const btn = document.getElementById("game-share-btn") as HTMLButtonElement;
      btn.click();
    });
  });

  // Sanity check the URL shape.
  expect(replayUrl).toContain("#r=");
  expect(replayUrl).toContain("mode=game");
  const m = replayUrl.match(/#r=([A-Za-z0-9_-]+)/);
  expect(m).toBeTruthy();
  // 2 down + 2 up events = 4 records × 4 bytes = 16 + 6 (dn fingerprint
  // + seed) = 22 bytes, base64 ≈ 30 chars. Tolerant assertion.
  expect(m![1].length).toBeGreaterThan(20);

  // Open the URL in a fresh context (same persistent profile) and verify
  // it enters replay mode with the right event count.
  const page2 = await page.context().newPage();
  await page2.goto(replayUrl);
  await waitGameReady(page2);

  // The replay overlay should appear with the keystroke count in its body.
  await page2.waitForFunction(
    () => {
      const o = document.querySelector("#game-overlay") as HTMLElement | null;
      const t = document.querySelector(".overlay-title")?.textContent ?? "";
      return o && o.style.display === "flex" && /replay/i.test(t);
    },
    null,
    { timeout: 30_000 },
  );

  const bodyText = await page2.locator(".overlay-body").textContent();
  // 4 events: q-down, q-up, e-down, e-up. Both keys are released before
  // the forced win, so the win adds no synthetic releases.
  expect(bodyText).toContain("4 keystrokes");

  // Same seed → same LCG → bit-identical target, on a different page.
  const t1 = await page2.evaluate(() => {
    const g = (window as unknown as {
      __game?: { ctx: { room: { targetPos: [number, number, number] } } };
    }).__game!;
    return [g.ctx.room.targetPos[0], g.ctx.room.targetPos[1]];
  });
  expect(t1[0]).toBeCloseTo(t0[0], 9);
  expect(t1[1]).toBeCloseTo(t0[1], 9);
  await page2.close();
});
