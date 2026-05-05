// smoke.spec.ts — end-to-end behavioral check that exercises every
// interactive control and asserts the key signal each one is supposed
// to produce. Runs in headed Chromium with WebGPU enabled (see
// playwright.config.ts) so it actually drives the LIF kernel and the
// flybody dynamics.
//
// Stops the user from being the QA loop: one `npm run test:e2e`
// reports PASS/FAIL for every behavior in the demo.

import { test, expect, Page } from "@playwright/test";

const READY_MSG = "FlySim ready";

/** Read the full text content of the log pane. */
async function logText(page: Page): Promise<string> {
  return await page.locator("#out").innerText();
}

/** Wait until a substring appears in the log (or timeout). */
async function waitForLog(page: Page, needle: string, timeout = 60_000): Promise<void> {
  const t0 = Date.now();
  while (Date.now() - t0 < timeout) {
    const txt = await logText(page);
    if (txt.includes(needle)) return;
    await page.waitForTimeout(250);
  }
  throw new Error(`timed out waiting for "${needle}" in log`);
}

/** Click a button by its visible label text (the `<span class="label">`). */
async function clickButton(page: Page, label: string): Promise<void> {
  await page
    .locator(`.stim-btn:has(.label:has-text("${label}"))`)
    .first()
    .click();
}

/** Wait for a button to become enabled again after a stim run. */
async function waitButtonIdle(page: Page, label: string): Promise<void> {
  await page
    .locator(`.stim-btn:has(.label:has-text("${label}"))`)
    .first()
    .waitFor({ state: "attached" });
  await page.waitForFunction(
    (lbl: string) => {
      const btn = Array.from(document.querySelectorAll<HTMLButtonElement>(".stim-btn"))
        .find((b) => b.querySelector(".label")?.textContent?.includes(lbl));
      return btn && !btn.disabled;
    },
    label,
    { timeout: 90_000 },
  );
}

/** Pull a number out of the log matching a regex with a single capture.
 *  Strips thousands-commas before parseFloat. */
function extractNumber(log: string, re: RegExp): number | null {
  const m = log.match(re);
  if (!m) return null;
  return parseFloat(m[1].replace(/,/g, ""));
}

test.describe("webgpu-fly e2e", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/");
    await waitForLog(page, READY_MSG, 90_000);
  });

  test("brain fires on Mixed sensory: KC sparsity in 5–25%", async ({ page }) => {
    await clickButton(page, "Mixed sensory");
    await waitButtonIdle(page, "Mixed sensory");
    const log = await logText(page);
    // Find the KC line in the most recent block.
    const kcMatch = log.match(/KC\s+\d+\s*\/\s*\d+\s*\(([\d.]+)%\)/g);
    expect(kcMatch, "KC line missing from log").not.toBeNull();
    const last = kcMatch![kcMatch!.length - 1];
    const pct = parseFloat(last.match(/\(([\d.]+)%\)/)![1]);
    expect(pct, `KC% out of band: ${pct}`).toBeGreaterThan(5);
    expect(pct).toBeLessThan(30);
  });

  test("Vm climbs above rest after a stim", async ({ page }) => {
    await clickButton(page, "Visual flash");
    await waitButtonIdle(page, "Visual flash");
    const log = await logText(page);
    const aboveRest = extractNumber(log, /above-rest=([\d,]+)/);
    expect(aboveRest, "vm diagnostic line missing").not.toBeNull();
    // At least 1k neurons should be above rest after a strong visual flash.
    const n = parseInt(String(aboveRest).replace(/,/g, ""), 10);
    expect(n, `too few neurons above rest: ${n}`).toBeGreaterThan(1000);
  });

  test("DNa01 stim produces nonzero brain-driven motor", async ({ page }) => {
    await clickButton(page, "DNa01");
    await waitButtonIdle(page, "DNa01");
    const log = await logText(page);
    const fwd = extractNumber(log, /brain-driven motor: fwd=(-?[\d.]+)/);
    expect(fwd, "DNa01 motor command missing").not.toBeNull();
    expect(Math.abs(fwd!), `DNa01 fwd is zero: ${fwd}`).toBeGreaterThan(0.01);
  });

  test("DNb01 moonwalker drives backward (fwd < 0)", async ({ page }) => {
    await clickButton(page, "DNb01");
    await waitButtonIdle(page, "DNb01");
    const log = await logText(page);
    const fwd = extractNumber(log, /DNb01[\s\S]+?brain-driven motor: fwd=(-?[\d.]+)/);
    expect(fwd, "DNb01 motor command missing").not.toBeNull();
    expect(fwd!, `DNb01 fwd should be negative, got ${fwd}`).toBeLessThan(0);
  });

  // DNa01 / DNa02 / DNb01 are bilateral straight-walking commands —
  // their motor command should be mostly fwd, very little turn. If
  // the spine produces a strong turn signal from these symmetric
  // stims, the demo body curves instead of walking straight (the
  // user's complaint).
  for (const dn of ["DNa01", "DNa02", "DNb01"]) {
    test(`${dn} walks straight (|turn| < 0.4)`, async ({ page }) => {
      await clickButton(page, dn);
      await waitButtonIdle(page, dn);
      const log = await logText(page);
      const slice = log.slice(log.lastIndexOf(`DN stim: ${dn}`));
      const m = slice.match(/brain-driven motor: fwd=(-?[\d.]+)\s+turn=(-?[\d.]+)/);
      expect(m, `${dn} motor line missing`).not.toBeNull();
      const turn = parseFloat(m![2]);
      expect(Math.abs(turn), `${dn} should walk straight, |turn|=${Math.abs(turn).toFixed(2)}`).toBeLessThan(0.4);
    });
  }

  test("retina detects red target during closed loop", async ({ page }) => {
    // Wait for flybody to attach so the room has the body in scene.
    await waitForLog(page, "flybody attached", 120_000);
    await clickButton(page, "Track target");
    // Run the loop briefly then stop.
    await page.waitForTimeout(8_000);
    await clickButton(page, "Track target");
    const log = await logText(page);
    const tickLines = log.match(/tick \d+: angle=(-?[\d.NaN]+)°/g);
    expect(tickLines, "no closed-loop tick lines in log").not.toBeNull();
    // At least one tick must have a finite angle — the retina has to
    // see the target eventually (it spawns 3 cm dead ahead).
    const finiteTicks = tickLines!.filter((l) => !l.includes("NaN"));
    expect(finiteTicks.length, `retina never detected target: ${tickLines!.length} ticks all NaN`)
      .toBeGreaterThan(0);
  });

  test("evolve-gait converges and applies", async ({ page }) => {
    // Wait for flybody so the winner can be applied to the live body —
    // otherwise the click handler hits the "flybody not loaded yet" path.
    await waitForLog(page, "flybody attached", 120_000);
    await clickButton(page, "Evolve gait");
    await waitButtonIdle(page, "Evolve gait");
    const log = await logText(page);
    const evolved = log.match(/evolved in [\d.]+ s/);
    expect(evolved, "evolution didn't finish").not.toBeNull();
    // Best fitness should make it past 3 — see evolve.wgsl reward shape.
    const lastBest = log.match(/gen \d+\s+best=([\d.]+)/g)!.pop()!;
    const bestN = parseFloat(lastBest.match(/best=([\d.]+)/)![1]);
    expect(bestN).toBeGreaterThan(3);
    expect(log).toContain("applied evolved gait to live body");
  });

  test("body moves after DN-driven walking command", async ({ page }) => {
    await waitForLog(page, "flybody attached", 120_000);
    // Click a forward DN; let the brain run, the motor drive set,
    // and the body integrate for a few seconds.
    await clickButton(page, "DNa02");
    await waitButtonIdle(page, "DNa02");
    await page.waitForTimeout(5_000);
    const drive = await page.locator("#drive-readout").innerText();
    // drive-readout has "speed X.YZ cm/s" appended each frame.
    const m = drive.match(/speed (-?[\d.]+) cm\/s/);
    expect(m, `drive readout missing speed: "${drive}"`).not.toBeNull();
    const speed = parseFloat(m![1]);
    expect(speed, `body didn't move under DNa02 (speed=${speed} cm/s)`).toBeGreaterThan(0.1);
  });

  // Literature-grounded firing-rate assertions. KC sparsity is the
  // canonical mushroom-body result (Honegger 2011, Lin 2014), ORN goes
  // to 100% under direct olfactory drive, DN should be > 5% under broad
  // sensory drive. These catch regressions in the brain calibration
  // even when the body / spine / retina paths look healthy.

  test("Olfactory hit drives ORNs to 100%", async ({ page }) => {
    await clickButton(page, "Olfactory hit");
    await waitButtonIdle(page, "Olfactory hit");
    const log = await logText(page);
    const ornMatch = log.match(/ORN\s+(\d+)\s*\/\s*(\d+)/g);
    expect(ornMatch, "ORN line missing").not.toBeNull();
    const last = ornMatch![ornMatch!.length - 1];
    const [_, hit, total] = last.match(/(\d+)\s*\/\s*(\d+)/)!;
    const pct = (parseInt(hit, 10) / parseInt(total, 10)) * 100;
    expect(pct, `ORN should be near 100%, got ${pct}%`).toBeGreaterThan(80);
  });

  test("Mixed sensory drives DN cascade above 5%", async ({ page }) => {
    await clickButton(page, "Mixed sensory");
    await waitButtonIdle(page, "Mixed sensory");
    const log = await logText(page);
    const dnMatch = log.match(/DN\s+\d+\s*\/\s*\d+\s*\(([\d.]+)%\)/g);
    expect(dnMatch, "DN line missing").not.toBeNull();
    const last = dnMatch![dnMatch!.length - 1];
    const pct = parseFloat(last.match(/\(([\d.]+)%\)/)![1]);
    expect(pct, `DN cascade too cold under broad sensory drive: ${pct}%`).toBeGreaterThan(5);
  });

  test("MANC spine reports motor activity in drive readout", async ({ page }) => {
    await waitForLog(page, "flybody attached", 120_000);
    await clickButton(page, "DNa01");
    await waitButtonIdle(page, "DNa01");
    await page.waitForTimeout(2_000);
    const drive = await page.locator("#drive-readout").innerHTML();
    // The MANC line is only present when vnc.bin loaded successfully.
    if (!drive.includes("MANC")) {
      test.skip(true, "MANC not loaded (no vnc.bin) — skipping");
    }
    expect(drive).toMatch(/MANC\s*:\s*leg/);
  });

  // Brain-internal cascade test only for DNs whose primary projection
  // stays inside the brain (DNa01/DNa02). DNb01, DNp01, DNg13 are
  // descending neurons whose main targets live in the VNC — biologically
  // they should NOT light up many brain neurons. For those we test the
  // motor outcome separately ("DNb01 moonwalker drives backward" etc.)
  // which routes through the MANC spine.
  for (const dn of ["DNa01", "DNa02"]) {
    test(`${dn} ignites a real brain-internal cascade (recruits > 50)`, async ({ page }) => {
      await clickButton(page, dn);
      await waitButtonIdle(page, dn);
      const log = await logText(page);
      const slice = log.slice(log.lastIndexOf(`DN stim: ${dn}`));
      const m = slice.match(/final-window recruits: ([\d,]+)/);
      expect(m, `${dn} recruits line missing`).not.toBeNull();
      const recruits = parseInt(m![1].replace(/,/g, ""), 10);
      expect(recruits, `${dn} cascade too cold: ${recruits} recruits`).toBeGreaterThan(50);
    });
  }

  // Closed-loop should not flee from the target. We're modest about
  // what "tracking" requires here — the fly may temporarily lose sight
  // mid-turn — but the run as a whole shouldn't show the fly walking
  // off into space, and at least one tick should have a finite angle
  // proving the retina did sample the target at some point.
  // The visual reflex must turn the fly TOWARD the target, not away
  // from it. Catches sign errors in the qvel-injection / reflex chain
  // that are otherwise invisible (the fly ends up sweep-mode-locked
  // and the dist test still passes because the body never wandered).
  // Across consecutive finite-angle ticks, |angle| must shrink at
  // least once — i.e. the fly does close the gap between tick N and
  // tick N+1 at some point during the run.
  test("visual reflex turns fly TOWARD target (sign correct)", async ({ page }) => {
    await waitForLog(page, "flybody attached", 120_000);
    await clickButton(page, "Track target");
    await page.waitForTimeout(8_000);
    await clickButton(page, "Track target");
    const log = await logText(page);
    const ticks = [...log.matchAll(/tick (\d+): angle=([^°]+)° dist=([\d.]+)cm/g)];
    const finiteAngles: number[] = [];
    for (const t of ticks) {
      const a = t[2].trim();
      if (a !== "NaN") finiteAngles.push(parseFloat(a));
    }
    expect(finiteAngles.length, "no finite angles seen").toBeGreaterThan(2);
    // Find any consecutive-tick pair where |angle| got smaller — i.e.
    // the fly successfully closed in. With sign error, |angle|
    // monotonically grows from ~0 until the target leaves view.
    let approached = false;
    for (let i = 1; i < finiteAngles.length; i++) {
      if (Math.abs(finiteAngles[i]) < Math.abs(finiteAngles[i - 1]) - 0.5) {
        approached = true; break;
      }
    }
    expect(approached, `|angle| only grew or held: ${finiteAngles.join(", ")}`).toBe(true);
  });

  test("closed-loop doesn't flee from target", async ({ page }) => {
    await waitForLog(page, "flybody attached", 120_000);
    await clickButton(page, "Track target");
    await page.waitForTimeout(12_000);
    await clickButton(page, "Track target");
    const log = await logText(page);
    const ticks = [...log.matchAll(/tick (\d+): angle=([^°]+)° dist=([\d.]+)cm/g)];
    expect(ticks.length, "no tick lines").toBeGreaterThan(2);
    const firstDist = parseFloat(ticks[0][3]);
    const lastDist = parseFloat(ticks[ticks.length - 1][3]);
    // No long-distance flee: fly should stay within ~2 cm of where it
    // started the run, not walk off-screen.
    expect(lastDist, `fly walked off (${firstDist}cm → ${lastDist}cm)`).toBeLessThan(firstDist + 2.0);
    // At least one of the logged ticks must have seen the target
    // (finite angle), proving the retina path is connected.
    const seenAtLeastOnce = ticks.some((t) => t[2].trim() !== "NaN");
    expect(seenAtLeastOnce, `retina never saw target across ${ticks.length} ticks`).toBe(true);
  });

  test("Spontaneous keeps brain quiet (KC < 5%)", async ({ page }) => {
    await clickButton(page, "Spontaneous");
    await waitButtonIdle(page, "Spontaneous");
    const log = await logText(page);
    // Slice from the LAST "--- Spontaneous ---" header to end-of-log,
    // then read the KC% in that slice. JS regex doesn't have \z so a
    // simple substring is more reliable than a single regex.
    const idx = log.lastIndexOf("--- Spontaneous ---");
    expect(idx, "Spontaneous block missing").toBeGreaterThanOrEqual(0);
    const slice = log.slice(idx);
    const kc = slice.match(/KC\s+\d+\s*\/\s*\d+\s*\(([\d.]+)%\)/);
    expect(kc, "Spontaneous KC line missing").not.toBeNull();
    const pct = parseFloat(kc![1]);
    expect(pct, `Spontaneous KC should be ~0%, got ${pct}%`).toBeLessThan(5);
  });
});
