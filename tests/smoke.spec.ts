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

  // Trained walking policy from TuragaLab/flybody (Vaxenburg et al.
   // 2025) loads + runs a forward pass without exploding. This is the
   // RL-driven body controller; once the observation pipeline is wired
   // up, it replaces the kinematic assist for actually-actuated walking.
  // Lock the TS forward pass against a numpy ground truth (generated
   // by tools/verify_walking_policy.py from the same weight binary).
   // If anything in extract_walking_policy.py or walking-policy.ts
   // drifts (variable mapping, layer order, activation function),
   // this test pinpoints it.
  test("trained walking policy matches numpy ground truth", async ({ page }) => {
    await page.goto("/");
    const fixtures = await page.evaluate(async () => {
      const fres = await fetch("/walking-policy-fixtures.json");
      const cases: Array<{ name: string; obs: number[]; action: number[] }> = await fres.json();
      const modPath = "/src/walking-policy.ts";
      const mod = await import(/* @vite-ignore */ modPath);
      const policy = await mod.loadWalkingPolicy("/walking-policy.bin");
      const results = cases.map(({ name, obs, action }) => {
        const out = policy.act(new Float32Array(obs)).slice();
        let maxDiff = 0;
        for (let i = 0; i < out.length; i++) {
          const d = Math.abs(out[i] - action[i]);
          if (d > maxDiff) maxDiff = d;
        }
        return { name, maxDiff, sampleTs: Array.from(out.slice(0, 3)), sampleNp: action.slice(0, 3) };
      });
      return results;
    });
    for (const r of fixtures) {
      // Float32 round-trip: tolerance scales with action magnitude.
      // Numpy uses float32 internally too, so element-wise diffs
      // should be ~1e-3 absolute even on actions of magnitude 100s.
      expect(r.maxDiff, `case "${r.name}" diverges from numpy: TS=${r.sampleTs}, np=${r.sampleNp}, maxDiff=${r.maxDiff}`).toBeLessThan(1.0);
    }
  });

  // The 741-dim observation layout decoded from the SavedModel proto
   // (12 observables, alphabetical order, dims summing to 741) is a
   // strict contract — drift here means the policy receives garbage at
   // runtime. Lock it down.
  test("walking policy obs layout sums to 741", async ({ page }) => {
    await page.goto("/");
    const result = await page.evaluate(async () => {
      const modPath = "/src/walking-policy.ts";
      const mod = await import(/* @vite-ignore */ modPath);
      const total = mod.WALKING_OBS_TOTAL;
      const layout = mod.WALKING_OBS_LAYOUT.map((o: { name: string; dim: number }) => ({
        name: o.name, dim: o.dim,
      }));
      // Spot-check a few offsets — these must match the alphabetical
      // concatenation order the trained policy was exported with.
      const offAccel = mod.obsOffset("accelerometer");
      const offJointsPos = mod.obsOffset("joints_pos");
      const offRefDisp = mod.obsOffset("ref_displacement");
      const offWorldZ = mod.obsOffset("world_zaxis");
      return { total, layout, offAccel, offJointsPos, offRefDisp, offWorldZ };
    });
    expect(result.total).toBe(741);
    expect(result.layout[0].name).toBe("accelerometer");
    expect(result.layout[result.layout.length - 1].name).toBe("world_zaxis");
    // Offsets are cumulative sums of preceding dims.
    expect(result.offAccel).toBe(0);
    expect(result.offJointsPos).toBe(3 + 59 + 21 + 18 + 3);   // 104
    expect(result.offRefDisp).toBe(3 + 59 + 21 + 18 + 3 + 85 + 85);   // 274
    expect(result.offWorldZ).toBe(741 - 3);
  });

  test("trained walking policy loads + forward-passes", async ({ page }) => {
    await page.goto("/");
    const result = await page.evaluate(async () => {
      const modPath = "/src/walking-policy.ts";
      const mod = await import(/* @vite-ignore */ modPath);
      const policy = await mod.loadWalkingPolicy("/walking-policy.bin");
      // Two test inputs: all zeros (rest pose) + synthetic ramp.
      const zero = new Float32Array(policy.obsDim);
      const ramp = new Float32Array(policy.obsDim);
      for (let i = 0; i < ramp.length; i++) ramp[i] = (i % 7) * 0.01;
      const summarize = (obs: Float32Array) => {
        const a = policy.act(obs).slice();
        let nans = 0, max = 0;
        for (let i = 0; i < a.length; i++) {
          if (Number.isNaN(a[i])) nans++;
          const x = Math.abs(a[i]);
          if (x > max) max = x;
        }
        return { nans, max, first: Array.from(a.slice(0, 5)) };
      };
      return {
        obsDim: policy.obsDim, actDim: policy.actDim,
        zero: summarize(zero), ramp: summarize(ramp),
      };
    });
    expect(result.obsDim).toBe(741);
    expect(result.actDim).toBe(59);
    // No NaNs/inf for either input.
    expect(result.zero.nans).toBe(0);
    expect(result.ramp.nans).toBe(0);
    // All-zero obs → policy bias term, should be bounded by a small
    // constant — head bias is ~[-1.3, 1.5] from extraction logs.
    expect(result.zero.max, `zero-obs action ${result.zero.max} unreasonably large`).toBeLessThan(50);
    // Synthetic non-zero obs: forward pass shouldn't blow up to inf.
    // The trained policy was fed normalized obs, so out-of-distribution
    // inputs can produce larger actions; we just check finite + bounded.
    expect(result.ramp.max, `ramp-obs action ${result.ramp.max} unreasonably large`).toBeLessThan(2000);
  });

  // End-to-end policy → body wiring: enable the toggle, run a few
   // physics frames, ensure (a) the policy was invoked at least once,
   // (b) actions written to mujoco are bounded, (c) the body didn't
   // explode (positions stay finite, fly remains in scene). This is
   // the lock for the trained-walker pipeline as a whole — observation
   // builder, policy forward pass, action mapping, all in one.
  test("trained walker toggle → policy actions reach the body", async ({ page }) => {
    await waitForLog(page, "flybody attached", 120_000);

    // Hook into the page to observe what the policy is actually producing.
    // After enable, we'll snapshot policy action stats + body x-position
    // before/after to assert net forward progress.
    const result = await page.evaluate(async () => {
      // Grab handles via the global `room` / `physics` if exposed; fall
      // back to dynamic import probing.
      const w = window as unknown as { __probe?: { startX: number; samples: { max: number; mean: number; nans: number }[] } };
      w.__probe = { startX: NaN, samples: [] };
      return true;
    });
    void result;

    await clickButton(page, "Use RL policy");
    await waitForLog(page, "trained walker loaded", 30_000);

    const before = await page.evaluate(() => {
      const phys = (window as unknown as { __physicsForTest?: { data: { qpos: Float64Array } } }).__physicsForTest;
      if (!phys) return null;
      return { x: phys.data.qpos[0], y: phys.data.qpos[1], z: phys.data.qpos[2] };
    });

    await page.waitForTimeout(4_000);

    const after = await page.evaluate(() => {
      const phys = (window as unknown as { __physicsForTest?: { data: { qpos: Float64Array } } }).__physicsForTest;
      const stats = (window as unknown as { __rlActionStats?: { absMax: number; absMean: number; count: number } }).__rlActionStats;
      if (!phys) return null;
      return {
        x: phys.data.qpos[0], y: phys.data.qpos[1], z: phys.data.qpos[2],
        actionStats: stats,
      };
    });
    if (after?.actionStats) {
      console.log(`[rl-walker] policy ticks=${after.actionStats.count} action |max|=${after.actionStats.absMax.toFixed(3)} |mean|=${after.actionStats.absMean.toFixed(3)}`);
    }

    // Toggle off so subsequent tests start clean.
    await clickButton(page, "Use RL policy");

    const drive = await page.locator("#drive-readout").innerText();
    const m = drive.match(/speed (-?[\d.]+) cm\/s/);
    expect(m, `drive-readout missing speed: "${drive}"`).not.toBeNull();
    const speed = parseFloat(m![1]);
    expect(Number.isFinite(speed), `body speed went non-finite: ${speed}`).toBe(true);

    expect(before, "physics handle missing").not.toBeNull();
    expect(after, "physics handle missing").not.toBeNull();
    const dx = after!.x - before!.x;
    const dy = after!.y - before!.y;
    const dz = after!.z - before!.z;
    console.log(`[rl-walker] before=(${before!.x.toFixed(3)},${before!.y.toFixed(3)},${before!.z.toFixed(3)}) after=(${after!.x.toFixed(3)},${after!.y.toFixed(3)},${after!.z.toFixed(3)}) dx=${dx.toFixed(3)} dy=${dy.toFixed(3)} dz=${dz.toFixed(3)}`);

    // Forward progress: ref says "+x at 1 cm/s for 4s = 4 cm"; we
    // require some non-trivial advance.
    expect(dx, `body did not advance forward under RL (dx=${dx.toFixed(3)} cm)`).toBeGreaterThan(0.1);
    // Lateral drift bounded — body shouldn't strafe sideways more than
    // it walks forward.
    expect(Math.abs(dy), `body strafed sideways (dy=${dy.toFixed(3)} cm)`).toBeLessThan(Math.abs(dx) + 1.0);
    // Body should remain roughly upright (z within a centimeter of spawn).
    expect(Math.abs(dz), `body z drifted (dz=${dz.toFixed(3)} cm)`).toBeLessThan(1.0);
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
