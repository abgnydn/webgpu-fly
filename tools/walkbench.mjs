// Measuring stick for the trained walker. Enables the policy with the
// kinematic assist OFF and samples time-resolved, because the interesting
// quantities are all "while the fly is still upright" — a running max taken
// over the whole window is dominated by the post-capsize blow-up and hides
// whether the policy was behaving before it fell.
//
// Reports per rep:
//   tCapsize   simulated seconds from policy takeover until uprightness <= 0
//              (null = never capsized, which is the goal)
//   actMaxUp   |action|max sampled only while upright  (native band ~6)
//   cmPerSimS  forward progress while upright
//   uprightEnd uprightness at the end of the window (+1 upright, -1 on back)
//
// The numbers quoted for the trained walker in LIMITATIONS.md come from this
// script, so they are reproducible rather than asserted.
//
// Needs `npm run dev` on :8766 and the connectome/body binaries in public/
// (see README Quickstart) — same prerequisites as the Playwright suite.
//
// Usage: node tools/walkbench.mjs [reps]   (default 3)
import { chromium } from "@playwright/test";

const REPS = Number(process.argv[2] ?? 3);
const URL = "http://127.0.0.1:8766/app?mode=science";
const SIM_WINDOW_S = 2.5;

const browser = await chromium.launch({
  headless: false,
  args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan,WebGPU", "--disable-dawn-features=disallow_unsafe_apis"],
});

const rows = [];
for (let rep = 0; rep < REPS; rep++) {
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 720 } });
  const page = await ctx.newPage();
  await page.goto(URL, { waitUntil: "domcontentloaded" });
  await page.waitForFunction(
    () => /flybody attached/.test(document.querySelector("#out")?.textContent ?? ""),
    null, { timeout: 240_000 },
  );

  const assistOff = await page.evaluate(() => {
    const p = window.__physicsForTest;
    if (!p) return false;
    p.constructor.kinematicAssistEnabled = false;
    return p.constructor.kinematicAssistEnabled === false;
  });

  await page.locator('.stim-btn:has(.label:has-text("Use RL policy"))').first().click();
  await page.waitForFunction(
    () => /trained walker loaded/.test(document.querySelector("#out")?.textContent ?? ""),
    null, { timeout: 60_000 },
  ).catch(() => {});

  const sample = () => page.evaluate(() => {
    const p = window.__physicsForTest;
    const q = p.data.qpos;
    return {
      x: q[0], y: q[1], z: q[2], t: p.data.time,
      upright: 1 - 2 * (q[4] * q[4] + q[5] * q[5]),
      actMax: window.__rlActionStats?.absMax ?? null,
      ticks: window.__rlActionStats?.count ?? 0,
    };
  });

  // Wait for the policy to actually take the loop before t0.
  await page.waitForFunction(() => (window.__rlActionStats?.count ?? 0) > 5, null, { timeout: 60_000 }).catch(() => {});
  const s0 = await sample();

  let tCapsize = null, lastUp = s0, prevActMax = s0.actMax ?? 0, actMaxUp = 0, capsizeSample = null;
  for (;;) {
    const s = await sample();
    // __rlActionStats.absMax is a running max; its per-interval delta tells
    // us what the policy emitted during THIS interval.
    if (s.actMax !== null && s.actMax > prevActMax) {
      if (s.upright > 0) actMaxUp = Math.max(actMaxUp, s.actMax);
      prevActMax = s.actMax;
    }
    if (tCapsize === null && s.upright <= 0) {
      tCapsize = +(s.t - s0.t).toFixed(3);
      capsizeSample = s;
    }
    if (s.upright > 0) lastUp = s;
    if (s.t - s0.t >= SIM_WINDOW_S) { lastUp = s.upright > 0 ? s : lastUp; break; }
    await page.waitForTimeout(60);
  }
  const sEnd = await sample();

  const upDt = lastUp.t - s0.t;
  rows.push({
    rep,
    assistOff,
    tCapsize,
    actMaxUp: +actMaxUp.toFixed(2),
    cmPerSimS: upDt > 0.05 ? +(Math.hypot(lastUp.x - s0.x, lastUp.y - s0.y) / upDt).toFixed(3) : null,
    dxUpright: +(lastUp.x - s0.x).toFixed(3),
    uprightStart: +s0.upright.toFixed(3),
    uprightEnd: +sEnd.upright.toFixed(3),
    ticks: sEnd.ticks,
    capsizeAt: capsizeSample ? { x: +capsizeSample.x.toFixed(2), z: +capsizeSample.z.toFixed(3) } : null,
  });
  await ctx.close();
}

console.log(JSON.stringify(rows, null, 1));
const survived = rows.filter((r) => r.tCapsize === null).length;
const mean = (k) => { const v = rows.map((r) => r[k]).filter((x) => x !== null); return v.length ? +(v.reduce((a, b) => a + b, 0) / v.length).toFixed(3) : null; };
console.log(`\nSURVIVED (never capsized): ${survived}/${rows.length}`);
console.log(`mean tCapsize = ${mean("tCapsize")} sim s   mean actMax-while-upright = ${mean("actMaxUp")}   mean cm/sim s upright = ${mean("cmPerSimS")}`);
console.log(`mean uprightEnd = ${mean("uprightEnd")}   assist off all reps: ${rows.every((r) => r.assistOff)}`);
await browser.close();
