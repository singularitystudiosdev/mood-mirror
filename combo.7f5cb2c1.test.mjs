// Unit tests for ComboEngine's reward attribution: pre-smile exclusion,
// late-peak discount, and the 48h baseline halving on restore. All timing is
// explicit (tick/fire/finalize take `now`), so these are deterministic.
import test from "node:test";
import assert from "node:assert/strict";
import { ComboEngine } from "./combo.7f5cb2c1.js";
import { makeRng, RATE_GAPS_MS } from "./brain.7f5cb2c1.js";

// ComboEngine persists through localStorage; node has none until we shim it.
const store = new Map();
globalThis.localStorage = {
  getItem: (k) => (store.has(k) ? store.get(k) : null),
  setItem: (k, v) => store.set(k, String(v)),
  removeItem: (k) => store.delete(k),
};

const MEMES = [{ file: "m01.jpg", name: "test meme" }];
const SOUNDS = [{ file: "s01.mp3", name: "boing" }];

function engine(opts = {}) {
  return new ComboEngine({
    memes: MEMES,
    sounds: SOUNDS,
    rng: makeRng(5),
    onCombo: () => {},
    onStats: () => {},
    onReward: () => {},
    ...opts,
  });
}

// drive a fresh engine through its seed pass (the brain seeds 100 memes
// regardless of deck size; each fire+finalize retires one)
function exhaustSeed(e) {
  let t = 10000;
  let guard = 0;
  while (e.brain.seeding && guard++ < 200) {
    e.fire(t, 0.2);
    e.finalize(t + 3000);
    t += 5000;
  }
  return t;
}

test("a smile already rising before the fire is not credited to the combo", () => {
  const e = engine();
  let t = exhaustSeed(e);
  t += 30000; // calm gap: the measured combo must not trip the spam rule
  // user's smile ramping up over the 1.5 s before the fire: 0.3 → 0.6
  // (faceFound=false: the rising smile alone must not trigger an auto-fire)
  for (let i = 0; i < 10; i++) e.tick(t + i * 150, 0.3 + i * 0.03, 0.5, false);
  t += 1500;
  e.fire(t, 0.6, 0.5, true);
  const pre = e.pending.preSmile;
  assert.equal(pre, 0.6, "preSmile takes the pre-fire max, not the fire frame");
  e.tick(t + 100, 0.6, 0.5, true); // combo adds nothing over the baseline
  e.finalize(t + 3000);
  assert.equal(e.rewardHistory.at(-1), 0, "no lift over the pre-smile baseline → zero reward");
});

test("a late peak is discounted as a plausibly unrelated burst", () => {
  const e = engine();
  let t = exhaustSeed(e);
  e.tick(t, 0.2, 0.5, true);
  e.fire(t + 100, 0.2, 0.5, true);
  e.tick(t + 2200, 0.9, 0.5, true); // huge spike, but 2.1 s after the fire
  e.finalize(t + 3200);
  const got = e.rewardHistory.at(-1);
  const expected = Math.min(1, (0.9 * 0.6 - 0.2) / 0.35); // 0.9 discounted to 0.54
  assert.ok(Math.abs(got - expected) < 1e-9, `late-peak discount: got ${got}, want ${expected}`);
});

test("a fresh peak inside the window is credited in full", () => {
  const e = engine();
  let t = exhaustSeed(e);
  e.tick(t, 0.2, 0.5, true);
  e.fire(t + 100, 0.2, 0.5, true);
  e.tick(t + 1000, 0.9, 0.5, true); // 0.9 s after fire — inside the window, credited
  e.finalize(t + 3200);
  assert.equal(e.rewardHistory.at(-1), 1);
});

test("restore halves the baseline when the save is older than 48h", () => {
  const e = engine();
  exhaustSeed(e);
  e.brain.baseline = 0.6;
  e.persist();
  const data = JSON.parse(localStorage.getItem("smile-brain-v1"));
  data.savedAt = Date.now() - 72 * 3600e3; // three days stale
  localStorage.setItem("smile-brain-v1", JSON.stringify(data));
  const restored = ComboEngine.restore(makeRng(9));
  assert.ok(Math.abs(restored.brain.baseline - 0.3) < 1e-9, `baseline halved, got ${restored.brain.baseline}`);
});

test("spam: a fast re-fire with no smile earns the penalty", () => {
  const e = engine();
  let t = exhaustSeed(e);
  e.tick(t, 0.2, 0.5, true);
  e.fire(t + 100, 0.2, 0.5, true);
  e.finalize(t + 3100);
  // re-fire 1 s later: actual preceding gap = 1s < 8s floor, dead lift → -0.15
  e.tick(t + 3200, 0.2, 0.5, true);
  e.fire(t + 4100, 0.2, 0.5, true);
  e.finalize(t + 7200);
  assert.equal(e.rewardHistory.at(-1), -0.15);
  assert.ok(RATE_GAPS_MS[0] < 8000, "fastest ladder bucket sits under the spam floor");
});
