// Unit tests for the smile-optimizer brain: seed coverage, gradient sanity,
// persistence roundtrip, and a small-scale convergence check.
import test from "node:test";
import assert from "node:assert/strict";
import { SmilePolicy, makeRng, shapeReward } from "./brain.7f5cb2c1.js";

test("seed phase shows every meme exactly once, then hands over to the policy", () => {
  const brain = new SmilePolicy({ nMemes: 10, nSounds: 3, rng: makeRng(3) });
  const seen = [];
  while (brain.seeding) {
    const a = brain.act([0, 0, 0, 0]);
    brain.observe(0.5);
    seen.push(a.meme);
    assert.equal(a.mode, "seed");
  }
  assert.deepEqual(seen.sort((a, b) => a - b), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
  const after = brain.act([0, 0, 0, 0]);
  assert.notEqual(after.mode, "seed");
});

test("observe lifts the log-probability of the action it was rewarded for", () => {
  const brain = new SmilePolicy({ nMemes: 5, nSounds: 2, rng: makeRng(11), lr: 0.5, explore: 0, postSeedExplore: 0 });
  while (brain.seeding) { brain.act([0, 0, 0, 0]); brain.observe(0); }
  const state = [0.5, 0.5, 0, 0.5];
  const a1 = brain.act(state);
  const soundBefore = a1.probs.sound[a1.sound];
  brain.observe(1); // strong reward for exactly this combo
  const a2 = brain.act(state);
  // the meme head masks the just-shown meme (recency), so assert on the sound head
  assert.ok(a2.probs.sound[a1.sound] > soundBefore, `rewarded action should gain prob: ${soundBefore} -> ${a2.probs.sound[a1.sound]}`);
  assert.equal(a2.probs.meme[a1.meme], 0, "just-shown meme is masked out by the recency gate");
});

test("serialize/load roundtrips weights and reproduces the same action", () => {
  const brain = new SmilePolicy({ nMemes: 8, nSounds: 4, rng: makeRng(21) });
  while (brain.seeding) { brain.act([0.3, 0.3, 0, 0.3]); brain.observe(0.7); }
  brain.act([0.3, 0.3, 0, 0.3]);
  brain.observe(0.9);
  const blob = JSON.parse(JSON.stringify(brain.serialize()));
  const restored = SmilePolicy.load(blob, makeRng(99));
  assert.deepEqual(Array.from(restored.W1), Array.from(brain.W1));
  assert.equal(restored.trials, brain.trials);
  assert.deepEqual(restored.topMemes(3), brain.topMemes(3));
});

test("shapeReward: no-face is negative, lift scales, spamming is punished", () => {
  const r = (o) => shapeReward({ preSmile: 0.2, ...o });
  assert.equal(r({ peakSmile: 0.9, hadFace: false, gapMs: 30000 }), -0.2);
  assert.equal(r({ peakSmile: 0.55, hadFace: true, gapMs: 30000 }), 1);
  assert.ok(Math.abs(r({ peakSmile: 0.25, hadFace: true, gapMs: 30000 }) - 0.05 / 0.35) < 1e-9);
  assert.ok(Math.abs(r({ peakSmile: 0.2, hadFace: true, gapMs: 30000 })) < 1e-9);
  assert.equal(r({ peakSmile: 0.21, hadFace: true, gapMs: 3000 }), -0.15); // fast bucket AND dead lift
  assert.equal(r({ peakSmile: 0.2, hadFace: true, gapMs: 5000 }), -0.15); // any sub-30s spam
  assert.ok(r({ peakSmile: 0.45, hadFace: true, gapMs: 30000 }) > 0.5);
});

test("no-face trials are kept out of the Beta leaderboard", () => {
  const brain = new SmilePolicy({ nMemes: 4, nSounds: 2, rng: makeRng(31), explore: 0, postSeedExplore: 0 });
  while (brain.seeding) { brain.act([0, 0, 0, 0]); brain.observe(0.5); }
  const a = brain.act([0, 0, 0, 0]);
  const alpha0 = brain.memeAlpha[a.meme];
  brain.observe(-0.2, false); // nobody home — the meme must not take the blame
  assert.equal(brain.memeAlpha[a.meme], alpha0);
  const b = brain.act([0, 0, 0, 0]); // observe is one-shot: a fresh action per update
  const alphaB = brain.memeAlpha[b.meme];
  brain.observe(1, true);
  assert.equal(brain.memeAlpha[b.meme], alphaB + 1);
});

test("converges on a 30-meme user: rewarded meme dominates the leaderboard", () => {
  const rng = makeRng(5);
  const p = Array.from({ length: 30 }, () => rng() * 0.8);
  p[17] = 0.95; // one clear winner
  const brain = new SmilePolicy({ nMemes: 30, nSounds: 2, rng: makeRng(2), lr: 0.05 });
  const draw = () => rng();
  for (let t = 0; t < 3000; t++) {
    const a = brain.act([0.5, 0.2, 0, 0.5]);
    brain.observe(draw() < p[a.meme] ? 1 : 0);
  }
  const top1 = brain.topMemes(1)[0];
  assert.equal(top1.meme, 17, `expected meme 17 on top, got ${top1.meme} (score ${top1.score.toFixed(2)})`);
});
