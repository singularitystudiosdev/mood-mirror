// Unit tests for the pure mood engine — margin/hold/neutral/dwell/discriminator.
import test from "node:test";
import assert from "node:assert/strict";
import { EMOTIONS, SMOOTH, foldShapes, scoreEmotion, faceEnergy, discriminate, MoodTracker } from "./engine.js";

const happy = EMOTIONS.find((e) => e.key === "happy");

const HAPPY_FACE = {
  mouthSmileLeft: 1, mouthSmileRight: 1,
  cheekSquintLeft: 0.8, cheekSquintRight: 0.8,
  eyeSquintLeft: 0.4, eyeSquintRight: 0.4,
};
const SURPRISED_FACE = {
  browInnerUp: 1, eyeWideLeft: 1, eyeWideRight: 1, jawOpen: 1, browOuterUpLeft: 1, browOuterUpRight: 1,
};

// Feed `count` frames spaced 50 ms apart starting at `startAt`; returns last `now`.
function feed(tracker, shapes, count, startAt) {
  let now = startAt;
  for (let i = 0; i < count; i++) {
    now += 50;
    tracker.update(foldShapes(Object.entries(shapes).map(([categoryName, score]) => ({ categoryName, score }))), now);
  }
  return now;
}

test("foldShapes averages left/right pairs and keeps singles", () => {
  const shapes = foldShapes([
    { categoryName: "mouthSmileLeft", score: 1 },
    { categoryName: "mouthSmileRight", score: 0.5 },
    { categoryName: "jawOpen", score: 0.8 },
  ]);
  assert.equal(shapes.mouthSmile, 0.75);
  assert.equal(shapes.jawOpen, 0.8);
});

test("scoreEmotion is the weighted mean of its shapes", () => {
  assert.ok(Math.abs(scoreEmotion(happy, { mouthSmile: 1 }) - 1 / 1.95) < 1e-9);
  const folded = { mouthSmile: 1, cheekSquint: 0.8, eyeSquint: 0.4 };
  assert.ok(Math.abs(scoreEmotion(happy, folded) - (1 + 0.6 * 0.8 + 0.35 * 0.4) / 1.95) < 1e-9);
});

test("faceEnergy is the mean of the ten strongest non-gaze shapes", () => {
  // folded names as the live loop feeds them; gaze and blink excluded from the window
  const shapes = {
    mouthSmile: 1, jawOpen: 0.5, eyeLookUp: 1, eyeBlink: 1,
    browDown: 0, mouthPress: 0, noseSneer: 0, cheekPuff: 0,
    jawForward: 0, mouthPucker: 0, mouthDimple: 0, mouthRollLower: 0,
  };
  const energy = faceEnergy(shapes);
  assert.ok(Math.abs(energy - 0.15) < 1e-9); // (1 + 0.5) / 10-entry window
});

test("discriminate boosts fear on mouthStretch/browDown, surprise on clean jawOpen", () => {
  const base = () => EMOTIONS.map(() => 0);
  const fearIdx = EMOTIONS.findIndex((e) => e.key === "fearful");
  const surpriseIdx = EMOTIONS.findIndex((e) => e.key === "surprised");
  assert.equal(discriminate({ mouthStretch: 0.5 }, base())[fearIdx], 0.15);
  assert.equal(discriminate({ jawOpen: 0.5 }, base())[surpriseIdx], 0.15);
  const both = discriminate({ mouthStretch: 0.5, jawOpen: 0.5 }, base());
  assert.equal(both[fearIdx], 0.15); // AU20 present wins over the surprise branch
  assert.equal(both[surpriseIdx], 0);
});

test("a close challenger below the switch margin never takes over", () => {
  const tracker = new MoodTracker(() => {});
  const now = feed(tracker, HAPPY_FACE, 10, 1000);
  assert.equal(tracker.current, "happy");
  // happy ≈0.83 vs sad ≈0.78 simultaneously — a 0.05 gap, under the 0.10 margin
  const ambivalent = {
    ...HAPPY_FACE,
    mouthFrownLeft: 0.9, mouthFrownRight: 0.9, browInnerUp: 0.8, mouthShrugLower: 0.5,
  };
  feed(tracker, ambivalent, 10, now);
  assert.equal(tracker.current, "happy");
});

test("a strong challenger past the margin switches, but only after the dwell lock", () => {
  const tracker = new MoodTracker(() => {});
  const now = feed(tracker, HAPPY_FACE, 10, 1000);
  assert.equal(tracker.current, "happy");
  // still inside the 400 ms dwell: margin is met, switch is blocked
  feed(tracker, SURPRISED_FACE, 2, now);
  assert.equal(tracker.current, "happy");
  // past dwell: three consecutive margin frames earn the switch
  const after = feed(tracker, SURPRISED_FACE, 8, now + 500);
  assert.equal(tracker.current, "surprised");
  assert.ok(after > 0);
});

test("an empty face settles to neutral through the low-energy gate", () => {
  const tracker = new MoodTracker(() => {});
  feed(tracker, {}, SMOOTH.neutralFrames + 1, 1000);
  assert.equal(tracker.current, "neutral");
});

test("reset clears every piece of hysteresis state", () => {
  const tracker = new MoodTracker(() => {});
  feed(tracker, HAPPY_FACE, 10, 1000);
  tracker.reset();
  assert.equal(tracker.current, "searching");
  assert.ok(tracker.ema.every((v) => v === 0));
  assert.equal(tracker.streak, 0);
  assert.equal(tracker.candidate, null);
  assert.equal(tracker.lastSwitch, 0);
});

test("noFace decays the bars and returns to searching", () => {
  const tracker = new MoodTracker(() => {});
  feed(tracker, HAPPY_FACE, 10, 1000);
  tracker.noFace(99999);
  assert.equal(tracker.current, "searching");
  assert.ok(Math.max(...tracker.ema) < 0.83);
});
