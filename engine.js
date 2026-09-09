// Pure mood engine: blendshape folding, FACS-grounded emotion scoring, and
// EMA + hysteresis mood tracking. No DOM — unit-tested in engine.test.mjs.

// Weights grounded in FACS action units / Ekman basic emotions (adapted from
// facemappr-blendshape's emotions.json, verified 2026-09-09; its weights-based
// "neutral" replaced by the low-energy rule in MoodTracker).
export const EMOTIONS = [
  { key: "happy", label: "Happy", emoji: "😄", color: "#ffd166",
    weights: { mouthSmile: 1.0, cheekSquint: 0.6, eyeSquint: 0.35 } }, // AU12+6
  { key: "sad", label: "Sad", emoji: "😢", color: "#74c0fc",
    weights: { mouthFrown: 1.0, browInnerUp: 0.7, mouthShrugLower: 0.5 } }, // AU15+1+17
  { key: "angry", label: "Angry", emoji: "😠", color: "#ff6b6b",
    weights: { browDown: 1.0, eyeSquint: 0.5, mouthPress: 0.5, noseSneer: 0.3 } }, // AU4+7+23+9
  { key: "surprised", label: "Surprised", emoji: "😮", color: "#b197fc",
    weights: { browInnerUp: 1.0, eyeWide: 0.9, jawOpen: 1.0, browOuterUp: 0.7 } }, // AU1+5+26+2
  { key: "fearful", label: "Fearful", emoji: "😨", color: "#e599f7",
    weights: { eyeWide: 1.0, mouthStretch: 0.8, browInnerUp: 0.7, browOuterUp: 0.5, browDown: 0.4 } }, // AU5+20+1+2+4
  { key: "disgusted", label: "Disgusted", emoji: "🤢", color: "#8ce99a",
    weights: { noseSneer: 1.0, mouthUpperUp: 0.7, mouthShrugUpper: 0.5, browDown: 0.4 } }, // AU9+10+17+4
];

export const NEUTRAL = { key: "neutral", label: "Neutral", emoji: "😐", color: "#9aa4b2" };
export const SEARCHING = { key: "searching", label: "Looking for a face…", emoji: "👀", color: "#5d6a7a" };

// Tuned at ~30 fps so a held expression lands well inside 1 s (review pass).
export const SMOOTH = {
  emaAlpha: 0.35, // score smoothing, settles in ~150 ms
  switchMargin: 0.1, // challenger must beat the incumbent by this much
  holdFrames: 3, // …for this many consecutive scored frames
  dwellMs: 400, // flicker lock after a switch
  neutralScore: 0.15, // neutral gate: top emotion under this …
  neutralEnergy: 0.12, // … AND facial energy under this …
  neutralFrames: 5, // … sustained this many frames
};

export const moodFor = (key) =>
  key === "neutral" ? NEUTRAL : key === "searching" ? SEARCHING : EMOTIONS.find((e) => e.key === key);

// Fold the 52 shapes into base names: left/right pairs averaged into one value.
export function foldShapes(categories) {
  const shapes = {};
  const pairs = {};
  for (const { categoryName, score } of categories) {
    const side = categoryName.match(/^(.*)(Left|Right)$/);
    if (side) (pairs[side[1]] ||= []).push(score);
    else shapes[categoryName] = score;
  }
  for (const [base, values] of Object.entries(pairs))
    shapes[base] = values.reduce((a, b) => a + b, 0) / values.length;
  return shapes;
}

// Weighted mean of an emotion's blendshape scores — a 0..1 "model-like" score.
export function scoreEmotion(emotion, shapes) {
  let sum = 0;
  let weightSum = 0;
  for (const [shape, weight] of Object.entries(emotion.weights)) {
    sum += (shapes[shape] || 0) * weight;
    weightSum += weight;
  }
  return sum / weightSum;
}

// Mean of the 10 strongest shapes, ignoring gaze direction and blinks.
export function faceEnergy(shapes) {
  const values = Object.entries(shapes)
    .filter(([name]) => !name.startsWith("eyeLook") && name !== "eyeBlink" && name !== "_neutral")
    .map(([, value]) => value)
    .sort((a, b) => b - a)
    .slice(0, 10);
  if (!values.length) return 0;
  return values.reduce((a, b) => a + b, 0) / values.length;
}

// Surprised vs fearful share raised brows and open eyes. AU20 (mouthStretch)
// and AU4 (browDown) mark fear; a clean jawOpen marks surprise. The boost (0.15)
// is sized to clear the 0.10 switch margin.
export function discriminate(shapes, scores) {
  const fear = EMOTIONS.findIndex((e) => e.key === "fearful");
  const surprise = EMOTIONS.findIndex((e) => e.key === "surprised");
  if ((shapes.mouthStretch || 0) > 0.3 || (shapes.browDown || 0) > 0.25) scores[fear] += 0.15;
  else if ((shapes.jawOpen || 0) > 0.35 && (shapes.mouthStretch || 0) < 0.2) scores[surprise] += 0.15;
  return scores;
}

export class MoodTracker {
  constructor(onMood) {
    this.reset();
    this.onMood = onMood;
  }

  reset() {
    this.ema = EMOTIONS.map(() => 0); // the bars render exactly these values
    this.current = "searching";
    this.candidate = null;
    this.streak = 0;
    this.neutralStreak = 0;
    this.lastSwitch = 0;
  }

  update(shapes, now) {
    const raw = discriminate(shapes, EMOTIONS.map((e) => scoreEmotion(e, shapes)));
    this.ema = this.ema.map((prev, i) => prev + SMOOTH.emaAlpha * (raw[i] - prev));
    this.#step(this.ema, faceEnergy(shapes), now);
  }

  noFace(now) {
    this.ema = this.ema.map((value) => value * (1 - SMOOTH.emaAlpha));
    if (this.current !== "searching") this.#set("searching", now);
  }

  #step(scores, energy, now) {
    let best = 0;
    scores.forEach((s, i) => { if (s > scores[best]) best = i; });

    if (scores[best] < SMOOTH.neutralScore && energy < SMOOTH.neutralEnergy) {
      if (++this.neutralStreak >= SMOOTH.neutralFrames) {
        this.#set("neutral", now);
        this.candidate = null;
        this.streak = 0;
        return;
      }
    } else {
      this.neutralStreak = 0;
    }

    if (now - this.lastSwitch < SMOOTH.dwellMs) return;

    const bestKey = EMOTIONS[best].key;
    if (bestKey === this.current) {
      this.candidate = null;
      this.streak = 0;
      return;
    }

    const incumbent = this.current === "neutral"
      ? SMOOTH.neutralScore
      : scores[EMOTIONS.findIndex((e) => e.key === this.current)] ?? 0;

    if (scores[best] - incumbent >= SMOOTH.switchMargin) {
      if (this.candidate === bestKey && ++this.streak >= SMOOTH.holdFrames) this.#set(bestKey, now);
      else if (this.candidate !== bestKey) {
        this.candidate = bestKey;
        this.streak = 1;
      }
    } else {
      this.candidate = null;
      this.streak = 0;
    }
  }

  #set(key, now) {
    this.current = key;
    this.lastSwitch = now;
    this.onMood(key);
  }
}
