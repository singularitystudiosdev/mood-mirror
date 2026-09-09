// Combo engine: the DOM-side half of the smile optimizer. Schedules meme+sfx
// combos through the brain, measures the smile lift each combo causes, and
// feeds that reward back. Pure scheduling/state — rendering is done by the
// callbacks the app wires in (onCombo / onStats / onReward).
import { SmilePolicy, RATE_GAPS_MS, makeRng, shapeReward } from "./brain.7f5cb2c1.js";


const STORAGE_KEY = "smile-brain-v1";
const SEED_GAP_MS = 3000; // introducing itself: a combo every 3 s (100 memes ≈ 5 min)
const REWARD_WINDOW_MS = 3000; // smile latency to a visual gag peaks at 1-3 s
const PRE_SMILE_MS = 1500; // a smile already rising at fire time is not the combo's credit
const LATE_PEAK_MS = 2000; // peaks after this are discounted: plausibly an unrelated burst
const LATE_PEAK_FACTOR = 0.6;
const SLOPE_BASE_MS = 800; // smile delta taken over ~1 s, timestamped (frame-rate independent)

export class ComboEngine {
  constructor({ brain, memes, sounds, onCombo, onStats, onReward, rng } = {}) {
    this.memes = memes || [];
    this.sounds = sounds || [];
    this.onCombo = onCombo || (() => {});
    this.onStats = onStats || (() => {});
    this.onReward = onReward || (() => {});
    this.rng = rng || makeRng(Date.now() >>> 0);
    this.brain = brain || new SmilePolicy({ rng: this.rng });
    this.lastComboEnd = 0;
    this.nextGapMs = SEED_GAP_MS;
    this.pending = null; // { firedAt, preSmile, peakSmile, peakAt, sawFace }
    this.prevFiredAt = 0; // for the actual elapsed gap the spam penalty needs
    this.samples = []; // ring of {t, smile} — pre-smile baseline + timestamped slope
    this.slope = 0.5;
    this.rewardHistory = [];
    this.muted = false;
    this.imageCache = new Map();
    this.audio = null;
  }

  static stored() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      return raw ? JSON.parse(raw) : null;
    } catch {
      return null;
    }
  }

  static restore(rng) {
    const data = ComboEngine.stored();
    if (!data?.brain) return null;
    try {
      const brain = SmilePolicy.load(data.brain, rng);
      // stale baseline after a 48h+ gap: halve it so early rewards aren't graded
      // against a mean the user's current mood may never reach
      if (data.savedAt && Date.now() - data.savedAt > 48 * 3600e3) brain.baseline /= 2;
      return { brain, history: data.history || [] };
    } catch {
      return null;
    }
  }

  persist() {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify({ brain: this.brain.serialize(), history: this.rewardHistory.slice(-50), savedAt: Date.now() }));
    } catch {
      // private mode / quota — learning continues in memory only
    }
  }

  setMuted(muted) {
    this.muted = muted;
    if (muted && this.audio) this.audio.pause();
  }

  get ready() {
    return this.memes.length > 0 && this.sounds.length > 0;
  }

  // Called every frame from the app loop with the live happy score.
  tick(now, smile, energy, faceFound) {
    if (!this.ready) return;
    this.samples.push({ t: now, smile });
    while (this.samples.length && now - this.samples[0].t > 2000) this.samples.shift();
    // timestamped slope over ~1 s: same face → same feature at any frame rate
    const base = this.samples.find((s) => now - s.t >= SLOPE_BASE_MS);
    if (base) this.slope = Math.max(0, Math.min(1, 0.5 + ((smile - base.smile) / ((now - base.t) / 1000)) * 1.0));
    if (this.pending) {
      if (faceFound) {
        if (smile > this.pending.peakSmile) {
          this.pending.peakSmile = smile;
          this.pending.peakAt = now;
        }
        this.pending.sawFace = true;
      }
      if (now - this.pending.firedAt >= REWARD_WINDOW_MS) this.finalize(now);
      return;
    }
    if (faceFound && now - this.lastComboEnd >= this.nextGapMs) this.fire(now, smile, energy, faceFound);
  }

  // Fire a combo. Rate head: index into RATE_GAPS_MS (seed phase pins 5 s).
  // Public so the e2e harness can force one without a face in frame.
  fire(now, smile, energy = 0.5, faceFound = false) {
    const sinceNorm = Math.min(1, (now - this.lastComboEnd) / 240000);
    const state = [sinceNorm, smile, this.slope, Math.min(1, energy)];
    if (state.length !== this.brain.cfg.stateDim) throw new Error(`state vector drifted: ${state.length} vs stateDim ${this.brain.cfg.stateDim}`);
    const action = this.brain.act(state);
    const meme = this.memes[action.meme % this.memes.length];
    const sound = this.sounds[action.sound % this.sounds.length];
    this.nextGapMs = this.brain.seeding ? SEED_GAP_MS : RATE_GAPS_MS[action.rate % RATE_GAPS_MS.length];
    // a smile already rising in the last 1.5 s belongs to the user, not the combo
    let preSmile = smile;
    for (const s of this.samples) if (now - s.t <= PRE_SMILE_MS && s.smile > preSmile) preSmile = s.smile;
    this.pending = {
      firedAt: now,
      prevFiredAt: this.prevFiredAt,
      preSmile,
      peakSmile: smile,
      peakAt: now,
      sawFace: faceFound === true,
      action, meme, sound,
    };
    this.prevFiredAt = now;
    this.onCombo({ ...this.pending, rateGapMs: this.nextGapMs });
    if (!this.muted) this.#play(sound.file);
    if (typeof Image !== "undefined") {
      let img = this.imageCache.get(meme.file);
      if (!img) {
        img = new Image();
        img.src = `assets/memes/${meme.file}`;
        this.imageCache.set(meme.file, img);
      }
    }
  }

  #play(file) {
    try {
      if (typeof Audio === "undefined") return; // node-side tests: no audio stack
      this.audio?.pause();
      this.audio = new Audio(`assets/sfx/${file}`);
      this.audio.volume = 0.6;
      this.audio.play().catch(() => {}); // autoplay policies — visual still lands
    } catch {
      // no audio support; the meme alone still fires
    }
  }

  finalize(now) {
    const { action, preSmile, peakSmile, peakAt, firedAt, prevFiredAt, sawFace } = this.pending;
    // a peak late in the window is plausibly an unrelated spontaneous burst (Li 2020:
    // facial feedback is bursty) — discount it so the combo doesn't collect the credit
    const effectivePeak = peakAt - firedAt > LATE_PEAK_MS ? peakSmile * LATE_PEAK_FACTOR : peakSmile;
    const reward = shapeReward({
      preSmile,
      peakSmile: effectivePeak,
      hadFace: sawFace,
      gapMs: firedAt - prevFiredAt, // the gap that PRECEDED this combo — the rate head's actual choice
      minGapMs: this.brain.seeding ? SEED_GAP_MS : undefined,
    });
    const advantage = this.brain.observe(reward, sawFace);
    this.rewardHistory.push(reward);
    this.lastComboEnd = firedAt; // the 5 s seed gap counts from the fire, not the reward window's end
    this.pending = null;
    this.onReward({ reward, advantage });
    this.onStats(this.stats());
    this.persist();
  }

  stats() {
    const history = this.rewardHistory;
    const mean = history.length ? history.reduce((a, b) => a + b, 0) / history.length : 0;
    const recent = history.slice(-20);
    return {
      seeding: this.brain.seeding,
      seedRemaining: this.brain.seedQueue.length,
      trials: this.brain.trials,
      meanReward: mean,
      recentMean: recent.length ? recent.reduce((a, b) => a + b, 0) / recent.length : 0,
      nextGapMs: this.nextGapMs,
      top: this.brain.topMemes(5),
      history: this.rewardHistory.slice(-50),
    };
  }
}
