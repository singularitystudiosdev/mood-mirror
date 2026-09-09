// The smile-optimizer brain: a tiny policy-gradient network (pure, no DOM).
//
// Architecture: state -> tanh hidden layer -> three softmax heads that sample
// (meme, sound, rate). Updated online with REINFORCE plus a running reward
// baseline (Williams 1992) — the same family the workspace's Thompson bandits
// sit in, generalized to a neural policy. Until every meme has been seen once
// it runs a forced round-robin seed phase (same doctrine as
// ChromaTube pipeline/bandit.py: no exploitation before every arm has data).

// mulberry32 — small seeded PRNG so simulations are deterministic.
export function makeRng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Cadence ladder, front-loaded for spam (user amendment): the two fastest
// buckets sit under the 8s spam floor, so the rate head feels the -0.15 signal.
export const RATE_GAPS_MS = [3000, 5000, 10000, 20000, 40000];

// Defaults sized for 100 memes; every value overridable for tests.
export const BRAIN_DEFAULTS = {
  nMemes: 100,
  nSounds: 20,
  nRates: RATE_GAPS_MS.length,
  stateDim: 4,
  hidden: 16,
  lr: 0.08, // sized to learn within one sitting, not one week (user amendment)
  temperature: 0.9, // slightly sharper exploitation than pure softmax
  explore: 0.12, // ε-greedy floor per head, decays once the seed phase is done
  postSeedExplore: 0.04,
  recencyK: 5, // memes this recent are masked out of the policy softmax
  baselineAlpha: 0.08, // running-mean reward baseline, tracked fast
};

export function zeros(n) {
  return new Float32Array(n);
}

// Gaussian-ish init: uniform(-k, k) with k scaled by fan-in.
export function initMatrix(rows, cols, rng) {
  const k = 1 / Math.sqrt(cols);
  const m = new Float32Array(rows * cols);
  for (let i = 0; i < m.length; i++) m[i] = (rng() * 2 - 1) * k;
  return m;
}

function softmax(logits, temperature) {
  let max = -Infinity;
  for (const l of logits) if (l > max) max = l;
  const scaled = logits.map((l) => Math.exp((l - max) / temperature));
  const sum = scaled.reduce((a, b) => a + b, 0);
  return scaled.map((s) => s / sum);
}

function sampleFrom(probs, rng) {
  let r = rng();
  for (let i = 0; i < probs.length; i++) {
    r -= probs[i];
    if (r <= 0) return i;
  }
  return probs.length - 1;
}

export function shapeReward({ preSmile, peakSmile, hadFace, gapMs, minGapMs = 30000 }) {
  if (!hadFace) return -0.2; // nobody watching — the only mildly negative arm
  const lift = Math.max(0, peakSmile - preSmile);
  const base = Math.min(1, lift / 0.35);
  // rapid-fire with no smile reads as spam (the two fastest rate buckets);
  // specialist-measured: nudges the rate head toward the calm cadences
  if (gapMs < minGapMs && base < 0.05) return -0.15;
  return base;
}

export class SmilePolicy {
  constructor(options = {}) {
    const cfg = { ...BRAIN_DEFAULTS, ...options };
    this.cfg = cfg;
    this.rng = options.rng || makeRng(1);
    const { stateDim, hidden, nMemes, nSounds, nRates } = cfg;
    this.W1 = initMatrix(hidden, stateDim, this.rng);
    this.b1 = zeros(hidden);
    this.Wm = initMatrix(nMemes, hidden, this.rng);
    this.Ws = initMatrix(nSounds, hidden, this.rng);
    this.Wr = initMatrix(nRates, hidden, this.rng);
    this.bs = zeros(nSounds);
    this.br = zeros(nRates);
    this.baseline = 0;
    this.trials = 0;
    // per-meme Beta counts for the leaderboard UI and optimistic seeding
    this.memeAlpha = new Float32Array(nMemes).fill(1);
    this.memeBeta = new Float32Array(nMemes).fill(1);
    // seed phase: every meme once, shuffled, round-robining the sounds
    this.seedQueue = Array.from({ length: nMemes }, (_, i) => i);
    for (let i = this.seedQueue.length - 1; i > 0; i--) {
      const j = Math.floor(this.rng() * (i + 1));
      [this.seedQueue[i], this.seedQueue[j]] = [this.seedQueue[j], this.seedQueue[i]];
    }
    this.seedSoundCursor = 0;
    this.exploreDecayed = false;
    this.recent = []; // last recencyK memes, masked out of the policy softmax
    this.lastState = null;
    this.lastAction = null;
  }

  get seeding() {
    return this.seedQueue.length > 0;
  }

  // Policy forward pass. Returns the sampled combo; stashes what the update needs.
  act(state) {
    const { temperature, nMemes, nSounds, nRates } = this.cfg;
    if (this.seeding) {
      const meme = this.seedQueue[0];
      const sound = this.seedSoundCursor++ % nSounds;
      const action = { meme, sound, rate: 0, mode: "seed" };
      this.lastState = state;
      this.lastAction = action;
      return action;
    }
    if (!this.exploreDecayed) {
      this.exploreDecayed = true; // seed coverage is done; explore can relax
      this.cfg.explore = this.cfg.postSeedExplore;
    }
    const explore = this.cfg.explore;
    const hidden = this.#hiddenOf(state);
    const logitsM = this.#logits(this.Wm, hidden);
    for (const m of this.recent) logitsM[m] = -Infinity; // recency mask: repetition is the failure mode
    const pM = softmax(logitsM, temperature);
    const pS = softmax(this.#logits(this.Ws, hidden, this.bs), temperature);
    const pR = softmax(this.#logits(this.Wr, hidden, this.br), temperature);
    // independent per-head ε-greedy; explored heads are off-policy (no gradient)
    const explored = { meme: this.rng() < explore, sound: this.rng() < explore, rate: this.rng() < explore };
    const pick = (probs, n, ex) => (ex ? Math.floor(this.rng() * n) : sampleFrom(probs, this.rng));
    const meme = pick(pM, nMemes, explored.meme);
    this.recent.push(meme);
    if (this.recent.length > this.cfg.recencyK) this.recent.shift();
    const action = {
      meme,
      sound: pick(pS, nSounds, explored.sound),
      rate: pick(pR, nRates, explored.rate),
      probs: { meme: pM, sound: pS, rate: pR },
      explored,
      hidden,
      mode: explored.meme || explored.sound || explored.rate ? "explore" : "policy",
    };
    this.lastState = state;
    this.lastAction = action;
    return action;
  }

  // REINFORCE with baseline: Δθ = lr · (r − b) · ∇ log π(a|s). One combo = one
  // joint action across the three heads, so the advantage is shared. Explored
  // heads are off-policy and get no gradient. `hadFace=false` keeps the trial
  // out of the Beta leaderboard — the user may never have seen the meme.
  observe(reward, hadFace = true) {
    const { lr, baselineAlpha, nMemes, nSounds, nRates, hidden } = this.cfg;
    const action = this.lastAction;
    const state = this.lastState;
    if (!action || !state) return 0;
    const advantage = reward - this.baseline;
    this.baseline += baselineAlpha * (reward - this.baseline);
    if (hadFace) {
      this.memeAlpha[action.meme] += Math.max(0, reward);
      this.memeBeta[action.meme] += Math.max(0, 1 - Math.max(reward, 0));
    }
    this.trials++;
    if (action.mode === "seed") {
      const idx = this.seedQueue.indexOf(action.meme);
      if (idx !== -1) this.seedQueue.splice(idx, 1);
      this.lastAction = null;
      return advantage;
    }
    const a = action.hidden;
    const dA = zeros(hidden);
    const learnHead = (W, b, probs, index, n) => {
      const dLogits = new Float32Array(n);
      for (let k = 0; k < n; k++) dLogits[k] = (k === index ? 1 : 0) - probs[k];
      const scale = lr * advantage;
      for (let k = 0; k < n; k++) {
        for (let j = 0; j < hidden; j++) dA[j] += scale * dLogits[k] * W[k * hidden + j];
        if (b) b[k] += scale * dLogits[k]; // meme head is bias-free: per-meme bias measured no effect
        for (let j = 0; j < hidden; j++) W[k * hidden + j] += scale * dLogits[k] * a[j];
      }
    };
    if (!action.explored?.meme) learnHead(this.Wm, null, action.probs.meme, action.meme, nMemes);
    if (!action.explored?.sound) learnHead(this.Ws, this.bs, action.probs.sound, action.sound, nSounds);
    if (!action.explored?.rate) learnHead(this.Wr, this.br, action.probs.rate, action.rate, nRates);
    for (let j = 0; j < hidden; j++) {
      const dPre = dA[j] * (1 - a[j] * a[j]); // tanh'
      for (let d = 0; d < state.length; d++) this.W1[j * state.length + d] += dPre * state[d];
      this.b1[j] += dPre;
    }
    this.lastAction = null;
    return advantage;
  }

  // Expected reward per meme from the Beta counts — leaderboard + display only;
  // the policy itself learns through the network.
  memeScore(i) {
    return this.memeAlpha[i] / (this.memeAlpha[i] + this.memeBeta[i]);
  }

  topMemes(k = 5) {
    return Array.from({ length: this.cfg.nMemes }, (_, i) => i)
      .sort((x, y) => this.memeScore(y) - this.memeScore(x))
      .slice(0, k)
      .map((i) => ({ meme: i, score: this.memeScore(i), n: this.memeAlpha[i] + this.memeBeta[i] - 2 }));
  }

  serialize() {
    const pack = (m) => Array.from(m);
    return {
      v: 1,
      cfg: { ...this.cfg },
      W1: pack(this.W1), b1: pack(this.b1),
      Wm: pack(this.Wm), Ws: pack(this.Ws), Wr: pack(this.Wr),
      bs: pack(this.bs), br: pack(this.br),
      baseline: this.baseline,
      trials: this.trials,
      memeAlpha: pack(this.memeAlpha),
      memeBeta: pack(this.memeBeta),
      seedQueue: [...this.seedQueue],
      seedSoundCursor: this.seedSoundCursor,
      recent: [...this.recent],
    };
  }

  static load(data, rng) {
    const p = new SmilePolicy({ ...data.cfg, rng });
    const unpack = (dst, src) => dst.set(src);
    unpack(p.W1, data.W1); unpack(p.b1, data.b1);
    unpack(p.Wm, data.Wm); unpack(p.Ws, data.Ws); unpack(p.Wr, data.Wr);
    unpack(p.bs, data.bs); unpack(p.br, data.br);
    unpack(p.memeAlpha, data.memeAlpha); unpack(p.memeBeta, data.memeBeta);
    p.baseline = data.baseline;
    p.trials = data.trials;
    p.seedQueue = [...data.seedQueue];
    p.seedSoundCursor = data.seedSoundCursor;
    p.recent = [...(data.recent || [])];
    return p;
  }

  #hiddenOf(state) {
    const { hidden } = this.cfg;
    const a = new Float32Array(hidden);
    for (let j = 0; j < hidden; j++) {
      let sum = this.b1[j];
      for (let d = 0; d < state.length; d++) sum += this.W1[j * state.length + d] * state[d];
      a[j] = Math.tanh(sum);
    }
    return a;
  }

  #logits(W, a, b) {
    const { nMemes, nSounds, nRates, hidden } = this.cfg;
    const n = W.length / hidden;
    if (n !== nMemes && n !== nSounds && n !== nRates) throw new Error("bad head size");
    const out = new Float32Array(n);
    for (let k = 0; k < n; k++) {
      let sum = b ? b[k] : 0;
      for (let j = 0; j < hidden; j++) sum += W[k * hidden + j] * a[j];
      out[k] = sum;
    }
    return out;
  }
}
