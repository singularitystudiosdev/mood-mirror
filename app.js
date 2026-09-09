// Mood Mirror — camera → MediaPipe FaceLandmarker blendshapes → dominant
// emotion → giant emoji — now with a smile-optimizer: the happy score is the
// reward signal for a policy network (brain.7f5cb2c1.js) that learns which
// meme+sound combos, and at what cadence, actually make you smile.

import { EMOTIONS, SMOOTH, faceEnergy, foldShapes, MoodTracker, moodFor } from "./engine.js";
import { ComboEngine } from "./combo.7f5cb2c1.js";
import { makeRng } from "./brain.7f5cb2c1.js";

const CDN_BASE = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1";
const CDN_MODEL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
const MODEL_LOAD_TIMEOUT_MS = 30000;
const DETECT_INTERVAL_MS = 33; // ~30 fps of inference, rAF can run faster

// ---- runtime loading (vendored first, CDN fallback) ---------------------------

async function headOk(url) {
  try {
    const response = await fetch(url, { method: "HEAD" });
    return response.ok;
  } catch {
    return false;
  }
}

async function withTimeout(promise, ms, label) {
  let timer;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error(`${label} timed out after ${ms / 1000}s`)), ms);
  });
  try {
    return await Promise.race([promise, timeout]);
  } finally {
    clearTimeout(timer);
  }
}

let landmarkerPromise = null;
function loadLandmarker() {
  if (!landmarkerPromise) {
    const attempt = withTimeout(
      (async () => {
        const local = await headOk("./vendor/vision_bundle.mjs");
        const localModel = local && (await headOk("./vendor/face_landmarker.task"));
        const base = local ? "." : CDN_BASE;
        const model = localModel ? "./vendor/face_landmarker.task" : CDN_MODEL;

        const { FilesetResolver, FaceLandmarker } = await import(
          local ? "./vendor/vision_bundle.mjs" : `${CDN_BASE}/vision_bundle.mjs`
        );
        const fileset = await FilesetResolver.forVisionTasks(local ? "./vendor/wasm" : `${CDN_BASE}/wasm`);
        const landmarker = await FaceLandmarker.createFromOptions(fileset, {
          baseOptions: { modelAssetPath: model, delegate: "CPU" },
          runningMode: "VIDEO",
          numFaces: 1,
          outputFaceBlendshapes: true,
        });
        ui.status.textContent = `${localModel ? "local" : "cdn"} model · cpu · on-device`;
        return landmarker;
      })(),
      MODEL_LOAD_TIMEOUT_MS,
      "Face model load",
    );
    landmarkerPromise = attempt;
    // A rejected promise must not be cached, or Retry can never re-load.
    attempt.catch(() => { landmarkerPromise = null; });
  }
  return landmarkerPromise;
}

// ---- UI ------------------------------------------------------------------------

const ui = {};
for (const id of ["intro", "loading", "live", "error", "cam", "scores", "status",
  "loading-msg", "error-msg", "big-emoji", "mood-label", "emoji-stage",
  "trainer", "trainer-score", "trainer-gap", "trainer-spark", "trainer-top",
  "trainer-status", "trainer-mute", "combo-card", "combo-img", "combo-sound"]) {
  ui[id.replace(/-(\w)/g, (_, c) => c.toUpperCase())] = document.getElementById(id);
}

function showPanel(name) {
  for (const panel of ["intro", "loading", "live", "error"])
    ui[panel].classList.toggle("hidden", panel !== name);
}

// The bars are the same EMA the hysteresis reads, so a switch is visibly earned.
const barRows = EMOTIONS.map((emotion) => {
  const row = document.createElement("li");
  row.className = "score-row";
  row.innerHTML = `<span class="score-name">${emotion.label}</span>
    <div class="score-track"><div class="score-fill" style="--c:${emotion.color}"></div></div>`;
  ui.scores.appendChild(row);
  return { key: emotion.key, row, fill: row.querySelector(".score-fill") };
});

let lastBarPaint = 0;
function paintBars(scores, now) {
  if (now - lastBarPaint < 100) return;
  lastBarPaint = now;
  let best = 0;
  scores.forEach((s, i) => { if (s > scores[best]) best = i; });
  barRows.forEach(({ key, row, fill }) => {
    const index = EMOTIONS.findIndex((e) => e.key === key);
    const pct = Math.max(0, Math.min(1, scores[index])) * 100;
    fill.style.width = `${pct.toFixed(1)}%`;
    row.classList.toggle("lead", index === best && scores[best] > SMOOTH.neutralScore);
  });
}

const tracker = new MoodTracker((key) => {
  const mood = moodFor(key) || { label: "Neutral", emoji: "😐" };
  ui.bigEmoji.textContent = mood.emoji;
  ui.moodLabel.textContent = mood.label;
  ui.moodLabel.dataset.emotion = mood.key;
  ui.emojiStage.classList.toggle("searching", mood.key === "searching");
  document.documentElement.style.setProperty("--aura", `${mood.color}33`);
  barRows.forEach(({ row }) => row.classList.remove("just-lost"));
  const previousLead = barRows.find(({ row }) => row.classList.contains("lead"));
  // restart the pop so every switch is a squash-and-stretch arrival
  ui.bigEmoji.classList.remove("pop");
  void ui.bigEmoji.offsetWidth;
  ui.bigEmoji.classList.add("pop");
  if (previousLead) previousLead.row.classList.add("just-lost");
});

// ---- smile trainer: memes + sfx combos, rewarded by your happy score ----------

let combo = null;
let comboHideTimer = 0;
let latestShapes = {};

function drawSpark(history) {
  const canvas = ui.trainerSpark;
  const ctx = canvas.getContext("2d");
  const { width: w, height: h } = canvas;
  ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = "rgba(255,255,255,0.15)";
  ctx.beginPath();
  ctx.moveTo(0, h / 2);
  ctx.lineTo(w, h / 2);
  ctx.stroke();
  if (history.length < 2) return;
  ctx.strokeStyle = "#ffd166";
  ctx.lineWidth = 2;
  ctx.beginPath();
  history.forEach((r, i) => {
    const x = (i / (history.length - 1)) * (w - 4) + 2;
    const y = h / 2 - (r * h) / 2.4; // rewards live in [-0.3, 1]
    i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  });
  ctx.stroke();
}

function paintTrainer(stats) {
  const fmt = (s) => (s >= 60000 ? `${Math.round(s / 60000)}m` : `${Math.round(s / 1000)}s`);
  ui.trainerScore.textContent =
    `reward ${stats.recentMean >= 0 ? "+" : ""}${stats.recentMean.toFixed(2)} · ${stats.trials} combos`;
  ui.trainerGap.textContent = `next in ~${fmt(stats.nextGapMs)}`;
  drawSpark(stats.history);
  ui.trainerTop.replaceChildren(
    ...stats.top.map(({ meme, score, n }) => {
      const li = document.createElement("li");
      li.innerHTML = `<img src="assets/memes/${combo.memes[meme]?.file ?? ""}" alt="">
        <span class="top-score">${Math.round(score * 100)}%${n ? ` · ${n}` : ""}</span>`;
      return li;
    }),
  );
  ui.trainerStatus.textContent = stats.seeding
    ? `introducing itself — ${stats.seedRemaining} memes left in the first pass`
    : "training live: every combo's smile lift feeds the policy";
}

async function initTrainer() {
  try {
    const manifest = await (await fetch("./assets/manifest.7f5cb2c1.json")).json();
    const rng = makeRng(Date.now() >>> 0);
    const restored = ComboEngine.restore(rng);
    combo = new ComboEngine({
      brain: restored?.brain ?? undefined,
      memes: manifest.memes ?? [],
      sounds: manifest.sfx ?? [],
      rng,
      onCombo({ meme, sound }) {
        ui.comboImg.src = `assets/memes/${meme.file}`;
        ui.comboSound.textContent = sound.name ?? "🔊";
        ui.comboCard.classList.remove("hidden", "play");
        void ui.comboCard.offsetWidth;
        ui.comboCard.classList.add("play");
        clearTimeout(comboHideTimer);
        comboHideTimer = setTimeout(() => ui.comboCard.classList.add("hidden"), 4000);
      },
      onStats: paintTrainer,
    });
    if (restored) combo.rewardHistory = restored.history ?? [];
    combo.onStats(combo.stats());
    ui.trainerMute.addEventListener("click", () => {
      combo.setMuted(!combo.muted);
      ui.trainerMute.textContent = combo.muted ? "🔇" : "🔊";
    });
  } catch (error) {
    console.error("smile trainer failed to init", error);
    ui.trainerStatus.textContent = "trainer offline — assets missing?";
  }
}
initTrainer();
// e2e hook: headless runs force a combo without a face in frame.
window.__smileTest = () => combo;

// ---- camera --------------------------------------------------------------------

let stream = null;
let landmarker = null;
let running = false;
let rafId = 0;
let lastDetect = 0;
let cancelRequested = false;

async function getCamera() {
  stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 960 } },
    audio: false,
  });
  ui.cam.srcObject = stream;
  await ui.cam.play();
}

function cameraErrorCopy(error) {
  if (error instanceof DOMException) {
    if (error.name === "NotAllowedError" || error.name === "SecurityError")
      return "Camera permission was denied. Click the camera icon in the address bar, allow access, and try again.";
    if (error.name === "NotFoundError" || error.name === "OverconstrainedError")
      return "No usable camera was found on this device.";
    if (error.name === "NotReadableError")
      return "The camera is busy — another app may be using it. Close it and try again.";
  }
  return `The camera could not start (${error?.name ?? "unknown"}).`;
}

function releaseCamera() {
  running = false;
  cancelAnimationFrame(rafId);
  stream?.getTracks().forEach((track) => track.stop());
  stream = null;
  ui.cam.srcObject = null;
}

function stopCamera() {
  cancelRequested = true;
  releaseCamera();
  showPanel("intro");
}

function loop(now) {
  if (!running) return;
  if (landmarker && ui.cam.readyState >= 2 && now - lastDetect >= DETECT_INTERVAL_MS) {
    lastDetect = now;
    const result = landmarker.detectForVideo(ui.cam, now);
    const categories = result.faceBlendshapes?.[0]?.categories;
    if (categories?.length) {
      latestShapes = foldShapes(categories);
      tracker.update(latestShapes, now);
    } else tracker.noFace(now);
    paintBars(tracker.ema, now);
    combo?.tick(now, tracker.ema[0], faceEnergy(latestShapes), categories?.length > 0);
  }
  rafId = requestAnimationFrame(loop);
}

async function start() {
  cancelRequested = false;
  showPanel("loading");
  ui.loadingMsg.textContent = "Waking up the face model…";
  try {
    const [, loaded] = await Promise.all([getCamera(), loadLandmarker()]);
    if (cancelRequested) return; // user hit Cancel mid-load; camera is already released
    landmarker = loaded;
    tracker.reset();
    showPanel("live");
    running = true;
    lastDetect = 0;
    rafId = requestAnimationFrame(loop);
  } catch (error) {
    console.error(error);
    releaseCamera();
    if (error instanceof DOMException) ui.errorMsg.textContent = cameraErrorCopy(error);
    else ui.errorMsg.textContent = "The face model could not load (network off and no local copy). Try again.";
    showPanel("error");
  }
}

document.getElementById("start").addEventListener("click", start);
document.getElementById("retry").addEventListener("click", start);
document.getElementById("stop").addEventListener("click", stopCamera);
document.getElementById("cancel").addEventListener("click", stopCamera);

// Pause inference while the tab is hidden; the camera prompt flow is untouched.
document.addEventListener("visibilitychange", () => {
  if (document.hidden && running) {
    running = false;
    cancelAnimationFrame(rafId);
  } else if (!document.hidden && !running && stream) {
    running = true;
    rafId = requestAnimationFrame(loop);
  }
});
