import { FaceTracker } from "./modules/face-tracker.js?v=20260721-pulselab-v1";
import { SignalProcessor } from "./modules/signal-processor.js?v=20260721-pulselab-v1";
import { drawFaceOverlay, drawSpectrum, drawWaveform } from "./modules/draw.js?v=20260721-pulselab-v1";

const $ = (id) => document.getElementById(id);
const ui = {
  lab: $("pulseLab"), camera: $("camera"), overlay: $("faceOverlay"), sampleCanvas: $("sampleCanvas"), empty: $("cameraEmpty"), guide: $("cameraGuide"),
  start: $("startButton"), phase: $("phaseLabel"), status: $("statusChip"), hint: $("demoHint"), cameraStatus: $("cameraStatus"), faceStatus: $("faceStatus"), roiStatus: $("roiStatus"),
  bpm: $("bpmValue"), sqi: $("qualityValue"), sqiOrb: $("sqiOrb"), fps: $("fpsValue"), window: $("windowValue"), calibrationLabel: $("calibrationLabel"), calibrationBar: $("calibrationBar"),
  faceQuality: $("faceQualityValue"), lightQuality: $("lightQualityValue"), motionQuality: $("motionQualityValue"), signalQuality: $("signalQualityValue"),
  faceBar: $("faceQualityBar"), lightBar: $("lightQualityBar"), motionBar: $("motionQualityBar"), signalBar: $("signalQualityBar"),
  wave: $("waveCanvas"), spectrum: $("spectrumCanvas"), waveState: $("waveState"), peak: $("peakValue"), checklist: $("captureChecklist"),
};

const WARMUP_SECONDS = 12;
const FACE_INTERVAL_MS = 90;
const SAMPLE_INTERVAL_MS = 33;
const state = { stream: null, tracker: null, processor: new SignalProcessor(), running: false, lastFaceAt: 0, lastSampleAt: 0, lastAnalysisAt: 0, lastFrameAt: 0, fpsSamples: [], face: null, metrics: emptyMetrics(), phase: "idle", frameRequest: null };
const sampleCtx = ui.sampleCanvas.getContext("2d", { willReadFrequently: true });

function emptyMetrics() { return { face: 0, light: 0, motion: 0, signal: 0, sqi: 0, fps: 0, brightness: 0, samples: 0, duration: 0, peak: 0 }; }
function clamp(value, min = 0, max = 1) { return Math.min(max, Math.max(min, Number.isFinite(value) ? value : min)); }
function percent(value) { return `${Math.round(clamp(value) * 100)}%`; }
function setPhase(phase, status, hint) { state.phase = phase; ui.lab.dataset.phase = phase; ui.phase.textContent = status; ui.status.textContent = status; ui.hint.textContent = hint; }
function setChecklist(name, active) { const item = ui.checklist.querySelector(`[data-check="${name}"]`); if (item) item.classList.toggle("is-pass", Boolean(active)); }
function resetChecklist() { ["face", "light", "motion", "signal"].forEach((key) => setChecklist(key, false)); }

async function start() {
  if (state.running) return stop();
  if (!navigator.mediaDevices?.getUserMedia) return fail("当前浏览器不支持摄像头访问", "请使用最新版 Chrome 或 Edge 打开此页面。");
  resetSession();
  try {
    setPhase("permission", "请求摄像头权限", "请在浏览器提示中允许使用前置摄像头。");
    ui.start.disabled = true;
    state.stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30, min: 20 } }, audio: false });
    ui.camera.srcObject = state.stream;
    await ui.camera.play();
    ui.empty.hidden = true;
    ui.guide.hidden = false;
    ui.cameraStatus.textContent = "CAMERA LIVE";
    setPhase("model", "加载本地人脸模型", "模型文件来自本站点，正在本机初始化。");
    state.tracker = await FaceTracker.create();
    state.running = true;
    ui.start.textContent = "停止采集 ×";
    ui.start.classList.add("stop");
    ui.start.disabled = false;
    setPhase("searching", "搜索人脸", "请将一张脸置于取景框中央，保持正对镜头。");
    nextFrame();
  } catch (error) {
    const message = error?.name === "NotAllowedError" ? "请在浏览器地址栏的站点权限中允许摄像头，然后重试。" : "未能初始化本地摄像头或人脸模型，请检查设备与网络缓存后重试。";
    fail("无法开启本地采集", message);
  } finally { ui.start.disabled = false; }
}

function stop() {
  state.running = false;
  if (state.frameRequest) cancelAnimationFrame(state.frameRequest);
  state.frameRequest = null;
  state.stream?.getTracks().forEach((track) => track.stop());
  state.stream = null;
  ui.camera.srcObject = null;
  state.tracker?.close();
  state.tracker = null;
  resetSession();
  ui.empty.hidden = false;
  ui.guide.hidden = false;
  ui.cameraStatus.textContent = "STANDBY";
  ui.start.textContent = "开启摄像头 →";
  ui.start.classList.remove("stop");
  setPhase("idle", "等待启动", "允许摄像头后，本机将检测人脸并开始校准。");
}

function resetSession() {
  state.processor.reset(); state.face = null; state.metrics = emptyMetrics(); state.lastFaceAt = 0; state.lastSampleAt = 0; state.lastAnalysisAt = 0; state.lastFrameAt = 0; state.fpsSamples = [];
  ui.bpm.textContent = "--"; ui.sqi.textContent = "--"; ui.sqiOrb.classList.remove("is-live"); ui.fps.textContent = "-- FPS"; ui.window.textContent = `0.0 / ${WARMUP_SECONDS.toFixed(1)} s`; ui.calibrationBar.style.width = "0%"; ui.calibrationLabel.textContent = "等待采样窗口"; ui.faceStatus.textContent = "FACE: --"; ui.roiStatus.textContent = "ROI: --"; ui.waveState.textContent = "等待有效 ROI"; ui.peak.textContent = "-- BPM";
  updateQuality(emptyMetrics()); drawFaceOverlay(ui.overlay, ui.camera, null); drawWaveform(ui.wave, []); drawSpectrum(ui.spectrum, []); resetChecklist();
}

function fail(status, hint) {
  state.running = false;
  state.stream?.getTracks().forEach((track) => track.stop());
  state.stream = null;
  state.tracker?.close(); state.tracker = null;
  ui.camera.srcObject = null; ui.empty.hidden = false; ui.start.textContent = "重新尝试 →"; ui.start.classList.remove("stop"); ui.cameraStatus.textContent = "CAMERA ERROR";
  setPhase("error", status, hint);
}

function nextFrame() {
  if (!state.running) return;
  state.frameRequest = requestAnimationFrame(processFrame);
}

function processFrame(now) {
  if (!state.running || !ui.camera.videoWidth) return nextFrame();
  recordFps(now);
  if (now - state.lastFaceAt >= FACE_INTERVAL_MS) {
    state.lastFaceAt = now;
    try { state.face = state.tracker.detect(ui.camera, now); } catch { state.face = null; }
  }
  drawFaceOverlay(ui.overlay, ui.camera, state.face);
  if (!state.face?.valid) {
    invalidateFace(state.face?.reason || "未检测到可用人脸");
    return nextFrame();
  }
  ui.guide.hidden = true;
  if (now - state.lastSampleAt >= SAMPLE_INTERVAL_MS) {
    state.lastSampleAt = now;
    const color = sampleRois(state.face.rois);
    if (color) ingestSample(now, color);
  }
  if (now - state.lastAnalysisAt > 700) { state.lastAnalysisAt = now; analyze(); }
  nextFrame();
}

function recordFps(now) {
  if (state.lastFrameAt) state.fpsSamples.push(now - state.lastFrameAt);
  state.lastFrameAt = now;
  if (state.fpsSamples.length > 30) state.fpsSamples.shift();
  const mean = state.fpsSamples.reduce((sum, value) => sum + value, 0) / Math.max(1, state.fpsSamples.length);
  state.metrics.fps = mean ? 1000 / mean : 0;
  ui.fps.textContent = state.metrics.fps ? `${Math.round(state.metrics.fps)} FPS` : "-- FPS";
}

function invalidateFace(reason) {
  state.processor.reset();
  state.metrics = { ...emptyMetrics(), fps: state.metrics.fps };
  ui.bpm.textContent = "--"; ui.sqi.textContent = "--"; ui.sqiOrb.classList.remove("is-live"); ui.faceStatus.textContent = "FACE: LOST"; ui.roiStatus.textContent = "ROI: PAUSED"; ui.window.textContent = `0.0 / ${WARMUP_SECONDS.toFixed(1)} s`; ui.calibrationBar.style.width = "0%"; ui.calibrationLabel.textContent = "等待可用人脸"; ui.waveState.textContent = "采样已暂停"; ui.peak.textContent = "-- BPM";
  updateQuality(state.metrics); drawWaveform(ui.wave, []); drawSpectrum(ui.spectrum, []); setChecklist("face", false); setChecklist("signal", false);
  const status = reason === "multiple" ? "检测到多张人脸" : "搜索人脸";
  const hint = reason === "multiple" ? "请保持仅一人入镜，系统不会自动切换检测对象。" : reason === "small" ? "请靠近镜头，让脸部占据更多画面。" : reason === "pose" ? "请正对镜头，避免大幅侧脸或低头。" : "请将一张脸置于取景框中央，保持正对镜头。";
  setPhase("searching", status, hint);
}

function sampleRois(rois) {
  const width = 192, height = Math.round(width * ui.camera.videoHeight / ui.camera.videoWidth);
  ui.sampleCanvas.width = width; ui.sampleCanvas.height = height;
  sampleCtx.drawImage(ui.camera, 0, 0, width, height);
  const pixels = sampleCtx.getImageData(0, 0, width, height).data;
  let r = 0, g = 0, b = 0, count = 0;
  for (const polygon of rois) {
    const points = polygon.map((point) => ({ x: point.x * width, y: point.y * height }));
    const bounds = polygonBounds(points, width, height);
    for (let y = bounds.y0; y <= bounds.y1; y += 2) for (let x = bounds.x0; x <= bounds.x1; x += 2) {
      if (!pointInPolygon(x, y, points)) continue;
      const index = (y * width + x) * 4; const rr = pixels[index], gg = pixels[index + 1], bb = pixels[index + 2];
      const spread = Math.max(rr, gg, bb) - Math.min(rr, gg, bb);
      if (rr < 24 || gg < 20 || bb < 14 || rr > 245 || gg > 245 || bb > 245 || spread < 3) continue;
      r += rr; g += gg; b += bb; count += 1;
    }
  }
  return count > 60 ? { r: r / count, g: g / count, b: b / count, count } : null;
}

function polygonBounds(points, width, height) { return { x0: Math.max(0, Math.floor(Math.min(...points.map((p) => p.x)))), x1: Math.min(width - 1, Math.ceil(Math.max(...points.map((p) => p.x)))), y0: Math.max(0, Math.floor(Math.min(...points.map((p) => p.y)))), y1: Math.min(height - 1, Math.ceil(Math.max(...points.map((p) => p.y)))) }; }
function pointInPolygon(x, y, points) { let inside = false; for (let i = 0, j = points.length - 1; i < points.length; j = i++) { const a = points[i], b = points[j]; if (((a.y > y) !== (b.y > y)) && x < ((b.x - a.x) * (y - a.y)) / (b.y - a.y) + a.x) inside = !inside; } return inside; }

function ingestSample(now, color) {
  const brightness = (color.r + color.g + color.b) / (3 * 255);
  const light = brightness < 0.16 || brightness > 0.9 ? 0.08 : clamp(1 - Math.abs(brightness - 0.52) / 0.43);
  const motion = state.face.motionQuality;
  const face = state.face.faceQuality;
  state.processor.push({ t: now, ...color });
  state.metrics = { ...state.metrics, brightness, light, motion, face, samples: state.processor.count, duration: state.processor.durationSeconds };
  ui.faceStatus.textContent = "FACE: LOCKED"; ui.roiStatus.textContent = "ROI: 3 ZONES"; setChecklist("face", true); setChecklist("light", light >= 0.55); setChecklist("motion", motion >= 0.55);
}

function analyze() {
  const result = state.processor.analyze();
  const metrics = { ...state.metrics, signal: result.signalQuality, sqi: result.sqi, duration: result.duration, peak: result.bpm || 0 };
  state.metrics = metrics;
  const readiness = clamp(result.duration / WARMUP_SECONDS);
  ui.window.textContent = `${Math.min(result.duration, WARMUP_SECONDS).toFixed(1)} / ${WARMUP_SECONDS.toFixed(1)} s`;
  ui.calibrationBar.style.width = percent(readiness);
  ui.calibrationLabel.textContent = result.duration < WARMUP_SECONDS ? "建立稳定色彩窗口" : "采样窗口已建立";
  ui.waveState.textContent = result.duration < WARMUP_SECONDS ? "正在累积样本" : result.signalQuality >= 0.45 ? "信号已滤波" : "信号质量不足";
  ui.peak.textContent = result.bpm ? `${Math.round(result.bpm)} BPM` : "-- BPM";
  updateQuality(metrics); drawWaveform(ui.wave, result.waveform); drawSpectrum(ui.spectrum, result.spectrum, result.bpm);
  const accepted = result.duration >= WARMUP_SECONDS && metrics.face >= 0.55 && metrics.light >= 0.55 && metrics.motion >= 0.55 && result.sqi >= 0.42;
  setChecklist("signal", accepted);
  if (accepted) {
    ui.bpm.textContent = Math.round(result.bpm); ui.sqi.textContent = result.sqi.toFixed(2); ui.sqiOrb.classList.add("is-live");
    setPhase("live", "稳定监测", "已检测到稳定的本地色彩周期变化；请保持自然呼吸与相对静止。");
  } else {
    ui.bpm.textContent = "--"; ui.sqi.textContent = result.sqi ? result.sqi.toFixed(2) : "--"; ui.sqiOrb.classList.remove("is-live");
    const issue = metrics.light < 0.55 ? "光照需调整" : metrics.motion < 0.55 ? "请保持静止" : result.duration < WARMUP_SECONDS ? "建立信号中" : "信号质量不足";
    const hint = metrics.light < 0.55 ? "避免逆光、过暗或明显过曝，让面部均匀受光。" : metrics.motion < 0.55 ? "请减少头部移动，保持正对镜头数秒。" : result.duration < WARMUP_SECONDS ? `正在累积色彩窗口，还需 ${Math.max(0, WARMUP_SECONDS - result.duration).toFixed(1)} 秒。` : "已暂停心率输出；请改善光照并保持脸部稳定。";
    setPhase("calibrating", issue, hint);
  }
}

function updateQuality(metrics) {
  const items = [[ui.faceQuality, ui.faceBar, metrics.face], [ui.lightQuality, ui.lightBar, metrics.light], [ui.motionQuality, ui.motionBar, metrics.motion], [ui.signalQuality, ui.signalBar, metrics.signal]];
  items.forEach(([value, bar, score]) => { value.textContent = score ? percent(score) : "--"; bar.style.width = percent(score); });
}

function handleVisibility() { if (document.hidden && state.running) { setPhase("paused", "采集已暂停", "页面进入后台后已暂停采样；返回此页后请重新开始。 "); stop(); } }
ui.start.addEventListener("click", start);
window.addEventListener("resize", () => { drawFaceOverlay(ui.overlay, ui.camera, state.face); });
window.addEventListener("beforeunload", stop);
document.addEventListener("visibilitychange", handleVisibility);
resetSession();
