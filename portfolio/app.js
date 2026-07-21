import { FaceTracker } from "./modules/face-tracker.js?v=20260721-facephys-v1";
import { FacePhysEngine } from "./modules/facephys-engine.js?v=20260721-facephys-v1";
import { drawFaceOverlay, drawSpectrum, drawWaveform } from "./modules/draw.js?v=20260721-facephys-v1";
import { QUALITY_GATES, evaluateGate, qualityLevel } from "./modules/quality-gate.js?v=20260721-facephys-v1";

const $ = (id) => document.getElementById(id);
const ui = {
  lab: $("pulseLab"), camera: $("camera"), overlay: $("faceOverlay"), sampleCanvas: $("sampleCanvas"), empty: $("cameraEmpty"), guide: $("cameraGuide"),
  start: $("startButton"), phase: $("phaseLabel"), status: $("statusChip"), hint: $("demoHint"), cameraStatus: $("cameraStatus"), faceStatus: $("faceStatus"), roiStatus: $("roiStatus"),
  bpm: $("bpmValue"), sqi: $("qualityValue"), sqiOrb: $("sqiOrb"), fps: $("fpsValue"), window: $("windowValue"), calibrationLabel: $("calibrationLabel"), calibrationBar: $("calibrationBar"),
  faceQuality: $("faceQualityValue"), lightQuality: $("lightQualityValue"), motionQuality: $("motionQualityValue"), signalQuality: $("signalQualityValue"),
  faceBar: $("faceQualityBar"), lightBar: $("lightQualityBar"), motionBar: $("motionQualityBar"), signalBar: $("signalQualityBar"),
  wave: $("waveCanvas"), spectrum: $("spectrumCanvas"), waveState: $("waveState"), peak: $("peakValue"), checklist: $("captureChecklist"),
};

const WARMUP_SECONDS = 15, FACE_INTERVAL_MS = 90, SAMPLE_INTERVAL_MS = 33;
const state = { stream: null, tracker: null, engine: null, running: false, lastFaceAt: 0, lastSampleAt: 0, lastEngineAt: 0, lastFrameAt: 0, fpsSamples: [], face: null, faceInvalid: false, metrics: emptyMetrics(), phase: "idle", frameRequest: null, result: emptyResult(), waveform: [] };
const sampleCtx = ui.sampleCanvas.getContext("2d", { willReadFrequently: true });

function emptyMetrics() { return { face: 0, light: 0, motion: 0, signal: 0, sqi: 0, fps: 0, brightness: 0, samples: 0, duration: 0, peak: 0 }; }
function emptyResult() { return { duration: 0, samples: 0, bpm: 0, sqi: 0, spectrum: [] }; }
function clamp(value, min = 0, max = 1) { return Math.min(max, Math.max(min, Number.isFinite(value) ? value : min)); }
function percent(value) { return `${Math.round(clamp(value) * 100)}%`; }
function setPhase(phase, status, hint) { state.phase = phase; ui.lab.dataset.phase = phase; ui.phase.textContent = status; ui.status.textContent = status; ui.hint.textContent = hint; }
function setChecklist(name, active) { const item = ui.checklist.querySelector(`[data-check="${name}"]`); if (!item) return; item.classList.toggle("is-pass", Boolean(active)); item.setAttribute("aria-checked", String(Boolean(active))); const icon = item.querySelector("i"); if (icon) icon.textContent = active ? "✓" : "○"; }
function resetChecklist() { ["face", "light", "motion", "signal"].forEach((key) => setChecklist(key, false)); }

async function start() {
  if (state.running || state.stream) return stop();
  if (!navigator.mediaDevices?.getUserMedia) return fail("当前浏览器不支持摄像头访问", "请使用最新版 Chrome 或 Edge 打开此页面。");
  resetSession();
  try {
    setPhase("permission", "请求摄像头权限", "请在浏览器提示中允许使用前置摄像头。"); ui.start.disabled = true;
    state.stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30, min: 20 } }, audio: false });
    ui.camera.srcObject = state.stream; await ui.camera.play(); ui.empty.hidden = true; ui.guide.hidden = false; ui.cameraStatus.textContent = "CAMERA LIVE";
    setPhase("model", "加载本地 FacePhys", "正在加载本站点的 LiteRT 模型；视频和生物数据不会离开设备。");
    state.tracker = await FaceTracker.create();
    state.engine = new FacePhysEngine(onFacePhysResult, onFacePhysError);
    await state.engine.initialize();
    state.running = true; ui.start.textContent = "停止采集 ×"; ui.start.classList.add("stop"); ui.start.disabled = false;
    setPhase("searching", "搜索人脸", "请将一张脸置于取景框中央，保持正对镜头。"); nextFrame();
  } catch (error) {
    const cameraIsVisible = Boolean(state.stream && ui.camera.videoWidth);
    if (error?.name === "NotAllowedError") fail("未获得摄像头权限", "请在浏览器地址栏的站点权限中允许摄像头，然后重试。");
    else if (cameraIsVisible) fail("FacePhys 模型未加载", "摄像头已成功开启；但本地 FacePhys 模型未能初始化。请刷新页面，确认 Vercel 已部署全部模型资源后重试。", { keepCamera: true });
    else fail("无法开启本地采集", "请确认设备存在可用摄像头，并使用 HTTPS 的 Chrome 或 Edge 页面访问。");
  } finally { ui.start.disabled = false; }
}

function stop() {
  state.running = false; if (state.frameRequest) cancelAnimationFrame(state.frameRequest); state.frameRequest = null;
  state.stream?.getTracks().forEach((track) => track.stop()); state.stream = null; ui.camera.srcObject = null;
  state.tracker?.close(); state.tracker = null; state.engine?.destroy(); state.engine = null; resetSession();
  ui.empty.hidden = false; ui.guide.hidden = false; ui.cameraStatus.textContent = "STANDBY"; ui.start.textContent = "开启摄像头 →"; ui.start.classList.remove("stop");
  setPhase("idle", "等待启动", "允许摄像头后，本机将以 FacePhys 建立脉搏信号窗口。");
}

function resetSession() {
  state.face = null; state.faceInvalid = false; state.metrics = emptyMetrics(); state.result = emptyResult(); state.waveform = []; state.lastFaceAt = 0; state.lastSampleAt = 0; state.lastEngineAt = 0; state.lastFrameAt = 0; state.fpsSamples = [];
  ui.bpm.textContent = "--"; ui.sqi.textContent = "--"; ui.sqiOrb.classList.remove("is-live"); ui.fps.textContent = "-- FPS"; ui.window.textContent = `0.0 / ${WARMUP_SECONDS.toFixed(1)} s`; ui.calibrationBar.style.width = "0%"; ui.calibrationLabel.textContent = "等待采样窗口"; ui.faceStatus.textContent = "FACE: --"; ui.roiStatus.textContent = "ROI: --"; ui.waveState.textContent = "等待有效 ROI"; ui.peak.textContent = "-- BPM";
  updateQuality(emptyMetrics()); drawFaceOverlay(ui.overlay, ui.camera, null); drawWaveform(ui.wave, []); drawSpectrum(ui.spectrum, []); resetChecklist();
}

function fail(status, hint, { keepCamera = false } = {}) {
  state.running = false; if (state.frameRequest) cancelAnimationFrame(state.frameRequest); state.frameRequest = null; state.tracker?.close(); state.tracker = null; state.engine?.destroy(); state.engine = null;
  if (keepCamera) { ui.empty.hidden = true; ui.guide.hidden = false; ui.start.textContent = "停止摄像头 ×"; ui.start.classList.add("stop"); ui.cameraStatus.textContent = "CAMERA LIVE"; ui.faceStatus.textContent = "FACE: MODEL ERROR"; ui.roiStatus.textContent = "ROI: PAUSED"; }
  else { state.stream?.getTracks().forEach((track) => track.stop()); state.stream = null; ui.camera.srcObject = null; ui.empty.hidden = false; ui.start.textContent = "重新尝试 →"; ui.start.classList.remove("stop"); ui.cameraStatus.textContent = "CAMERA ERROR"; }
  setPhase("error", status, hint);
}

function nextFrame() { if (state.running) state.frameRequest = requestAnimationFrame(processFrame); }
function processFrame(now) {
  if (!state.running || !ui.camera.videoWidth) return nextFrame(); recordFps(now);
  if (now - state.lastFaceAt >= FACE_INTERVAL_MS) { state.lastFaceAt = now; try { state.face = state.tracker.detect(ui.camera, now); } catch { state.face = null; } }
  drawFaceOverlay(ui.overlay, ui.camera, state.face);
  if (!state.face?.valid) { if (!state.faceInvalid) invalidateFace(state.face?.reason || "未检测到可用人脸"); state.faceInvalid = true; return nextFrame(); }
  state.faceInvalid = false;
  ui.guide.hidden = true;
  if (now - state.lastSampleAt >= SAMPLE_INTERVAL_MS) { state.lastSampleAt = now; const frame = sampleFace(state.face.bounds); if (frame) ingestSample(now, frame); }
  nextFrame();
}

function recordFps(now) { if (state.lastFrameAt) state.fpsSamples.push(now - state.lastFrameAt); state.lastFrameAt = now; if (state.fpsSamples.length > 30) state.fpsSamples.shift(); const mean = state.fpsSamples.reduce((sum, value) => sum + value, 0) / Math.max(1, state.fpsSamples.length); state.metrics.fps = mean ? 1000 / mean : 0; ui.fps.textContent = state.metrics.fps ? `${Math.round(state.metrics.fps)} FPS` : "-- FPS"; }

function invalidateFace(reason) {
  state.engine?.reset(); state.result = emptyResult(); state.waveform = []; state.metrics = { ...emptyMetrics(), fps: state.metrics.fps }; state.lastEngineAt = 0;
  ui.bpm.textContent = "--"; ui.sqi.textContent = "--"; ui.sqiOrb.classList.remove("is-live"); ui.faceStatus.textContent = "FACE: LOST"; ui.roiStatus.textContent = "ROI: PAUSED"; ui.window.textContent = `0.0 / ${WARMUP_SECONDS.toFixed(1)} s`; ui.calibrationBar.style.width = "0%"; ui.calibrationLabel.textContent = "等待可用人脸"; ui.waveState.textContent = "采样已暂停"; ui.peak.textContent = "-- BPM";
  updateQuality(state.metrics); drawWaveform(ui.wave, []); drawSpectrum(ui.spectrum, []); setChecklist("face", false); setChecklist("signal", false);
  const status = reason === "multiple" ? "检测到多张人脸" : "搜索人脸"; const hint = reason === "multiple" ? "请保持仅一人入镜，系统不会自动切换检测对象。" : reason === "small" ? "请靠近镜头，让脸部占据更多画面。" : reason === "pose" ? "请正对镜头，避免大幅侧脸或低头。" : "请将一张脸置于取景框中央，保持正对镜头。"; setPhase("searching", status, hint);
}

function sampleFace(bounds) {
  if (!bounds) return null; const vw = ui.camera.videoWidth, vh = ui.camera.videoHeight; const width = bounds.width, height = bounds.height * 1.2, x = clamp(bounds.x, 0, 1) * vw, y = clamp(bounds.y - bounds.height * .2, 0, 1) * vh;
  const sw = Math.min(vw - x, width * vw), sh = Math.min(vh - y, height * vh); if (sw < 24 || sh < 24) return null;
  ui.sampleCanvas.width = 36; ui.sampleCanvas.height = 36; sampleCtx.drawImage(ui.camera, x, y, sw, sh, 0, 0, 36, 36); const pixels = sampleCtx.getImageData(0, 0, 36, 36).data, frame = new Float32Array(36 * 36 * 3); let brightness = 0;
  for (let i = 0, j = 0; i < pixels.length; i += 4) { frame[j++] = pixels[i] / 255; frame[j++] = pixels[i + 1] / 255; frame[j++] = pixels[i + 2] / 255; brightness += (pixels[i] + pixels[i + 1] + pixels[i + 2]) / (3 * 255); }
  return { frame, brightness: brightness / (36 * 36) };
}

function ingestSample(now, sample) {
  const brightness = sample.brightness, light = brightness < .16 || brightness > .9 ? .08 : clamp(1 - Math.abs(brightness - .52) / .43), motion = state.face.motionQuality, face = state.face.faceQuality;
  const dt = state.lastEngineAt ? (now - state.lastEngineAt) / 1000 : 1 / 30; state.lastEngineAt = now; state.engine?.submit(sample.frame, dt, now);
  state.metrics = { ...state.metrics, brightness, light, motion, face }; ui.faceStatus.textContent = "FACE: LOCKED"; ui.roiStatus.textContent = "ROI: FACEPHYS"; setChecklist("face", face >= QUALITY_GATES.face); setChecklist("light", light >= QUALITY_GATES.light); setChecklist("motion", motion >= QUALITY_GATES.motion);
}

function onFacePhysResult(result) {
  if (!state.running || !state.face?.valid) return; state.result = { ...state.result, ...result, spectrum: Array.isArray(result.spectrum) ? result.spectrum : [] }; state.waveform.push(result.value || 0); if (state.waveform.length > 450) state.waveform.shift(); analyze();
}
function onFacePhysError(error) { if (state.running) fail("FacePhys 推理异常", `本地模型已停止：${error.message || "未知错误"}。请刷新页面后重试。`, { keepCamera: Boolean(state.stream) }); }

function analyze() {
  const result = state.result, signal = result.sqi || 0, metrics = { ...state.metrics, signal, sqi: result.sqi || 0, samples: result.samples || 0, duration: result.duration || 0, peak: result.bpm || 0 }; state.metrics = metrics;
  const readiness = clamp(metrics.duration / WARMUP_SECONDS); ui.window.textContent = `${Math.min(metrics.duration, WARMUP_SECONDS).toFixed(1)} / ${WARMUP_SECONDS.toFixed(1)} s`; ui.calibrationBar.style.width = percent(readiness); ui.calibrationLabel.textContent = metrics.duration < WARMUP_SECONDS ? "FacePhys 正在建立时序窗口" : "FacePhys 信号窗口已建立";
  const gate = evaluateGate(metrics, result, WARMUP_SECONDS); ui.waveState.textContent = metrics.duration < WARMUP_SECONDS ? "正在累积 FacePhys 时序样本" : gate.code === "signal" ? "FacePhys 质量门控未通过" : gate.accepted ? "FacePhys 本地推理中" : `等待${gate.title}`; ui.peak.textContent = result.bpm ? `${Math.round(result.bpm)} BPM` : "-- BPM";
  updateQuality(metrics); drawWaveform(ui.wave, state.waveform); drawSpectrum(ui.spectrum, result.spectrum, result.bpm); setChecklist("face", metrics.face >= QUALITY_GATES.face); setChecklist("light", metrics.light >= QUALITY_GATES.light); setChecklist("motion", metrics.motion >= QUALITY_GATES.motion); setChecklist("signal", gate.accepted);
  if (gate.accepted) { ui.bpm.textContent = Math.round(result.bpm); ui.sqi.textContent = result.sqi.toFixed(2); ui.sqiOrb.classList.add("is-live"); setPhase("live", gate.title, gate.hint); }
  else { ui.bpm.textContent = "--"; ui.sqi.textContent = result.sqi ? result.sqi.toFixed(2) : "--"; ui.sqiOrb.classList.remove("is-live"); if (gate.candidateBpm) ui.waveState.textContent = `候选峰值 ${Math.round(gate.candidateBpm)} BPM，等待条件达标`; setPhase("calibrating", gate.title, gate.hint); }
}

function updateQuality(metrics) { [[ui.faceQuality, ui.faceBar, metrics.face], [ui.lightQuality, ui.lightBar, metrics.light], [ui.motionQuality, ui.motionBar, metrics.motion], [ui.signalQuality, ui.signalBar, metrics.signal]].forEach(([value, bar, score]) => { value.textContent = score ? percent(score) : "--"; bar.style.width = percent(score); bar.closest(".quality-card").dataset.level = qualityLevel(score); }); }
function handleVisibility() { if (document.hidden && state.running) { setPhase("paused", "采集已暂停", "页面进入后台后已暂停采样；返回此页后请重新开始。"); stop(); } }
ui.start.addEventListener("click", () => state.stream ? stop() : start()); window.addEventListener("resize", () => drawFaceOverlay(ui.overlay, ui.camera, state.face)); window.addEventListener("beforeunload", stop); document.addEventListener("visibilitychange", handleVisibility); resetSession();
