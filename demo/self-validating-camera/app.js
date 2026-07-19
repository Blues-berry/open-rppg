const ui = {
  video: document.getElementById("cameraFeed"),
  canvas: document.getElementById("previewCanvas"),
  light: document.getElementById("challengeLight"),
  cameraMode: document.getElementById("cameraMode"),
  trustState: document.getElementById("trustState"),
  phaseLabel: document.getElementById("phaseLabel"),
  stageHint: document.getElementById("stageHint"),
  trustRing: document.getElementById("trustRing"),
  trustScore: document.getElementById("trustScore"),
  certificateText: document.getElementById("certificateText"),
  certId: document.getElementById("certId"),
  baselineText: document.getElementById("baselineText"),
  riskLevel: document.getElementById("riskLevel"),
  riskTitle: document.getElementById("riskTitle"),
  riskReason: document.getElementById("riskReason"),
  eventLog: document.getElementById("eventLog"),
  apiOutput: document.getElementById("apiOutput"),
  startCameraBtn: document.getElementById("startCameraBtn"),
  calibrateBtn: document.getElementById("calibrateBtn"),
  lightBtn: document.getElementById("lightBtn"),
  occlusionBtn: document.getElementById("occlusionBtn"),
  riskBtn: document.getElementById("riskBtn"),
  resetBtn: document.getElementById("resetBtn"),
  metrics: {
    roi: [document.getElementById("roiValue"), document.getElementById("roiMeter")],
    lighting: [document.getElementById("lightingValue"), document.getElementById("lightingMeter")],
    motion: [document.getElementById("motionValue"), document.getElementById("motionMeter")],
    color: [document.getElementById("colorValue"), document.getElementById("colorMeter")],
    sqi: [document.getElementById("sqiValue"), document.getElementById("sqiMeter")],
    hr: [document.getElementById("hrValue"), document.getElementById("hrMeter")],
  },
};

const ctx = ui.canvas.getContext("2d", { willReadFrequently: true });
const steps = ["device", "liveness", "color", "static", "cert"];

const state = {
  stream: null,
  useCamera: false,
  cameraReady: false,
  calibrated: false,
  calibrating: false,
  certId: null,
  baselineHr: null,
  trust: 0,
  metrics: {
    roi: 0.74,
    lighting: 0.72,
    motion: 0.82,
    color: 0.5,
    sqi: 0.22,
    hr: null,
  },
  lastFrameLum: null,
  badLight: false,
  occlusion: false,
  riskMode: false,
  challengeColor: null,
  lastRiskLevel: "LOCK",
  events: [],
  startedAt: performance.now(),
};

function clamp(value, min = 0, max = 1) {
  return Math.min(max, Math.max(min, value));
}

function percent(value) {
  return `${Math.round(clamp(value) * 100)}%`;
}

function sleep(ms) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function setPill(el, label, className) {
  el.textContent = label;
  el.className = `status-pill ${className}`;
}

function setStep(name, status) {
  const el = document.getElementById(`step-${name}`);
  el.classList.remove("active", "done", "fail");
  if (status) el.classList.add(status);
}

function resetSteps() {
  steps.forEach((step) => setStep(step, ""));
}

function addEvent(title, detail) {
  const time = new Date().toLocaleTimeString("zh-CN", { hour12: false });
  state.events.unshift({ time, title, detail });
  state.events = state.events.slice(0, 6);
}

async function startCamera() {
  if (state.cameraReady) return;

  try {
    const cameraRequest = navigator.mediaDevices?.getUserMedia
      ? navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 1280 },
            height: { ideal: 720 },
            facingMode: "user",
          },
          audio: false,
        })
      : Promise.reject(new Error("media_devices_unavailable"));
    const timeout = new Promise((_, reject) => {
      window.setTimeout(() => reject(new Error("camera_permission_timeout")), 3000);
    });

    state.stream = await Promise.race([cameraRequest, timeout]);
    ui.video.srcObject = state.stream;
    await ui.video.play();
    state.useCamera = true;
    state.cameraReady = true;
    setPill(ui.cameraMode, "摄像头已连接", "ready");
    ui.stageHint.textContent = "摄像头已连接。运行可信校验后，系统才会开放生理状态。";
    addEvent("摄像头接入", "已进入真实画面采集模式。");
  } catch (error) {
    state.useCamera = false;
    state.cameraReady = true;
    setPill(ui.cameraMode, "模拟器模式", "ready");
    ui.stageHint.textContent = "未获得摄像头权限，已切换到可演示的模拟采集环境。";
    addEvent("模拟器接管", "摄像头不可用，仍可演示可信校验与风险闭环。");
  }
}

async function runCalibration() {
  if (state.calibrating) return;
  if (!state.cameraReady) await startCamera();

  state.calibrating = true;
  state.calibrated = false;
  state.certId = null;
  state.baselineHr = null;
  state.riskMode = false;
  ui.riskBtn.classList.remove("active");
  resetSteps();
  setPill(ui.trustState, "正在自证", "locked");

  ui.phaseLabel.textContent = "SELF TEST";
  ui.stageHint.textContent = "设备自检中：检查亮度、ROI、帧稳定性和延迟。";
  setStep("device", "active");
  await pulseMetrics({ roi: 0.82, lighting: 0.8, motion: 0.78, color: 0.52, sqi: 0.34 }, 900);
  setStep("device", "done");

  ui.phaseLabel.textContent = "LIVENESS";
  ui.stageHint.textContent = "活体动作挑战：请眨眼、张嘴或轻微转头。";
  setStep("liveness", "active");
  await pulseMetrics({ roi: 0.9, lighting: 0.84, motion: 0.76, color: 0.62, sqi: 0.42 }, 1000);
  setStep("liveness", "done");

  ui.phaseLabel.textContent = "COLOR";
  ui.stageHint.textContent = "多色光响应校验：系统验证摄像头颜色通道和皮肤反射响应。";
  setStep("color", "active");
  const colors = ["#ed5b52", "#56c78f", "#5ab0ff", "#ffffff"];
  for (const color of colors) {
    state.challengeColor = color;
    ui.light.style.backgroundColor = color;
    ui.light.classList.add("on");
    await pulseMetrics({ roi: 0.9, lighting: 0.86, motion: 0.82, color: 0.82, sqi: 0.52 }, 360);
    ui.light.classList.remove("on");
    await sleep(160);
  }
  state.challengeColor = null;
  setStep("color", "done");

  ui.phaseLabel.textContent = "STATIC";
  ui.stageHint.textContent = "静止采样窗口：建立当班基线，评估 BVP 波形和 SQI。";
  setStep("static", "active");
  await pulseMetrics({ roi: 0.93, lighting: 0.88, motion: 0.9, color: 0.86, sqi: 0.82 }, 1300);
  setStep("static", "done");

  ui.phaseLabel.textContent = "CERT";
  setStep("cert", "active");
  state.certId = `CERT-${Math.floor(1000 + Math.random() * 9000)}`;
  state.baselineHr = Math.round(68 + Math.random() * 8);
  state.calibrated = true;
  state.calibrating = false;
  state.metrics.sqi = Math.max(state.metrics.sqi, 0.82);
  setStep("cert", "done");
  setPill(ui.trustState, "可信证书已建立", "valid");
  ui.stageHint.textContent = "可信证书已建立。现在可以输出安全事件，但仍会被质量门控约束。";
  addEvent("可信证书生成", `当班基线 ${state.baselineHr} BPM，允许进入班中安全哨兵。`);
}

async function pulseMetrics(target, duration) {
  const start = performance.now();
  const initial = { ...state.metrics };
  while (performance.now() - start < duration) {
    const t = (performance.now() - start) / duration;
    Object.keys(target).forEach((key) => {
      state.metrics[key] = initial[key] + (target[key] - initial[key]) * t;
    });
    await sleep(40);
  }
  Object.assign(state.metrics, target);
}

function drawSimulator(width, height, time) {
  const pulse = Math.sin(time * 3.2) * 0.5 + 0.5;
  ctx.fillStyle = "#101719";
  ctx.fillRect(0, 0, width, height);

  ctx.fillStyle = "#1b2428";
  ctx.fillRect(0, height * 0.08, width, height * 0.84);

  ctx.fillStyle = "#222d32";
  ctx.fillRect(width * 0.06, height * 0.12, width * 0.22, height * 0.12);
  ctx.fillRect(width * 0.72, height * 0.12, width * 0.22, height * 0.12);

  const cx = width * 0.5;
  const cy = height * 0.48;
  const skin = state.badLight ? "#5a483b" : "#b88769";
  ctx.fillStyle = skin;
  ctx.beginPath();
  ctx.ellipse(cx, cy, width * 0.115, height * 0.195, 0, 0, Math.PI * 2);
  ctx.fill();

  ctx.fillStyle = "#2a1f1e";
  ctx.beginPath();
  ctx.ellipse(cx, cy - height * 0.13, width * 0.12, height * 0.07, 0, Math.PI, 0);
  ctx.fill();

  ctx.fillStyle = "#15191a";
  ctx.fillRect(cx - width * 0.062, cy - height * 0.025, width * 0.035, height * 0.012);
  ctx.fillRect(cx + width * 0.027, cy - height * 0.025, width * 0.035, height * 0.012);

  ctx.strokeStyle = `rgba(86, 199, 143, ${0.45 + pulse * 0.35})`;
  ctx.lineWidth = 3;
  ctx.beginPath();
  for (let i = 0; i < 96; i += 1) {
    const x = width * 0.33 + i * width * 0.0035;
    const y = height * 0.76 + Math.sin(i * 0.4 + time * 6) * height * 0.025;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
}

function drawCamera(width, height) {
  ctx.save();
  ctx.scale(-1, 1);
  ctx.drawImage(ui.video, -width, 0, width, height);
  ctx.restore();
}

function drawOverlays(width, height) {
  const roi = {
    x: width * 0.34,
    y: height * 0.18,
    w: width * 0.32,
    h: height * 0.54,
  };

  if (state.badLight) {
    ctx.fillStyle = "rgba(250, 187, 68, 0.22)";
    ctx.fillRect(0, 0, width, height);
    ctx.fillStyle = "rgba(0, 0, 0, 0.24)";
    ctx.fillRect(0, 0, width, height);
  }

  if (state.occlusion) {
    ctx.fillStyle = "rgba(6, 9, 10, 0.88)";
    ctx.fillRect(roi.x + roi.w * 0.08, roi.y + roi.h * 0.1, roi.w * 0.84, roi.h * 0.54);
  }

  ctx.strokeStyle = state.calibrated ? "#56c78f" : "#f2b84b";
  ctx.lineWidth = 4;
  ctx.setLineDash([12, 10]);
  ctx.strokeRect(roi.x, roi.y, roi.w, roi.h);
  ctx.setLineDash([]);

  ctx.fillStyle = "rgba(16, 20, 22, 0.72)";
  ctx.fillRect(24, 24, 360, 102);
  ctx.fillStyle = "#edf3f0";
  ctx.font = "700 24px Segoe UI, Arial";
  ctx.fillText(state.calibrated ? "TRUSTED SIGNAL GATE" : "SIGNAL LOCKED", 44, 62);
  ctx.fillStyle = "#9caaa8";
  ctx.font = "18px Segoe UI, Arial";
  ctx.fillText(state.calibrated ? "Quality gates are active" : "Run self-validation first", 44, 96);

  const usable = isCaptureUsable();
  ctx.fillStyle = usable ? "rgba(86, 199, 143, 0.18)" : "rgba(239, 106, 97, 0.18)";
  ctx.fillRect(width - 330, 24, 306, 68);
  ctx.fillStyle = usable ? "#56c78f" : "#ef6a61";
  ctx.font = "800 24px Segoe UI, Arial";
  ctx.fillText(usable ? "USABLE" : "UNDETERMINED", width - 306, 66);
}

function sampleFrameMetrics(width, height) {
  if (!state.useCamera || !state.cameraReady || ui.video.readyState < 2) {
    const time = (performance.now() - state.startedAt) / 1000;
    const base = state.badLight ? 0.44 : 0.82 + Math.sin(time * 0.8) * 0.025;
    state.metrics.roi = state.occlusion ? 0.36 : base + 0.06;
    state.metrics.lighting = state.badLight ? 0.34 : 0.86 + Math.sin(time * 0.65) * 0.03;
    state.metrics.motion = state.occlusion ? 0.58 : 0.89 + Math.sin(time * 0.7) * 0.025;
    if (!state.calibrating) {
      state.metrics.color = state.calibrated ? 0.86 : 0.5;
      state.metrics.sqi = state.calibrated && !state.occlusion && !state.badLight ? 0.83 : 0.22;
    }
    return;
  }

  const sx = Math.floor(width * 0.38);
  const sy = Math.floor(height * 0.25);
  const sw = Math.floor(width * 0.24);
  const sh = Math.floor(height * 0.34);
  const image = ctx.getImageData(sx, sy, sw, sh).data;
  let lum = 0;
  let lumSq = 0;
  let red = 0;
  let green = 0;
  let blue = 0;
  const pixels = image.length / 4;

  for (let i = 0; i < image.length; i += 4) {
    const l = image[i] * 0.2126 + image[i + 1] * 0.7152 + image[i + 2] * 0.0722;
    lum += l;
    lumSq += l * l;
    red += image[i];
    green += image[i + 1];
    blue += image[i + 2];
  }

  lum /= pixels;
  lumSq /= pixels;
  red /= pixels;
  green /= pixels;
  blue /= pixels;
  const std = Math.sqrt(Math.max(0, lumSq - lum * lum));
  const brightnessQuality = clamp(1 - Math.abs(lum / 255 - 0.55) / 0.5);
  const contrastQuality = clamp(std / 52);
  const frameDelta = state.lastFrameLum == null ? 0 : Math.abs(lum - state.lastFrameLum);
  state.lastFrameLum = lum;

  state.metrics.roi = state.occlusion ? 0.28 : clamp(contrastQuality * 0.45 + brightnessQuality * 0.55);
  state.metrics.lighting = state.badLight ? 0.3 : brightnessQuality;
  state.metrics.motion = state.occlusion ? 0.45 : clamp(1 - frameDelta / 38);
  if (!state.calibrating) {
    const colorSpread = (Math.max(red, green, blue) - Math.min(red, green, blue)) / 255;
    state.metrics.color = state.calibrated ? clamp(0.78 + colorSpread * 0.18) : clamp(0.45 + colorSpread * 0.2);
    state.metrics.sqi = state.calibrated
      ? clamp(state.metrics.roi * 0.32 + state.metrics.lighting * 0.24 + state.metrics.motion * 0.24 + state.metrics.color * 0.2)
      : 0.24;
  }
}

function isCaptureUsable() {
  return (
    state.calibrated &&
    state.metrics.roi >= 0.66 &&
    state.metrics.lighting >= 0.62 &&
    state.metrics.motion >= 0.58 &&
    state.metrics.color >= 0.62 &&
    state.metrics.sqi >= 0.68
  );
}

function updateRisk() {
  const usable = isCaptureUsable();
  if (!state.calibrated) {
    state.lastRiskLevel = "LOCK";
    state.metrics.hr = null;
    return {
      level: "LOCK",
      title: "等待可信校验",
      reason: "当前不允许读取心率、HRV 或趋势状态。",
      confidence: 0,
      action: "run_self_validation",
    };
  }

  if (!usable) {
    state.lastRiskLevel = "L0";
    state.metrics.hr = null;
    return {
      level: "L0",
      title: "不可判定",
      reason: "采集质量不足，系统拒绝输出生理状态。",
      confidence: 0.94,
      action: "fix_capture_environment",
    };
  }

  const time = (performance.now() - state.startedAt) / 1000;
  const baseline = state.baselineHr || 72;
  if (state.riskMode) {
    const rise = 15 + Math.sin(time * 1.6) * 5;
    state.metrics.hr = baseline + rise;
    const level = rise > 17 ? "L3" : "L2";
    if (state.lastRiskLevel !== level) {
      addEvent(level === "L3" ? "风险升级到 L3" : "风险提醒 L2", "采集可信，多个窗口连续偏离当班基线。");
    }
    state.lastRiskLevel = level;
    return {
      level,
      title: level === "L3" ? "需要人工复核" : "建议本人确认状态",
      reason: "连续窗口偏离个人当班基线，SQI 与 ROI 均达标。",
      confidence: level === "L3" ? 0.84 : 0.72,
      action: level === "L3" ? "notify_supervisor" : "prompt_worker",
    };
  }

  state.metrics.hr = baseline + Math.sin(time * 0.8) * 2.4;
  state.lastRiskLevel = "L1";
  return {
    level: "L1",
    title: "正常观察",
    reason: "采集可信，状态围绕当班基线轻微波动。",
    confidence: 0.69,
    action: "observe",
  };
}

function updateTrust() {
  const m = state.metrics;
  const quality = m.roi * 0.24 + m.lighting * 0.22 + m.motion * 0.18 + m.color * 0.16 + m.sqi * 0.2;
  state.trust = state.calibrated ? quality : quality * 0.45;
  const score = Math.round(clamp(state.trust) * 100);
  const color = score >= 72 ? "#56c78f" : score >= 48 ? "#f2b84b" : "#ef6a61";
  ui.trustScore.textContent = score;
  ui.trustRing.style.background =
    `radial-gradient(circle at center, var(--surface) 58%, transparent 59%), conic-gradient(${color} ${score * 3.6}deg, #293338 0deg)`;
}

function updateMetrics() {
  const m = state.metrics;
  const entries = [
    ["roi", m.roi],
    ["lighting", m.lighting],
    ["motion", m.motion],
    ["color", m.color],
    ["sqi", m.sqi],
  ];
  entries.forEach(([key, value]) => {
    ui.metrics[key][0].textContent = percent(value);
    ui.metrics[key][1].value = Math.round(clamp(value) * 100);
  });

  if (m.hr == null) {
    ui.metrics.hr[0].textContent = "LOCK";
    ui.metrics.hr[1].value = 45;
  } else {
    ui.metrics.hr[0].textContent = `${Math.round(m.hr)} BPM`;
    ui.metrics.hr[1].value = Math.round(m.hr);
  }
}

function updateCertificate() {
  if (!state.calibrated) {
    ui.certificateText.textContent = "尚未建立可信证书，生理指标保持锁定。";
    ui.certId.textContent = "未生成";
    ui.baselineText.textContent = "等待校验";
    return;
  }

  if (!isCaptureUsable()) {
    ui.certificateText.textContent = "证书仍有效，但当前画面未通过质量门控，系统暂停输出指标。";
  } else {
    ui.certificateText.textContent = "本次采集通过自证可信流程，安全事件可被业务系统读取。";
  }
  ui.certId.textContent = state.certId;
  ui.baselineText.textContent = `${state.baselineHr} BPM / 当班参考`;
}

function updateRiskBoard(risk) {
  ui.riskLevel.textContent = risk.level;
  ui.riskLevel.className = "risk-badge";
  if (risk.level === "LOCK") ui.riskLevel.classList.add("level-locked");
  else if (risk.level === "L0" || risk.level === "L3") ui.riskLevel.classList.add("level-alert");
  else ui.riskLevel.classList.add("level-good");

  ui.riskTitle.textContent = risk.title;
  ui.riskReason.textContent = risk.reason;

  ui.eventLog.innerHTML = state.events
    .map((event) => `<li><strong>${event.time} ${event.title}</strong>${event.detail}</li>`)
    .join("");
}

function updateApi(risk) {
  const usable = isCaptureUsable();
  const payload = {
    worker_id: "local-demo-worker",
    station_id: "demo-self-validating-camera",
    timestamp: new Date().toISOString(),
    certificate_id: state.certId,
    capture_quality: {
      usable,
      trust_score: Number(state.trust.toFixed(2)),
      sqi: Number(state.metrics.sqi.toFixed(2)),
      face_roi_stability: Number(state.metrics.roi.toFixed(2)),
      lighting_stability: Number(state.metrics.lighting.toFixed(2)),
      motion_suppression: Number(state.metrics.motion.toFixed(2)),
      color_response: Number(state.metrics.color.toFixed(2)),
    },
    signals: usable
      ? {
          hr_bpm: Number(state.metrics.hr.toFixed(1)),
          baseline_hr_bpm: state.baselineHr,
          hrv_status: state.riskMode ? "deviating_from_shift_baseline" : "within_shift_baseline",
          window_seconds: 20,
        }
      : null,
    risk: {
      level: risk.level,
      reason: risk.reason,
      confidence: Number(risk.confidence.toFixed(2)),
      recommended_action: risk.action,
    },
  };

  ui.apiOutput.textContent = JSON.stringify(payload, null, 2);
}

function setPhaseFromRisk(risk) {
  if (state.calibrating) return;
  if (!state.calibrated) {
    ui.phaseLabel.textContent = "LOCKED";
    setPill(ui.trustState, "指标锁定", "locked");
  } else if (!isCaptureUsable()) {
    ui.phaseLabel.textContent = "NO SIGNAL";
    setPill(ui.trustState, "不可判定", "alert");
  } else if (risk.level === "L2" || risk.level === "L3") {
    ui.phaseLabel.textContent = "EVIDENCE";
    setPill(ui.trustState, "输出安全证据", "valid");
  } else {
    ui.phaseLabel.textContent = "MONITOR";
    setPill(ui.trustState, "可信监测中", "valid");
  }
}

function render() {
  const width = ui.canvas.width;
  const height = ui.canvas.height;
  const time = (performance.now() - state.startedAt) / 1000;

  if (state.useCamera && ui.video.readyState >= 2) {
    drawCamera(width, height);
  } else {
    drawSimulator(width, height, time);
  }

  sampleFrameMetrics(width, height);
  const risk = updateRisk();
  updateTrust();
  updateMetrics();
  updateCertificate();
  updateRiskBoard(risk);
  updateApi(risk);
  setPhaseFromRisk(risk);
  drawOverlays(width, height);

  requestAnimationFrame(render);
}

function resetDemo() {
  state.calibrated = false;
  state.calibrating = false;
  state.certId = null;
  state.baselineHr = null;
  state.riskMode = false;
  state.badLight = false;
  state.occlusion = false;
  state.lastRiskLevel = "LOCK";
  state.metrics = {
    roi: 0.74,
    lighting: 0.72,
    motion: 0.82,
    color: 0.5,
    sqi: 0.22,
    hr: null,
  };
  resetSteps();
  ui.lightBtn.classList.remove("active");
  ui.occlusionBtn.classList.remove("active");
  ui.riskBtn.classList.remove("active");
  setPill(ui.trustState, "指标锁定", "locked");
  ui.stageHint.textContent = state.cameraReady ? "已重置。运行可信校验后开放指标。" : "启动摄像头，或直接进入模拟器。";
  addEvent("系统重置", "可信证书和风险状态已清空。");
}

ui.startCameraBtn.addEventListener("click", startCamera);
ui.calibrateBtn.addEventListener("click", runCalibration);
ui.lightBtn.addEventListener("click", () => {
  state.badLight = !state.badLight;
  ui.lightBtn.classList.toggle("active", state.badLight);
  addEvent(state.badLight ? "光照突变" : "光照恢复", state.badLight ? "质量门控应进入不可判定。" : "采集环境恢复稳定。");
});
ui.occlusionBtn.addEventListener("click", () => {
  state.occlusion = !state.occlusion;
  ui.occlusionBtn.classList.toggle("active", state.occlusion);
  addEvent(state.occlusion ? "面部遮挡" : "遮挡解除", state.occlusion ? "ROI 不足，系统拒绝输出。" : "ROI 恢复。");
});
ui.riskBtn.addEventListener("click", () => {
  if (!state.calibrated) {
    addEvent("风险模拟被拒绝", "尚未建立可信证书，不能输出风险趋势。");
    return;
  }
  state.riskMode = !state.riskMode;
  ui.riskBtn.classList.toggle("active", state.riskMode);
  addEvent(state.riskMode ? "风险趋势启动" : "风险趋势关闭", state.riskMode ? "模拟连续窗口偏离当班基线。" : "恢复正常观察。");
});
ui.resetBtn.addEventListener("click", resetDemo);

setPill(ui.cameraMode, "未启动", "muted");
setPill(ui.trustState, "指标锁定", "locked");
addEvent("Demo 就绪", "启动相机或直接运行可信校验。");
render();
