import {
  AGENT_API,
  CAPTURE_API,
  DEFAULT_HR_WINDOW_SECONDS,
  FLOATING_HEART_POSITION_KEY,
  HIGHLIGHT_API,
  HISTORY_SECONDS,
  LAST_BPM_HOLD_MS,
  MODEL_API,
  OVERLAY_API,
  OUTPUT_SQI_THRESHOLD,
  POLL_INTERVAL_MS,
  PREVIEW_SQI_THRESHOLD,
  VIDEO_API,
  VIDEO_POLL_INTERVAL_MS,
} from "./config.js?v=20260719-live-v4";
import { apiJson } from "./api.js?v=20260719-live-v4";
import {
  checkedValue,
  hideBanner,
  numberValue,
  on,
  setChecked,
  setClassName,
  setDatasetFlag,
  setDisabled,
  setHidden,
  setMeter,
  setStatePill,
  setStyleProp,
  setText,
  setValue,
  showBanner,
  toggleClass,
  missingUiIds,
  ui,
  validateDom,
} from "./dom.js?v=20260719-live-v4";
import {
  basename,
  clamp,
  formatBpm,
  formatCount,
  formatElapsed,
  formatFps,
  formatMs,
  formatOutputStatus,
  formatPercent,
  formatReason,
} from "./format.js?v=20260719-live-v4";

const waveCtx = ui.waveCanvas?.getContext("2d") || null;

const state = {
  overlay: null,
  video: null,
  agent: null,
  history: [],
  lastPollAt: 0,
  lastVideoPollAt: 0,
  settingsPushTimer: null,
  lightBackendReady: false,
  lightPreviewAvailable: true,
  pendingOverlaySettings: null,
  localSettingsUntil: 0,
  settingsWriteSeq: 0,
  lastOutputBpm: null,
  lastOutputBpmAt: 0,
  heldOutputBpm: null,
  heldOutputIsFresh: false,
  heartbeatTimer: null,
  heartbeatBpm: null,
  highlightExportingId: null,
};


function hrWindowSeconds(model = {}) {
  return Number.isFinite(model.hr_window_seconds) ? model.hr_window_seconds : DEFAULT_HR_WINDOW_SECONDS;
}

function canHoldPreviousBpm(capture, output) {
  return capture.state === "running" && output.status === "warming" && output.reason === "no_recent_input";
}

function displayBpmFromOutput(capture, output) {
  const bpm = Number.isFinite(output.bpm) ? output.bpm : null;
  if (bpm != null) {
    state.lastOutputBpm = bpm;
    state.lastOutputBpmAt = Date.now();
    state.heldOutputBpm = bpm;
    state.heldOutputIsFresh = true;
    return bpm;
  }

  const canHold = canHoldPreviousBpm(capture, output);
  const lastAgeMs = Date.now() - state.lastOutputBpmAt;
  if (canHold && Number.isFinite(state.lastOutputBpm) && lastAgeMs <= LAST_BPM_HOLD_MS) {
    state.heldOutputBpm = state.lastOutputBpm;
    state.heldOutputIsFresh = false;
    return state.lastOutputBpm;
  }

  state.heldOutputBpm = null;
  state.heldOutputIsFresh = false;
  return null;
}

async function startCapture() {
  setStatePill("STARTING", "warn");
  setText(ui.captureHint, "正在启动后端摄像头采集和 Open-rppg 模型。");
  try {
    await apiJson(`${CAPTURE_API}/start`, {
      method: "POST",
      body: JSON.stringify({ device_index: 0, width: 1280, height: 720, fps: 30 }),
    });
    await pollState(true);
  } catch (error) {
    setStatePill("ERROR", "bad");
    setText(ui.captureHint, `启动失败：${error.message}`);
  }
}

async function stopCapture() {
  setStatePill("STOPPING", "warn");
  setText(ui.captureHint, "正在停止后端采集并清空直播输出。");
  try {
    await apiJson(`${CAPTURE_API}/stop`, { method: "POST", body: "{}" });
    await pollState(true);
  } catch (error) {
    setStatePill("ERROR", "bad");
    setText(ui.captureHint, `停止失败：${error.message}`);
  }
}

async function resetModel() {
  try {
    await apiJson(`${MODEL_API}/reset`, { method: "POST", body: "{}" });
    state.history = [];
    await pollState(true);
  } catch (error) {
    setText(ui.captureHint, `重置失败：${error.message}`);
  }
}

async function analyzeVideo() {
  const file = ui.videoFileInput?.files?.[0];
  if (!file) {
    setText(ui.videoAdvice, "请先选择一个视频文件。");
    return;
  }

  setDisabled(ui.analyzeVideoBtn, true);
  setText(ui.videoStatusText, "上传中");
  setText(ui.videoAdvice, "正在上传视频并准备离线分析。");
  setMeter(ui.videoProgressMeter, 0);
  setText(ui.videoProgressText, "0%");

  try {
    const query = new URLSearchParams({ name: file.name });
    const response = await fetch(`${VIDEO_API}/analyze?${query}`, {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream" },
      body: file,
    });
    const status = await response.json();
    if (!response.ok) {
      throw new Error(status.error || `${response.status} ${response.statusText}`);
    }
    renderVideo(status);
  } catch (error) {
    setText(ui.videoStatusText, "失败");
    setText(ui.videoAdvice, `上传失败：${error.message}`);
    setDisabled(ui.analyzeVideoBtn, false);
  }
}

async function resetVideoAnalysis() {
  try {
    const status = await apiJson(`${VIDEO_API}/reset`, { method: "POST", body: "{}" });
    renderVideo(status);
  } catch (error) {
    setText(ui.videoAdvice, `暂时不能清空：${error.message}`);
  }
}

async function toggleRecording() {
  const enabled = checkedValue(ui.recordingToggle, false);
  setDisabled(ui.recordingToggle, true);
  setText(ui.recordingStatusText, enabled ? "正在开启" : "正在关闭");
  setText(ui.highlightExportText, enabled ? "等待录制文件" : "录制关闭中");
  setText(
    ui.highlightEmptyText,
    enabled ? "本地保存请求已发送，收到下一帧后开始写入视频。" : "正在关闭本地保存。",
  );
  try {
    const highlights = await apiJson(`${HIGHLIGHT_API}/recording`, {
      method: "POST",
      body: JSON.stringify({ enabled }),
    });
    renderHighlights(highlights);
    await pollState(true);
  } catch (error) {
    setText(ui.highlightEmptyText, `录制状态更新失败：${error.message}`);
    setChecked(ui.recordingToggle, !enabled);
  } finally {
    setDisabled(ui.recordingToggle, false);
  }
}

async function exportHighlight(highlightId) {
  if (!highlightId) return;
  state.highlightExportingId = highlightId;
  renderHighlights(state.overlay?.highlights || {});
  try {
    const result = await apiJson(`${HIGHLIGHT_API}/export`, {
      method: "POST",
      body: JSON.stringify({ highlight_id: highlightId }),
    });
    await pollState(true);
    if (result.download_url) {
      window.location.href = result.download_url;
    }
  } catch (error) {
    setText(ui.highlightExportText, `导出失败：${error.message}`);
    await pollState(true);
  } finally {
    state.highlightExportingId = null;
    renderHighlights(state.overlay?.highlights || {});
  }
}

async function sendAgentMessage() {
  const text = ui.agentInput?.value?.trim() || "";
  if (!text) {
    setText(ui.agentErrorText, "请输入一条消息。");
    return;
  }

  setDisabled(ui.sendAgentBtn, true);
  setText(ui.agentErrorText, "正在生成回复。");
  try {
    const status = await apiJson(`${AGENT_API}/message`, {
      method: "POST",
      body: JSON.stringify({ text }),
    });
    if (ui.agentInput) ui.agentInput.value = "";
    renderAgent(status);
    await pollState(true);
  } catch (error) {
    setText(ui.agentErrorText, `发送失败：${error.message}`);
    await pollState(true);
  }
}

async function resetAgent() {
  try {
    const status = await apiJson(`${AGENT_API}/reset`, { method: "POST", body: "{}" });
    renderAgent(status);
    await pollState(true);
  } catch (error) {
    setText(ui.agentErrorText, `重置失败：${error.message}`);
  }
}

async function enableAgentApi() {
  const payload = {
    protocol: ui.agentProtocol?.value || "anthropic",
    base_url: ui.agentBaseUrl?.value?.trim() || "",
    model: ui.agentModel?.value?.trim() || "",
    api_key: ui.agentApiKey?.value?.trim() || "",
  };
  setDisabled(ui.enableAgentBtn, true);
  setText(ui.agentErrorText, "正在保存设置并开启外部 API。 ");
  try {
    const status = await apiJson(`${AGENT_API}/enable`, { method: "POST", body: JSON.stringify(payload) });
    if (ui.agentApiKey) ui.agentApiKey.value = "";
    if (ui.agentSettings) ui.agentSettings.open = false;
    renderAgent(status);
    await pollState(true);
  } catch (error) {
    setText(ui.agentErrorText, `开启失败：${error.message}`);
  } finally {
    setDisabled(ui.enableAgentBtn, false);
  }
}

async function disableAgentApi() {
  setDisabled(ui.disableAgentBtn, true);
  try {
    const status = await apiJson(`${AGENT_API}/disable`, { method: "POST", body: "{}" });
    renderAgent(status);
    await pollState(true);
  } catch (error) {
    setText(ui.agentErrorText, `关闭失败：${error.message}`);
  } finally {
    setDisabled(ui.disableAgentBtn, false);
  }
}

async function pollVideoStatus(force = false) {
  const now = performance.now();
  if (!force && now - state.lastVideoPollAt < VIDEO_POLL_INTERVAL_MS) return;
  state.lastVideoPollAt = now;

  try {
    const status = await apiJson(`${VIDEO_API}/status`);
    renderVideo(status);
  } catch (error) {
    setText(ui.videoAdvice, `视频分析状态不可用：${error.message}`);
  }
}

function readOverlaySettingsFromControls() {
  return {
    pulse: checkedValue(ui.pulseToggle, true),
    light_enabled: checkedValue(ui.lightToggle, false),
    brightness: numberValue(ui.brightnessInput, 72),
    temperature: numberValue(ui.temperatureInput, 4800),
    light_x: numberValue(ui.lightXInput, 50),
    light_y: numberValue(ui.lightYInput, 38),
    light_z: numberValue(ui.lightZInput, 45),
    light_range: numberValue(ui.lightRangeInput, 58),
    light_angle_enabled: checkedValue(ui.angleToggle, false),
    light_angle: numberValue(ui.lightAngleInput, 0),
  };
}

function applyOverlaySettingsToControls(settings = {}) {
  if (typeof settings.pulse === "boolean") setChecked(ui.pulseToggle, settings.pulse);
  if (typeof settings.light_enabled === "boolean") setChecked(ui.lightToggle, settings.light_enabled);
  if (Number.isFinite(settings.brightness)) setValue(ui.brightnessInput, settings.brightness);
  if (Number.isFinite(settings.temperature)) setValue(ui.temperatureInput, settings.temperature);
  if (Number.isFinite(settings.light_x)) setValue(ui.lightXInput, settings.light_x);
  if (Number.isFinite(settings.light_y)) setValue(ui.lightYInput, settings.light_y);
  if (Number.isFinite(settings.light_z)) setValue(ui.lightZInput, settings.light_z);
  if (Number.isFinite(settings.light_range)) setValue(ui.lightRangeInput, settings.light_range);
  if (typeof settings.light_angle_enabled === "boolean") setChecked(ui.angleToggle, settings.light_angle_enabled);
  if (Number.isFinite(settings.light_angle)) setValue(ui.lightAngleInput, settings.light_angle);
}

async function pushOverlaySettings() {
  updateLightValueText();
  const writeSeq = state.settingsWriteSeq;
  const desiredSettings = { ...(state.pendingOverlaySettings || readOverlaySettingsFromControls()) };
  try {
    const settings = await apiJson(`${OVERLAY_API}/settings`, {
      method: "POST",
      body: JSON.stringify(desiredSettings),
    });
    state.lightBackendReady = hasLightBackend(settings);
    if (writeSeq === state.settingsWriteSeq) {
      state.pendingOverlaySettings = null;
      state.localSettingsUntil = performance.now() + 250;
      applyOverlaySettingsToControls(settings);
      updateLightValueText();
    }
    updateLightBackendWarning(settings);
  } catch (error) {
    setText(ui.captureHint, `Overlay 设置未同步：${error.message}`);
  }
}

function scheduleOverlaySettings({ immediate = false } = {}) {
  updateLightValueText();
  state.pendingOverlaySettings = readOverlaySettingsFromControls();
  state.localSettingsUntil = performance.now() + 1200;
  state.settingsWriteSeq += 1;
  window.clearTimeout(state.settingsPushTimer);
  if (immediate) {
    pushOverlaySettings();
  } else {
    state.settingsPushTimer = window.setTimeout(pushOverlaySettings, 70);
  }
}

async function copyUrl(path, button) {
  const url = new URL(path, window.location.href).href;
  await navigator.clipboard.writeText(url);
  const previous = button?.textContent || "已复制";
  setText(button, "已复制");
  window.setTimeout(() => setText(button, previous), 1200);
}

async function pollState(force = false) {
  const now = performance.now();
  if (!force && now - state.lastPollAt < POLL_INTERVAL_MS) return;
  state.lastPollAt = now;

  let snapshot;
  try {
    snapshot = await apiJson(`${OVERLAY_API}/state`);
  } catch (error) {
    renderOffline(error);
    return;
  }

  try {
    state.overlay = normalizeSnapshot(snapshot);
    updateHistory(state.overlay);
    renderDashboard(state.overlay);
    if (Array.isArray(missingUiIds) && missingUiIds.length === 0) {
      hideBanner(ui.frontendWarning);
    }
  } catch (error) {
    renderFrontendError(error);
  }
}

function objectOrEmpty(value) {
  return value && typeof value === "object" && !Array.isArray(value) ? value : {};
}

function arrayOrEmpty(value) {
  return Array.isArray(value) ? value : [];
}

function normalizeSnapshot(snapshot) {
  const root = objectOrEmpty(snapshot);
  const model = objectOrEmpty(root.model);
  const waveform = objectOrEmpty(model.waveform);
  const agent = objectOrEmpty(root.agent);
  const highlights = objectOrEmpty(root.highlights);
  return {
    ...root,
    capture: objectOrEmpty(root.capture),
    model: { ...model, waveform: { ...waveform, bvp: arrayOrEmpty(waveform.bvp), ts: arrayOrEmpty(waveform.ts) } },
    output: objectOrEmpty(root.output),
    settings: objectOrEmpty(root.settings),
    agent: { ...agent, history: arrayOrEmpty(agent.history), latest: objectOrEmpty(agent.latest) },
    highlights: { ...highlights, items: arrayOrEmpty(highlights.items), recording: objectOrEmpty(highlights.recording), export: objectOrEmpty(highlights.export) },
  };
}

function renderOffline(error) {
  state.heldOutputBpm = null;
  state.heldOutputIsFresh = false;
  stopHeartbeatTimer();
  setStatePill("OFFLINE", "bad");
  setText(ui.heartTitle, "服务未连接");
  setText(ui.heartDescription, `无法连接本地模型服务：${error.message}`);
  setText(ui.captureHint, "请先运行 model_server.py。");
  renderAgent({ configured: false, error: "服务离线", history: [], latest: {} });
  renderHighlights({ recording: { enabled: checkedValue(ui.recordingToggle, false), state: "idle" }, items: [] });
  hideBanner(ui.backendWarning);
}

function renderFrontendError(error) {
  const message = `前端渲染错误：${error.message}`;
  setStatePill("FRONTEND ERR", "bad");
  setText(ui.heartTitle, "前端渲染错误");
  setText(ui.heartDescription, message);
  showBanner(ui.frontendWarning, message, "bad");
  console.error(error);
}

function updateHistory(snapshot) {
  const model = snapshot.model || {};
  const output = snapshot.output || {};
  const now = Date.now();
  if (!Array.isArray(state.history)) state.history = [];
  state.history.push({
    t: now,
    bpm: Number.isFinite(output.bpm) ? output.bpm : Number.isFinite(model.hr) ? model.hr : null,
    sqi: Number.isFinite(model.SQI) ? model.SQI : 0,
    status: output.status,
  });
  const cutoff = now - HISTORY_SECONDS * 1000;
  state.history = state.history.filter((item) => item.t >= cutoff);
}

function renderVideo(status) {
  state.video = status;
  const progress = Number.isFinite(status.progress) ? Math.round(status.progress * 100) : 0;
  const result = status.result || {};
  const hr = Number.isFinite(result.hr) ? Math.round(result.hr) : null;
  const sqi = Number.isFinite(result.SQI) ? result.SQI : null;
  const busy = ["saving", "queued", "processing"].includes(status.state);

  setMeter(ui.videoProgressMeter, progress / 100);
  setText(ui.videoProgressText, `${progress}%`);
  setText(ui.videoHrText, hr == null ? "--" : `${hr}`);
  setText(ui.videoSqiText, sqi == null ? "--" : sqi.toFixed(2));
  setText(
    ui.videoFramesText,
    status.frames_total
      ? `${status.frames_processed || 0}/${status.frames_total}`
      : status.frames_processed
        ? String(status.frames_processed)
        : "--",
  );
  setDisabled(ui.analyzeVideoBtn, busy);
  setDisabled(ui.resetVideoBtn, busy);

  if (status.state === "idle") {
    setText(ui.videoStatusText, "待上传");
    setText(ui.videoAdvice, "建议上传正脸、稳定补光、少压缩闪烁的视频；结果仅用于互动展示。");
  } else if (status.state === "saving") {
    setText(ui.videoStatusText, "上传中");
    setText(ui.videoAdvice, "正在保存视频文件。");
  } else if (status.state === "queued") {
    setText(ui.videoStatusText, "排队中");
    setText(ui.videoAdvice, "视频已上传，正在准备 FacePhys 模型分析。");
  } else if (status.state === "processing") {
    setText(ui.videoStatusText, "识别中");
    setText(ui.videoAdvice, "正在逐帧检测人脸并估计 BVP/HR，请保持页面打开查看进度。");
  } else if (status.state === "done") {
    setText(ui.videoStatusText, "完成");
    setText(
      ui.videoAdvice,
      hr == null
        ? "分析完成，但没有得到可靠 HR；请换一段正脸更稳定的视频。"
        : `分析完成：HR ${hr} BPM，SQI ${sqi == null ? "--" : sqi.toFixed(2)}。`,
    );
  } else if (status.state === "failed") {
    setText(ui.videoStatusText, "失败");
    setText(ui.videoAdvice, status.error || "视频分析失败。");
  }
}

function recordingStatusLabel(recording = {}) {
  if (recording.state === "recording") return "录制中";
  if (recording.state === "finalized") return "已保存";
  if (recording.state === "error") return "录制异常";
  if (recording.enabled) return "已勾选，等待采集";
  return "未开启";
}

function highlightStatusLabel(status) {
  if (status === "observing") return "观察中";
  if (status === "confirmed") return "已确认";
  return status || "待确认";
}

function highlightLevelLabel(level) {
  if (level === "high") return "强高能";
  if (level === "medium") return "中高能";
  return "高能";
}

function formatSignedBpm(value) {
  return Number.isFinite(value) ? `${value >= 0 ? "+" : ""}${value.toFixed(1)}` : "--";
}

function highlightExportLabel(item, recording = {}) {
  if (item.status === "observing") return "观察中";
  if (item.export_url) return "下载";
  if (state.highlightExportingId === item.id) return "导出中";
  if (item.exportable) return "导出片段";
  if (!recording.enabled && !recording.file) return "未保存视频";
  if (recording.state === "recording") return "停止后导出";
  return "不可导出";
}

function renderHighlights(highlights = {}) {
  const recording = highlights.recording || {};
  const items = Array.isArray(highlights.items) ? highlights.items : [];
  const exportState = highlights.export || {};

  setChecked(ui.recordingToggle, Boolean(recording.enabled));
  setText(ui.recordingStatusText, recordingStatusLabel(recording));
  setText(ui.recordingFileText, basename(recording.file));
  const itemCount = Array.isArray(items) ? items.length : 0;
  setText(ui.highlightCountText, String(itemCount));
  if (exportState.state === "exporting") {
    setText(ui.highlightExportText, "导出中");
  } else if (exportState.state === "failed") {
    setText(ui.highlightExportText, exportState.error || "导出失败");
  } else if (exportState.state === "done") {
    setText(ui.highlightExportText, "导出完成");
  } else {
    setText(ui.highlightExportText, recording.exportable ? "可导出" : "等待录制文件");
  }

  if (!ui.highlightList) return;
  ui.highlightList.replaceChildren();
  if (itemCount === 0) {
    setHidden(ui.highlightEmptyText, false);
    setText(
      ui.highlightEmptyText,
      recording.enabled
        ? "等待连续心率波动；达到稳定窗口后会自动出现高能时间段。"
        : "未开启本地保存时仍会判断时间段；需要导出视频片段时请先勾选本地保存。",
    );
    return;
  }

  setHidden(ui.highlightEmptyText, true);
  items.forEach((item, index) => {
    const row = document.createElement("article");
    row.className = `highlight-item ${item.status || ""} ${item.level || ""}`.trim();

    const main = document.createElement("div");
    main.className = "highlight-main";

    const title = document.createElement("strong");
    title.textContent = `#${index + 1} ${formatElapsed(item.start)} - ${formatElapsed(item.end)} · ${highlightStatusLabel(item.status)} · ${highlightLevelLabel(item.level)}`;

    const meta = document.createElement("span");
    const peakHr = Number.isFinite(item.peak_hr)
      ? Math.round(item.peak_hr)
      : Number.isFinite(item.max_hr)
        ? Math.round(item.max_hr)
        : "--";
    const prominence = Number.isFinite(item.prominence_hr)
      ? item.prominence_hr
      : Number.isFinite(item.delta_hr)
        ? item.delta_hr
        : null;
    const confidence = Number.isFinite(item.confidence) ? Math.round(clamp(item.confidence) * 100) : "--";
    const sqi = Number.isFinite(item.avg_sqi) ? item.avg_sqi.toFixed(2) : "--";
    meta.textContent = `峰值 ${peakHr} BPM · 较基线 ${formatSignedBpm(prominence)} BPM · 置信度 ${confidence}% · SQI ${sqi} · ${item.reason || "连续高位"}`;
    main.append(title, meta);

    const action = document.createElement(item.export_url ? "a" : "button");
    action.className = item.export_url ? "link-button highlight-action" : "highlight-action";
    action.textContent = highlightExportLabel(item, recording);
    if (item.export_url) {
      action.href = item.export_url;
      action.target = "_blank";
      action.rel = "noreferrer";
    } else {
      action.type = "button";
      action.disabled = item.status !== "confirmed" || !item.exportable || state.highlightExportingId === item.id;
      action.addEventListener("click", () => exportHighlight(item.id));
    }

    row.append(main, action);
    ui.highlightList.append(row);
  });
}

function renderAgent(agent = {}) {
  state.agent = agent;
  const apiMode = agent.mode === "api";
  const configured = apiMode && Boolean(agent.configured);
  const busy = Boolean(agent.busy);
  const latest = agent.latest || {};
  const history = Array.isArray(agent.history) ? agent.history : [];
  const modelLabel = agent.model || "Opus";

  setText(ui.agentModeText, apiMode ? "外部 API" : "本地自动播报");
  if (!apiMode) {
    setText(ui.agentStatusText, "本地规则播报中");
  } else if (!configured) {
    setText(ui.agentStatusText, "配置不完整");
  } else if (busy) {
    setText(ui.agentStatusText, `${modelLabel} 生成中`);
  } else if (agent.error) {
    setText(ui.agentStatusText, `${modelLabel} 异常`);
  } else {
    setText(ui.agentStatusText, `${modelLabel} 就绪`);
  }

  setText(ui.agentSubtitleText, latest.visible && latest.subtitle ? latest.subtitle : "暂无字幕");
  setText(ui.agentLogPathText, agent.log_path || "--");
  if (!apiMode) {
    setText(ui.agentErrorText, "本地模式不会调用外部 API；开启 API 后可使用自由对话。");
  } else {
    setText(ui.agentErrorText, agent.error || "");
  }
  if (ui.agentProtocol && document.activeElement !== ui.agentProtocol) setValue(ui.agentProtocol, agent.protocol || "anthropic");
  if (ui.agentBaseUrl && document.activeElement !== ui.agentBaseUrl) setValue(ui.agentBaseUrl, agent.base_url || "");
  if (ui.agentModel && document.activeElement !== ui.agentModel) setValue(ui.agentModel, agent.model || "bedrock-claude-haiku");
  setDisabled(ui.sendAgentBtn, busy || !configured);
  setDisabled(ui.resetAgentBtn, busy);
  setDisabled(ui.enableAgentBtn, busy);
  setDisabled(ui.disableAgentBtn, busy || !apiMode);

  if (!ui.agentMessages) return;
  ui.agentMessages.replaceChildren();
  const historyCount = Array.isArray(history) ? history.length : 0;
  if (historyCount === 0) {
    const empty = document.createElement("div");
    empty.className = "agent-message empty";
    empty.textContent = apiMode ? "等待对话或自动反馈。" : "本地自动播报已开启，等待心率或采集状态变化。";
    ui.agentMessages.append(empty);
    return;
  }

  history.slice(-12).forEach((item) => {
    const message = document.createElement("div");
    message.className = `agent-message ${item.role === "user" ? "user" : "assistant"}`;
    const meta = document.createElement("span");
    meta.textContent = item.role === "user" ? "你" : item.event === "auto" ? "Agent 自动" : "Agent";
    const body = document.createElement("p");
    body.textContent = item.text || item.subtitle || "";
    message.append(meta, body);
    ui.agentMessages.append(message);
  });
  ui.agentMessages.scrollTop = ui.agentMessages.scrollHeight;
}

function updateLightValueText() {
  setText(ui.brightnessValue, `${Math.round(numberValue(ui.brightnessInput, 72))}%`);
  setText(ui.temperatureValue, `${Math.round(numberValue(ui.temperatureInput, 4800))}K`);
  setText(ui.lightXValue, `${Math.round(numberValue(ui.lightXInput, 50))}%`);
  setText(ui.lightYValue, `${Math.round(numberValue(ui.lightYInput, 38))}%`);
  setText(ui.lightZValue, `${Math.round(numberValue(ui.lightZInput, 45))}%`);
  setText(ui.lightRangeValue, `${Math.round(numberValue(ui.lightRangeInput, 58))}%`);
  setText(ui.lightAngleValue, `${Math.round(numberValue(ui.lightAngleInput, 0))}°`);
  setDisabled(ui.lightAngleInput, !checkedValue(ui.angleToggle, false));
}

function hasLightBackend(settings = {}) {
  return [
    "light_enabled",
    "brightness",
    "temperature",
    "light_x",
    "light_y",
    "light_z",
    "light_range",
    "light_angle_enabled",
    "light_angle",
    "light_revision",
  ].every((key) => Object.prototype.hasOwnProperty.call(settings, key));
}

function updateLightBackendWarning(settings = {}) {
  const settingsReady = hasLightBackend(settings);
  state.lightBackendReady = settingsReady;
  if (!settingsReady) {
    showBanner(ui.backendWarning, "后端未重启：请重启 model_server.py 后刷新页面，补光 Z/角度和成对预览才会生效。", "warn");
    return true;
  }
  if (!state.lightPreviewAvailable) {
    showBanner(ui.backendWarning, "补光预览端点暂不可用：请确认后端提供 /api/capture/light-preview.mjpg。", "warn");
    return true;
  }
  hideBanner(ui.backendWarning);
  return false;
}

function attachStreamRetry(image) {
  if (!image) return;
  const source = image.dataset.src || image.getAttribute("src");
  image.dataset.src = source;
  image.addEventListener("error", () => {
    if (image.dataset.paused === "true") return;
    if (image === ui.pairedLightPreview) {
      state.lightPreviewAvailable = false;
      if (state.overlay) updateLightBackendWarning(state.overlay.settings || {});
    }
    window.setTimeout(() => {
      if (image.dataset.paused === "true") return;
      image.src = `${source}${source.includes("?") ? "&" : "?"}t=${Date.now()}`;
    }, 1000);
  });
  image.addEventListener("load", () => {
    if (image === ui.pairedLightPreview) {
      state.lightPreviewAvailable = true;
      if (state.overlay) updateLightBackendWarning(state.overlay.settings || {});
    }
  });
}

function setStreamPaused(image, paused) {
  if (!image) return;
  const source = image.dataset.src || image.getAttribute("src");
  image.dataset.src = source;
  if (paused) {
    image.dataset.paused = "true";
    image.removeAttribute("src");
    return;
  }
  delete image.dataset.paused;
  if (!image.getAttribute("src")) {
    image.src = `${source}${source.includes("?") ? "&" : "?"}t=${Date.now()}`;
  }
}

function renderLightPreviewVisibility(settings = {}) {
  const enabled = Boolean(settings.light_enabled);
  setHidden(ui.lightPreviewStream, !enabled);
  toggleClass(ui.pairedPreviewStage, "disabled", !enabled);
  setDatasetFlag(ui.pairedPreviewStage, "disabled", !enabled);
  setStreamPaused(ui.pairedLightPreview, !enabled);
  state.lightPreviewAvailable = !enabled ? true : state.lightPreviewAvailable;
  if (state.overlay) updateLightBackendWarning(state.overlay.settings || {});
}

function updateLightPositionFromPointer(event, options = {}) {
  if (!checkedValue(ui.lightToggle, false)) return false;
  const image = ui.pairedLightPreview;
  if (!image) return false;
  const rect = image.getBoundingClientRect();
  if (!rect.width || !rect.height) return false;
  const x = clamp((event.clientX - rect.left) / rect.width, 0, 1);
  const y = clamp((event.clientY - rect.top) / rect.height, 0, 1);
  setValue(ui.lightXInput, Math.round(x * 100));
  setValue(ui.lightYInput, Math.round(y * 100));
  scheduleOverlaySettings(options);
  return true;
}

function enablePairedLightDrag() {
  const stage = ui.pairedPreviewStage;
  if (!stage) return;
  let dragging = false;
  stage.addEventListener("pointerdown", (event) => {
    if (!updateLightPositionFromPointer(event)) return;
    dragging = true;
    event.preventDefault();
    stage.setPointerCapture(event.pointerId);
  });
  stage.addEventListener("pointermove", (event) => {
    if (dragging) {
      event.preventDefault();
      updateLightPositionFromPointer(event);
    }
  });
  stage.addEventListener("pointerup", (event) => {
    if (dragging) {
      updateLightPositionFromPointer(event, { immediate: true });
    }
    dragging = false;
    if (stage.hasPointerCapture(event.pointerId)) {
      stage.releasePointerCapture(event.pointerId);
    }
  });
  stage.addEventListener("pointercancel", () => {
    dragging = false;
  });
}

function floatingHeartBounds(x, y) {
  const widget = ui.floatingHeartWidget;
  if (!widget) return { x, y };
  const rect = widget.getBoundingClientRect();
  const padding = 8;
  const maxX = Math.max(padding, window.innerWidth - rect.width - padding);
  const maxY = Math.max(padding, window.innerHeight - rect.height - padding);
  return {
    x: Math.round(clamp(x, padding, maxX)),
    y: Math.round(clamp(y, padding, maxY)),
  };
}

function setFloatingHeartPosition(x, y, { persist = false } = {}) {
  const widget = ui.floatingHeartWidget;
  if (!widget) return;
  const position = floatingHeartBounds(x, y);
  widget.style.left = `${position.x}px`;
  widget.style.top = `${position.y}px`;
  widget.style.right = "auto";
  widget.style.bottom = "auto";
  if (persist) {
    window.localStorage.setItem(FLOATING_HEART_POSITION_KEY, JSON.stringify(position));
  }
}

function restoreFloatingHeartPosition() {
  const widget = ui.floatingHeartWidget;
  if (!widget) return;
  try {
    const saved = JSON.parse(window.localStorage.getItem(FLOATING_HEART_POSITION_KEY) || "null");
    if (Number.isFinite(saved?.x) && Number.isFinite(saved?.y)) {
      window.requestAnimationFrame(() => setFloatingHeartPosition(saved.x, saved.y));
    }
  } catch (error) {
    window.localStorage.removeItem(FLOATING_HEART_POSITION_KEY);
  }
}

function enableFloatingHeartDrag() {
  const widget = ui.floatingHeartWidget;
  if (!widget) return;
  let dragging = false;
  let dragStart = null;

  widget.addEventListener("pointerdown", (event) => {
    if (event.button != null && event.button !== 0) return;
    const rect = widget.getBoundingClientRect();
    dragging = true;
    dragStart = {
      pointerX: event.clientX,
      pointerY: event.clientY,
      widgetX: rect.left,
      widgetY: rect.top,
    };
    widget.classList.add("dragging");
    widget.setPointerCapture(event.pointerId);
    event.preventDefault();
  });

  widget.addEventListener("pointermove", (event) => {
    if (!dragging || !dragStart) return;
    event.preventDefault();
    setFloatingHeartPosition(
      dragStart.widgetX + event.clientX - dragStart.pointerX,
      dragStart.widgetY + event.clientY - dragStart.pointerY,
    );
  });

  function finishDrag(event) {
    if (!dragging) return;
    dragging = false;
    dragStart = null;
    widget.classList.remove("dragging");
    if (widget.hasPointerCapture(event.pointerId)) {
      widget.releasePointerCapture(event.pointerId);
    }
    const rect = widget.getBoundingClientRect();
    setFloatingHeartPosition(rect.left, rect.top, { persist: true });
  }

  widget.addEventListener("pointerup", finishDrag);
  widget.addEventListener("pointercancel", finishDrag);
  window.addEventListener("resize", () => {
    const rect = widget.getBoundingClientRect();
    setFloatingHeartPosition(rect.left, rect.top, { persist: true });
  });
  restoreFloatingHeartPosition();
}

function stopHeartbeatTimer() {
  window.clearTimeout(state.heartbeatTimer);
  state.heartbeatTimer = null;
  state.heartbeatBpm = null;
  toggleClass(ui.floatingHeartWidget, "beating", false);
  toggleClass(ui.floatingHeartWidget, "pulsing", false);
}

function triggerHeartbeatBeat() {
  if (!ui.floatingHeartWidget || !Number.isFinite(state.heartbeatBpm)) return;
  ui.floatingHeartWidget.classList.remove("beating");
  void ui.floatingHeartWidget.offsetWidth;
  ui.floatingHeartWidget.classList.add("beating");
  window.setTimeout(() => toggleClass(ui.floatingHeartWidget, "beating", false), 420);
  const interval = Math.max(430, Math.min(1200, 60000 / state.heartbeatBpm));
  state.heartbeatTimer = window.setTimeout(triggerHeartbeatBeat, interval);
}

function updateHeartbeatTimer(bpm, confidence, pulseEnabled) {
  const active = pulseEnabled && Number.isFinite(bpm) && confidence > 0.45;
  if (!active) {
    stopHeartbeatTimer();
    return;
  }
  if (state.heartbeatTimer && state.heartbeatBpm === bpm) return;
  window.clearTimeout(state.heartbeatTimer);
  state.heartbeatTimer = null;
  state.heartbeatBpm = bpm;
  triggerHeartbeatBeat();
}

function renderDashboard(snapshot) {
  const capture = snapshot.capture || {};
  const model = snapshot.model || {};
  const output = snapshot.output || {};
  const settings = snapshot.settings || {};
  const agent = snapshot.agent || {};
  const captureRunning = capture.state === "running";

  renderPart("心率状态", () => renderHeart(capture, model, output));
  renderPart("输出遥测", () => renderOutputTelemetry(model, output));
  renderPart("采集状态", () => renderStateCopy(capture, model, output));
  renderPart("链路性能", () => renderPerf(capture, model, output));
  renderPart("质量指标", () => renderQuality(capture, model));
  renderPart("互动 Agent", () => renderAgent(agent));
  renderPart("高光片段", () => renderHighlights(snapshot.highlights || {}));
  renderPart("补光控制", () => renderLightControls(capture, model, output, settings));
  renderPart("采集按钮", () => renderControls(captureRunning, capture.state));
  renderPart("波形", drawWave);
}

function renderPart(name, render) {
  try {
    render();
  } catch (error) {
    console.error(`渲染分区失败：${name}`, error);
    if (name === "互动 Agent") setText(ui.agentErrorText, `该分区暂不可用：${error.message}`);
    if (name === "波形") setText(ui.captureHint, `波形暂不可用：${error.message}`);
  }
}

function renderHeart(capture, model, output) {
  const captureRunning = capture.state === "running";
  const confidence = clamp(Number(output.confidence || 0));
  const confidencePct = Math.round(confidence * 100);
  const bpm = Number.isFinite(output.bpm) ? output.bpm : null;
  const displayBpm = displayBpmFromOutput(capture, output);
  const previewBpm = Number.isFinite(model.hr) ? Math.round(model.hr) : null;
  const windowSeconds = hrWindowSeconds(model);

  setText(ui.bpmValue, displayBpm == null ? "--" : String(displayBpm));
  setText(ui.confidenceText, `${confidencePct}%`);
  setText(ui.previewHrText, previewBpm == null ? "--" : formatBpm(previewBpm));
  setText(ui.windowText, `${Math.round(windowSeconds)}s`);
  setText(ui.trackerText, captureRunning ? "后端 BlazeFace" : "待机");
  setText(ui.modelText, model.ready ? model.model || "Open-rppg" : model.state || "loading");
  setText(ui.outputText, formatOutputStatus(output.status, bpm != null));

  const arc = confidencePct * 3.6;
  const beatBpm = displayBpm;
  const beatInterval = beatBpm ? Math.max(430, Math.min(1200, 60000 / beatBpm)) : 900;
  if (ui.heartOrb) {
    ui.heartOrb.style.background =
      `radial-gradient(circle at center, #15191d 58%, transparent 59%), conic-gradient(var(--rose) ${arc}deg, #2b3237 0deg)`;
  }
  setStyleProp(ui.heartOrb, "--beat-ms", `${beatInterval}ms`);
  setStyleProp(ui.heartOrb, "--heart-arc", `${arc}deg`);
  toggleClass(ui.heartOrb, "has-output", displayBpm != null);
  toggleClass(ui.heartOrb, "holding-output", displayBpm != null && !state.heldOutputIsFresh);
  setStyleProp(ui.floatingHeartWidget, "--beat-ms", `${beatInterval}ms`);
  setStyleProp(ui.floatingHeartWidget, "--heart-arc", `${arc}deg`);
  toggleClass(ui.floatingHeartWidget, "has-output", displayBpm != null);
  toggleClass(ui.floatingHeartWidget, "holding-output", displayBpm != null && !state.heldOutputIsFresh);
  updateHeartbeatTimer(displayBpm, confidence, checkedValue(ui.pulseToggle, true));
}

function renderOutputTelemetry(model, output) {
  const confidence = clamp(Number(output.confidence || 0));
  const confidencePct = Math.round(confidence * 100);
  const bpm = Number.isFinite(output.bpm) ? output.bpm : null;
  const displayBpm = Number.isFinite(state.heldOutputBpm) ? state.heldOutputBpm : bpm;
  const previewBpm = Number.isFinite(model.hr) ? model.hr : null;
  const sqi = Number.isFinite(model.SQI) ? model.SQI : null;
  const hasBpm = bpm != null;

  setText(
    ui.telemetryBpmText,
    displayBpm == null ? "--" : `${formatBpm(displayBpm)} BPM${state.heldOutputIsFresh ? "" : " 沿用"}`,
  );
  setText(ui.telemetryModelHrText, previewBpm == null ? "--" : `${formatBpm(previewBpm)} BPM`);
  setText(ui.telemetrySqiText, sqi == null ? "--" : sqi.toFixed(2));
  setText(ui.telemetryConfidenceText, `${confidencePct}%`);
  setText(ui.telemetryStatusText, formatOutputStatus(output.status, hasBpm));
  setText(ui.telemetryWindowText, `${Math.round(hrWindowSeconds(model))}s`);
}

function renderStateCopy(capture, model, output) {
  const previewBpm = Number.isFinite(model.hr) ? `${Math.round(model.hr)} BPM，` : "";
  if (capture.state === "camera_error") {
    setText(ui.heartTitle, "摄像头不可用");
    setText(ui.heartDescription, capture.error || "摄像头被占用、断开，或没有权限。");
    setText(ui.captureHint, "请确认 OBS 没有直接占用摄像头；第一阶段由 Python 服务独占摄像头。");
    setStatePill("CAMERA ERR", "bad");
    return;
  }

  if (capture.state !== "running") {
    setText(ui.heartTitle, "等待后端采集");
    setText(ui.heartDescription, "启动后端采集后，页面可关闭，服务仍会持续监测。");
    setText(ui.captureHint, "OBS 中添加 camera.html 作为摄像头源，overlay.html 作为心率 Overlay。");
    setStatePill("WAITING", "waiting");
    return;
  }

  setText(
    ui.captureHint,
    `后端采集运行中：device ${capture.device_index}，${capture.width}x${capture.height}，输入 ${formatFps(capture.input_fps)}。`,
  );

  if (!model.ready) {
    setText(ui.heartTitle, "模型加载中");
    setText(ui.heartDescription, "摄像头已持续采集，Open-rppg 权重加载完成后会自动开始推理。");
    setStatePill("LOADING", "warn");
  } else if (output.status === "stable") {
    setText(ui.heartTitle, "直播心率输出中");
    setText(ui.heartDescription, `SQI=${Number(model.SQI).toFixed(2)}，Overlay 正在输出 Open-rppg 心率。`);
    setStatePill("READY", "ready");
  } else if (output.status === "preview") {
    setText(ui.heartTitle, "低置信预览");
    setText(ui.heartDescription, `模型已返回 ${previewBpm}但 SQI=${Number(model.SQI).toFixed(2)}，暂不推送到直播 Overlay。`);
    setStatePill("PREVIEW", "warn");
  } else if (output.status === "no_face") {
    setText(ui.heartTitle, "等待人脸");
    setText(ui.heartDescription, "后端仍在持续采集，但没有稳定人脸时不会沿用旧 BPM。");
    setStatePill("NO FACE", "bad");
  } else if (output.status === "low_sqi") {
    setText(ui.heartTitle, "SQI 偏低，暂不输出");
    setText(ui.heartDescription, `模型 HR ${previewBpm || "--，"}当前 SQI=${Number(model.SQI || 0).toFixed(2)}，建议稳定补光、减少头动。`);
    setStatePill("LOW SQI", "bad");
  } else {
    setText(ui.heartTitle, "正在建立窗口");
    setText(ui.heartDescription, "后端正在持续积累 BVP/SQI 窗口，达标后 Overlay 自动输出。");
    setStatePill("MODEL", "warn");
  }
}

function renderPerf(capture, model, output) {
  const perf = model.perf || {};
  setText(ui.cameraFpsText, formatFps(capture.input_fps));
  setText(ui.detectMsText, formatMs(capture.read_ms));
  setText(ui.cropMsText, formatCount(capture.frames_read));
  setText(ui.requestMsText, formatReason(output.reason));
  setText(ui.modelInputFpsText, formatFps(model.input_fps));
  setText(ui.modelUpdateMsText, formatMs(perf.update_ms));
  setText(ui.modelMetricMsText, formatMs(perf.metric_ms));
  setText(ui.dropText, `${model.no_face_count || 0}/${capture.dropped_frames || 0}`);
}

function renderQuality(capture, model) {
  const captureScore = capture.state === "running" && Number.isFinite(capture.input_fps)
    ? clamp(capture.input_fps / Math.max(1, capture.target_fps || 30))
    : 0;
  const faceScore = model.has_face ? 1 : 0;
  const sqiScore = clamp(Number(model.SQI || 0));

  setMeter(ui.brightnessMeter, captureScore);
  setMeter(ui.motionMeter, faceScore);
  setMeter(ui.peakMeter, sqiScore);
  setText(ui.brightnessText, formatPercent(captureScore));
  setText(ui.motionText, formatPercent(faceScore));
  setText(ui.peakText, formatPercent(sqiScore));
}

function renderLightControls(capture, model, output, settings = {}) {
  const localEditing = performance.now() < state.localSettingsUntil && state.pendingOverlaySettings;
  const effectiveSettings = localEditing ? { ...settings, ...state.pendingOverlaySettings } : settings;
  applyOverlaySettingsToControls(effectiveSettings);
  updateLightValueText();
  renderLightPreviewVisibility(effectiveSettings);
  const hasWarning = updateLightBackendWarning(settings);
  if (!hasWarning) {
    renderLightAdvice(capture, model, output, effectiveSettings);
  }
}

function renderControls(captureRunning, captureState) {
  setDisabled(ui.stopBtn, !captureRunning && captureState !== "starting");
  setDisabled(ui.startBtn, captureRunning || captureState === "starting");
}

function renderLightAdvice(capture, model, output, settings = {}) {
  if (capture.state !== "running") {
    setText(ui.lightAdvice, "建议：先启动后端采集，再在 OBS 添加摄像头源和 Overlay。");
  } else if (!settings.light_enabled) {
    setText(ui.lightAdvice, "当前 OBS 输出使用原始图像；勾选应用补光后才显示补光预览和灯位拖动。");
  } else if (!model.has_face) {
    setText(ui.lightAdvice, "模拟补光已开启；让脸部进入画面中央，避免麦克风、手或头发遮挡。");
  } else if ((model.SQI || 0) < PREVIEW_SQI_THRESHOLD) {
    setText(ui.lightAdvice, "建议：提高并锁定补光亮度，减少画面自动曝光和快速头动。");
  } else if ((model.SQI || 0) < OUTPUT_SQI_THRESHOLD) {
    setText(ui.lightAdvice, "建议：当前已有低置信预览，继续保持正脸和稳定补光等待 SQI 提升。");
  } else if (output.status === "stable") {
    setText(ui.lightAdvice, "建议：保持当前补光，不要让灯光跟随音乐或弹幕快速闪烁。");
  } else {
    setText(ui.lightAdvice, "建议：保持柔和稳定补光，等待模型窗口稳定。");
  }
}

function drawWave() {
  if (!waveCtx || !ui.waveCanvas) return;
  const width = ui.waveCanvas.width;
  const height = ui.waveCanvas.height;
  waveCtx.clearRect(0, 0, width, height);
  waveCtx.fillStyle = "#11161a";
  waveCtx.fillRect(0, 0, width, height);
  waveCtx.strokeStyle = "rgba(255,255,255,0.08)";
  waveCtx.lineWidth = 1;
  for (let x = 0; x < width; x += 64) {
    waveCtx.beginPath();
    waveCtx.moveTo(x, 0);
    waveCtx.lineTo(x, height);
    waveCtx.stroke();
  }

  const centerY = height * 0.52;
  waveCtx.strokeStyle = "rgba(255,255,255,0.16)";
  waveCtx.lineWidth = 1;
  waveCtx.beginPath();
  waveCtx.moveTo(0, centerY);
  waveCtx.lineTo(width, centerY);
  waveCtx.stroke();

  const waveform = state.overlay?.model?.waveform || {};
  const rawSamples = Array.isArray(waveform.bvp) ? waveform.bvp : [];
  const rawTimestamps = Array.isArray(waveform.ts) ? waveform.ts : [];
  const sampleCount = Array.isArray(rawSamples) ? rawSamples.length : 0;
  const timestampCount = Array.isArray(rawTimestamps) ? rawTimestamps.length : 0;
  if (sampleCount !== timestampCount || sampleCount < 2) return;
  const points = rawSamples.map((value, index) => ({
    value: Number(value),
    ts: Number(rawTimestamps[index]),
  }));
  if (points.some((point) => !Number.isFinite(point.value) || !Number.isFinite(point.ts))) return;
  if (points.some((point, index) => index > 0 && point.ts <= points[index - 1].ts)) return;
  const pointCount = Array.isArray(points) ? points.length : 0;
  if (pointCount < 2) return;

  const mean = points.reduce((sum, point) => sum + point.value, 0) / pointCount;
  const centered = points.map((point) => ({ ...point, value: point.value - mean }));
  const sortedAbs = centered.map((point) => Math.abs(point.value)).sort((a, b) => a - b);
  const scaleIndex = Math.max(0, Math.floor(pointCount * 0.92) - 1);
  const scale = Math.max(sortedAbs[scaleIndex] || 0, 1e-6);
  const firstTs = centered[0].ts;
  const lastTs = centered[pointCount - 1].ts;
  const tsSpan = lastTs - firstTs;
  if (!(tsSpan > 0)) return;
  const amplitude = height * 0.38;

  waveCtx.strokeStyle = "#55d2b0";
  waveCtx.lineWidth = 3;
  waveCtx.shadowColor = "rgba(85, 210, 176, 0.34)";
  waveCtx.shadowBlur = 10;
  waveCtx.beginPath();
  centered.forEach((point, index) => {
    const x = ((point.ts - firstTs) / tsSpan) * width;
    const y = centerY - clamp(point.value / scale, -1, 1) * amplitude;
    if (index === 0) waveCtx.moveTo(x, y);
    else waveCtx.lineTo(x, y);
  });
  waveCtx.stroke();
  waveCtx.shadowBlur = 0;
}

function init() {
  validateDom();
  on(ui.startBtn, "click", startCapture);
  on(ui.stopBtn, "click", stopCapture);
  on(ui.resetBtn, "click", resetModel);
  on(ui.analyzeVideoBtn, "click", analyzeVideo);
  on(ui.resetVideoBtn, "click", resetVideoAnalysis);
  on(ui.recordingToggle, "change", toggleRecording);
  on(ui.sendAgentBtn, "click", sendAgentMessage);
  on(ui.resetAgentBtn, "click", resetAgent);
  on(ui.enableAgentBtn, "click", enableAgentApi);
  on(ui.disableAgentBtn, "click", disableAgentApi);
  on(ui.cameraSourceBtn, "click", () => copyUrl("camera.html", ui.cameraSourceBtn));
  on(ui.copyOverlayBtn, "click", () => copyUrl("overlay.html", ui.copyOverlayBtn));
  on(ui.agentInput, "keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendAgentMessage();
    }
  });

  [
    ui.lightToggle,
    ui.brightnessInput,
    ui.temperatureInput,
    ui.lightXInput,
    ui.lightYInput,
    ui.lightZInput,
    ui.lightRangeInput,
    ui.angleToggle,
    ui.lightAngleInput,
    ui.pulseToggle,
  ].forEach((input) => on(input, "input", scheduleOverlaySettings));

  setDisabled(ui.stopBtn, true);
  setDisabled(ui.resetVideoBtn, false);
  updateLightValueText();
  renderLightPreviewVisibility(readOverlaySettingsFromControls());
  attachStreamRetry(ui.currentOutputPreview);
  attachStreamRetry(ui.pairedLightPreview);
  enablePairedLightDrag();
  enableFloatingHeartDrag();
  pollState(true);
  pollVideoStatus(true);
  window.setInterval(() => pollState(false), POLL_INTERVAL_MS);
  window.setInterval(() => pollVideoStatus(false), VIDEO_POLL_INTERVAL_MS);
}

init();
