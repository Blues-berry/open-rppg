const ui = {
  serviceState: document.getElementById("serviceState"),
  bpmText: document.getElementById("bpmText"),
  sqiText: document.getElementById("sqiText"),
  statusText: document.getElementById("statusText"),
  frameText: document.getElementById("frameText"),
  hintText: document.getElementById("hintText"),
  startBtn: document.getElementById("startBtn"),
  stopBtn: document.getElementById("stopBtn"),
};
const STATUS_REFRESH_MS = 250;
const HEALTH_REFRESH_MS = 1000;

let serviceOnline = false;
let lastHealthAt = 0;

function setText(element, value) {
  if (element) element.textContent = value == null ? "" : String(value);
}

function statusLabel(status, reason) {
  const map = {
    stable: "输出中",
    preview: "低置信预览",
    low_sqi: "SQI 偏低",
    no_face: "等待人脸",
    warming: reason === "no_recent_input" ? "等待视频帧" : "建立窗口",
    failed: "失败",
    stopped: "已停止",
    waiting: "待机",
  };
  return map[status] || status || "待机";
}

function friendlyError(message) {
  const text = String(message || "");
  if (text.includes("Receiving end does not exist") || text.includes("后台服务没有响应")) {
    return "扩展后台未响应：请在 chrome://extensions 重新加载本插件后重试。";
  }
  if (text.includes("Failed to fetch")) {
    return "本地服务未连接：请先运行 python demo/browser_video_heart_server.py。";
  }
  if (text.includes("Cannot access contents") || text.includes("无法注入")) {
    return "当前页面不允许注入检测脚本，请换到普通网页视频页后重试。";
  }
  return text || "未知错误。";
}

async function activeTab() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  return tab;
}

async function sendBackground(message) {
  return new Promise((resolve) => {
    chrome.runtime.sendMessage(message, (response) => {
      const error = chrome.runtime.lastError;
      if (error) {
        resolve({ ok: false, error: friendlyError(error.message) });
        return;
      }
      resolve(response || { ok: false, error: "后台服务没有响应。" });
    });
  });
}

function renderStatus(status) {
  const result = status?.result || {};
  const bpm = Number.isFinite(result.bpm) ? result.bpm : Number.isFinite(result.hr) ? Math.round(result.hr) : null;
  setText(ui.bpmText, bpm == null ? "--" : `${bpm}`);
  setText(ui.sqiText, Number.isFinite(result.SQI) ? result.SQI.toFixed(2) : "--");
  setText(ui.statusText, statusLabel(result.status || status?.state, result.reason));
  setText(ui.frameText, status?.frame_count || 0);
  setText(ui.hintText, status?.error || "检测普通网页视频时，请保持视频播放且人物脸部清晰。");
}

async function refresh(forceHealth = false) {
  const now = Date.now();
  if (forceHealth || now - lastHealthAt >= HEALTH_REFRESH_MS) {
    lastHealthAt = now;
    const health = await sendBackground({ type: "health" });
    serviceOnline = Boolean(health?.ok);
    if (!serviceOnline) {
      setText(ui.serviceState, "服务离线");
      setText(ui.hintText, friendlyError(health?.error) || "请先运行 python demo/browser_video_heart_server.py。");
      return;
    }
    setText(ui.serviceState, "服务在线");
  }
  if (!serviceOnline) return;
  const status = await sendBackground({ type: "status" });
  if (status?.ok) renderStatus(status);
  else setText(ui.hintText, friendlyError(status?.error) || "检测状态不可用。");
}

ui.startBtn.addEventListener("click", async () => {
  const tab = await activeTab();
  ui.startBtn.disabled = true;
  setText(ui.hintText, "正在选取网页视频并启动检测。");
  try {
    const response = await sendBackground({ type: "start", tabId: tab.id });
    if (!response?.ok) throw new Error(friendlyError(response?.error || "启动失败"));
    renderStatus(response.status);
    await refresh(true);
  } catch (error) {
    setText(ui.hintText, friendlyError(error.message));
  } finally {
    ui.startBtn.disabled = false;
  }
});

ui.stopBtn.addEventListener("click", async () => {
  await sendBackground({ type: "stop" });
  await refresh();
});

window.setInterval(refresh, STATUS_REFRESH_MS);
refresh(true);
