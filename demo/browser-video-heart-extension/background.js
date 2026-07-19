const SERVICE_BASE = "http://127.0.0.1:8030/api/browser-video";
const TARGET_CAPTURE_FPS = 30;

let activeSession = null;

async function apiJson(path, options = {}) {
  const response = await fetch(`${SERVICE_BASE}${path}`, {
    cache: "no-store",
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `${response.status} ${response.statusText}`);
  return data;
}

async function ensureOffscreenDocument() {
  if (!chrome.offscreen?.createDocument) {
    throw new Error("当前 Chrome 不支持 offscreen capture，请升级浏览器后重试。");
  }
  if (chrome.offscreen.hasDocument && await chrome.offscreen.hasDocument()) return;
  await chrome.offscreen.createDocument({
    url: "offscreen.html",
    reasons: ["USER_MEDIA"],
    justification: "Capture the selected tab video and forward cropped frames to the local rPPG service.",
  });
}

function isReceivingEndError(error) {
  return String(error?.message || "").includes("Receiving end does not exist");
}

async function injectContentScript(tabId) {
  try {
    await chrome.scripting.executeScript({
      target: { tabId },
      files: ["content.js"],
    });
  } catch (error) {
    throw new Error(`无法注入视频检测脚本：${error.message}`);
  }
}

async function sendTabMessage(tabId, message) {
  try {
    return await chrome.tabs.sendMessage(tabId, message);
  } catch (error) {
    if (!isReceivingEndError(error)) throw error;
    await injectContentScript(tabId);
    try {
      return await chrome.tabs.sendMessage(tabId, message);
    } catch (retryError) {
      throw new Error(`页面视频检测脚本未响应：${retryError.message}`);
    }
  }
}

async function activeStatus() {
  if (!activeSession?.session_id) return { ok: true, state: "idle", frame_count: 0, result: { status: "waiting" } };
  return apiJson(`/session/${activeSession.session_id}/status`);
}

async function startCapture(tabId) {
  const rectResponse = await sendTabMessage(tabId, { type: "video-rect" });
  if (!rectResponse?.ok) throw new Error(rectResponse?.error || "没有找到可用视频。");

  await stopCapture();
  const session = await apiJson("/session", { method: "POST", body: "{}" });
  try {
    await ensureOffscreenDocument();
    const streamId = await chrome.tabCapture.getMediaStreamId({ targetTabId: tabId });
    activeSession = {
      tabId,
      session_id: session.session_id,
      video: rectResponse.video,
    };
    const offscreenResponse = await chrome.runtime.sendMessage({
      type: "offscreen-start",
      streamId,
      sessionId: session.session_id,
      serviceBase: SERVICE_BASE,
      video: rectResponse.video,
      targetFps: TARGET_CAPTURE_FPS,
    });
    if (!offscreenResponse?.ok) throw new Error(offscreenResponse?.error || "offscreen capture failed");
    await sendTabMessage(tabId, { type: "browser-heart-started", status: session }).catch(() => {});
    return { ok: true, status: session };
  } catch (error) {
    activeSession = null;
    await apiJson(`/session/${session.session_id}/stop`, { method: "POST", body: "{}" }).catch(() => {});
    throw error;
  }
}

async function stopCapture() {
  const stoppedSession = activeSession;
  await chrome.runtime.sendMessage({ type: "offscreen-stop" }).catch(() => {});
  if (activeSession?.session_id) {
    await apiJson(`/session/${activeSession.session_id}/stop`, { method: "POST", body: "{}" }).catch(() => {});
  }
  activeSession = null;
  if (stoppedSession?.tabId) {
    await chrome.tabs.sendMessage(stoppedSession.tabId, { type: "browser-heart-stopped" }).catch(() => {});
  }
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  (async () => {
    if (message?.type === "health") {
      sendResponse(await apiJson("/health"));
    } else if (message?.type === "start") {
      sendResponse(await startCapture(message.tabId));
    } else if (message?.type === "stop") {
      await stopCapture();
      sendResponse({ ok: true });
    } else if (message?.type === "status") {
      sendResponse(await activeStatus());
    } else {
      sendResponse({ ok: false, error: "unknown extension message" });
    }
  })().catch((error) => sendResponse({ ok: false, error: error.message }));
  return true;
});
