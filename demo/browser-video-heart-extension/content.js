if (!window.__openRppgVideoHeartContentScript) {
  window.__openRppgVideoHeartContentScript = true;

  const OVERLAY_HOST_ID = "__open_rppg_browser_video_heart_overlay";
  const OVERLAY_POSITION_KEY = "open-rppg-browser-video-heart-position";
  const STATUS_POLL_MS = 250;
  const LAST_HR_HOLD_MS = 8000;

  const overlayState = {
    host: null,
    shadow: null,
    statusTimer: null,
    heartbeatTimer: null,
    heartbeatHr: null,
    lastHr: null,
    lastHrAt: 0,
  };

  function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
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

  function visibleVideoInfo(video, index) {
    const rect = video.getBoundingClientRect();
    const area = Math.max(0, rect.width) * Math.max(0, rect.height);
    if (area < 120 * 90) return null;
    const style = window.getComputedStyle(video);
    if (style.visibility === "hidden" || style.display === "none" || Number(style.opacity) === 0) return null;
    return {
      index,
      area,
      rect: {
        x: rect.left,
        y: rect.top,
        width: rect.width,
        height: rect.height,
      },
      viewport: {
        width: window.innerWidth,
        height: window.innerHeight,
        devicePixelRatio: window.devicePixelRatio || 1,
      },
      paused: video.paused,
      currentTime: video.currentTime,
      duration: Number.isFinite(video.duration) ? video.duration : null,
      src: video.currentSrc || video.src || location.href,
    };
  }

  function bestVideo() {
    return Array.from(document.querySelectorAll("video"))
      .map(visibleVideoInfo)
      .filter(Boolean)
      .sort((a, b) => b.area - a.area)[0] || null;
  }

  function sendBackground(message) {
    return new Promise((resolve) => {
      chrome.runtime.sendMessage(message, (response) => {
        const error = chrome.runtime.lastError;
        if (error) {
          resolve({ ok: false, error: error.message });
          return;
        }
        resolve(response || { ok: false, error: "background unavailable" });
      });
    });
  }

  function overlayCss() {
    return `
      :host {
        all: initial;
        color-scheme: dark;
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      }

      .widget {
        box-sizing: border-box;
        width: 168px;
        min-height: 128px;
        padding: 12px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 16px;
        background: rgba(9, 12, 15, 0.82);
        color: #f6f4ed;
        box-shadow: 0 18px 48px rgba(0, 0, 0, 0.38);
        backdrop-filter: blur(14px);
        user-select: none;
        cursor: grab;
      }

      .widget.dragging {
        cursor: grabbing;
      }

      .top {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 8px;
        margin-bottom: 8px;
      }

      .title {
        color: #b8d8e5;
        font-size: 12px;
        font-weight: 850;
        letter-spacing: 0;
      }

      .close {
        display: grid;
        place-items: center;
        width: 24px;
        height: 24px;
        border: 1px solid rgba(255, 255, 255, 0.18);
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.06);
        color: #f6f4ed;
        font-size: 17px;
        line-height: 1;
        cursor: pointer;
      }

      .main {
        display: grid;
        grid-template-columns: 58px minmax(0, 1fr);
        gap: 10px;
        align-items: center;
      }

      .pulse {
        position: relative;
        display: grid;
        place-items: center;
        width: 58px;
        aspect-ratio: 1;
        border-radius: 50%;
        background:
          radial-gradient(circle at 48% 44%, rgba(255, 255, 255, 0.2) 0 16%, transparent 17%),
          radial-gradient(circle at center, rgba(255, 107, 131, 0.2) 0 54%, transparent 55%),
          conic-gradient(#ff6b83 var(--heart-arc, 0deg), rgba(255, 255, 255, 0.16) 0deg);
        box-shadow:
          inset 0 0 16px rgba(255, 255, 255, 0.12),
          0 12px 30px rgba(255, 57, 91, 0.26);
      }

      .ring {
        position: absolute;
        inset: 8px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 50%;
        opacity: 0;
      }

      .heart {
        position: relative;
        width: 24px;
        height: 24px;
        margin-top: 5px;
        border-radius: 6px 6px 3px 6px;
        background:
          radial-gradient(circle at 24% 20%, rgba(255, 255, 255, 0.95) 0 9%, transparent 10%),
          linear-gradient(135deg, #ff9aad 0%, #ff4f70 43%, #bc1636 100%);
        box-shadow:
          inset -5px -7px 11px rgba(101, 0, 21, 0.28),
          inset 4px 4px 9px rgba(255, 255, 255, 0.2),
          0 9px 22px rgba(255, 57, 91, 0.4);
        transform: rotate(-45deg) scale(1);
        transform-origin: 50% 64%;
      }

      .heart::before,
      .heart::after {
        position: absolute;
        content: "";
        width: 24px;
        height: 24px;
        border-radius: 50%;
        background: inherit;
        box-shadow: inherit;
      }

      .heart::before {
        top: -12px;
        left: 0;
      }

      .heart::after {
        top: 0;
        left: 12px;
      }

      .widget.beating .pulse {
        animation: pulse-once 420ms ease-out 1;
      }

      .widget.beating .ring {
        animation: ring-once 420ms ease-out 1;
      }

      .widget.beating .heart {
        animation: heart-once 420ms cubic-bezier(0.18, 0.88, 0.3, 1.08) 1;
      }

      .bpm {
        display: block;
        font-size: 32px;
        line-height: 1;
        font-weight: 950;
        font-variant-numeric: tabular-nums;
      }

      .unit {
        margin-top: 4px;
        color: #aab3b8;
        font-size: 11px;
        font-weight: 900;
      }

      .status {
        margin-top: 8px;
        color: #f5c15b;
        font-size: 12px;
        font-weight: 850;
      }

      .meta {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 6px;
        margin-top: 10px;
      }

      .meta div {
        min-width: 0;
        padding: 6px;
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 8px;
        background: rgba(255, 255, 255, 0.05);
      }

      .meta span {
        display: block;
        color: #9ea9ae;
        font-size: 10px;
        font-weight: 760;
      }

      .meta strong {
        display: block;
        margin-top: 3px;
        color: #f6f4ed;
        font-size: 12px;
        font-weight: 900;
        font-variant-numeric: tabular-nums;
      }

      .widget.has-output .status {
        color: #55d2b0;
      }

      .widget.preview .status,
      .widget.holding .status {
        color: #f5c15b;
      }

      .widget.error .status {
        color: #ff6b83;
      }

      @keyframes pulse-once {
        0%, 100% { transform: scale(1); }
        34% { transform: scale(1.08); }
        62% { transform: scale(0.98); }
      }

      @keyframes ring-once {
        0% { opacity: 0.48; transform: scale(0.7); }
        72%, 100% { opacity: 0; transform: scale(1.38); }
      }

      @keyframes heart-once {
        0%, 100% { filter: saturate(1) brightness(1); transform: rotate(-45deg) scale(1); }
        34% { filter: saturate(1.16) brightness(1.08); transform: rotate(-45deg) scale(1.16); }
        62% { filter: saturate(0.98) brightness(0.98); transform: rotate(-45deg) scale(0.98); }
      }
    `;
  }

  function setOverlayPosition(x, y, { persist = false } = {}) {
    const host = overlayState.host;
    if (!host) return;
    const rect = host.getBoundingClientRect();
    const padding = 10;
    const maxX = Math.max(padding, window.innerWidth - rect.width - padding);
    const maxY = Math.max(padding, window.innerHeight - rect.height - padding);
    const position = {
      x: Math.round(clamp(x, padding, maxX)),
      y: Math.round(clamp(y, padding, maxY)),
    };
    host.style.left = `${position.x}px`;
    host.style.top = `${position.y}px`;
    host.style.right = "auto";
    host.style.bottom = "auto";
    if (persist) {
      try {
        window.localStorage.setItem(OVERLAY_POSITION_KEY, JSON.stringify(position));
      } catch (error) {
        // Ignore storage restrictions on embedded or privacy-hardened pages.
      }
    }
  }

  function restoreOverlayPosition() {
    try {
      const saved = JSON.parse(window.localStorage.getItem(OVERLAY_POSITION_KEY) || "null");
      if (Number.isFinite(saved?.x) && Number.isFinite(saved?.y)) {
        window.requestAnimationFrame(() => setOverlayPosition(saved.x, saved.y));
      }
    } catch (error) {
      try {
        window.localStorage.removeItem(OVERLAY_POSITION_KEY);
      } catch (_storageError) {
        // Ignore storage restrictions.
      }
    }
  }

  function bindOverlayDrag(widget) {
    let dragging = false;
    let dragStart = null;

    widget.addEventListener("pointerdown", (event) => {
      if (event.button != null && event.button !== 0) return;
      if (event.target?.classList?.contains("close")) return;
      const rect = overlayState.host.getBoundingClientRect();
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
      setOverlayPosition(
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
      const rect = overlayState.host.getBoundingClientRect();
      setOverlayPosition(rect.left, rect.top, { persist: true });
    }

    widget.addEventListener("pointerup", finishDrag);
    widget.addEventListener("pointercancel", finishDrag);
    window.addEventListener("resize", () => {
      const rect = overlayState.host?.getBoundingClientRect();
      if (rect) setOverlayPosition(rect.left, rect.top, { persist: true });
    });
  }

  function ensureOverlay() {
    if (overlayState.host && overlayState.shadow) return overlayState.shadow;

    let host = document.getElementById(OVERLAY_HOST_ID);
    if (!host) {
      host = document.createElement("div");
      host.id = OVERLAY_HOST_ID;
      Object.assign(host.style, {
        position: "fixed",
        top: "96px",
        right: "28px",
        zIndex: "2147483647",
        display: "none",
        pointerEvents: "auto",
      });
      document.documentElement.appendChild(host);
    }

    const shadow = host.shadowRoot || host.attachShadow({ mode: "open" });
    if (!shadow.firstChild) {
      shadow.innerHTML = `
        <style>${overlayCss()}</style>
        <section class="widget" role="status" aria-live="polite">
          <div class="top">
            <span class="title">网页视频心率</span>
            <button class="close" type="button" title="停止检测">×</button>
          </div>
          <div class="main">
            <div class="pulse" aria-hidden="true">
              <span class="ring"></span>
              <span class="heart"></span>
            </div>
            <div>
              <strong class="bpm">--</strong>
              <div class="unit">BPM</div>
              <div class="status">待机</div>
            </div>
          </div>
          <div class="meta">
            <div>
              <span>SQI</span>
              <strong class="sqi">--</strong>
            </div>
            <div>
              <span>帧数</span>
              <strong class="frames">0</strong>
            </div>
          </div>
        </section>
      `;
      const widget = shadow.querySelector(".widget");
      bindOverlayDrag(widget);
      shadow.querySelector(".close")?.addEventListener("click", async (event) => {
        event.stopPropagation();
        await sendBackground({ type: "stop" });
        stopOverlayPolling();
        hideOverlay();
      });
    }

    overlayState.host = host;
    overlayState.shadow = shadow;
    restoreOverlayPosition();
    return shadow;
  }

  function showOverlay() {
    ensureOverlay();
    overlayState.host.style.display = "block";
  }

  function hideOverlay() {
    if (overlayState.host) overlayState.host.style.display = "none";
    stopHeartbeatTimer();
  }

  function currentDisplayHr(result = {}) {
    if (["no_face", "failed", "stopped", "waiting"].includes(result.status)) {
      overlayState.lastHr = null;
      overlayState.lastHrAt = 0;
      return { hr: null, fresh: false };
    }

    const current = Number.isFinite(result.bpm)
      ? result.bpm
      : Number.isFinite(result.hr)
        ? Math.round(result.hr)
        : null;
    if (current != null) {
      overlayState.lastHr = current;
      overlayState.lastHrAt = Date.now();
      return { hr: current, fresh: true };
    }

    const canHold = result.status === "warming" && result.reason === "no_recent_input";
    if (canHold && Number.isFinite(overlayState.lastHr) && Date.now() - overlayState.lastHrAt <= LAST_HR_HOLD_MS) {
      return { hr: overlayState.lastHr, fresh: false };
    }

    overlayState.lastHr = null;
    overlayState.lastHrAt = 0;
    return { hr: null, fresh: false };
  }

  function stopHeartbeatTimer() {
    window.clearTimeout(overlayState.heartbeatTimer);
    overlayState.heartbeatTimer = null;
    overlayState.heartbeatHr = null;
    overlayState.shadow?.querySelector(".widget")?.classList.remove("beating");
  }

  function triggerHeartbeatBeat() {
    const widget = overlayState.shadow?.querySelector(".widget");
    if (!widget || !Number.isFinite(overlayState.heartbeatHr)) return;
    widget.classList.remove("beating");
    void widget.offsetWidth;
    widget.classList.add("beating");
    window.setTimeout(() => widget.classList.remove("beating"), 420);
    const interval = Math.max(430, Math.min(1400, 60000 / overlayState.heartbeatHr));
    overlayState.heartbeatTimer = window.setTimeout(triggerHeartbeatBeat, interval);
  }

  function updateHeartbeat(hr, result = {}) {
    const inactive = ["no_face", "failed", "stopped", "waiting"].includes(result.status);
    if (!Number.isFinite(hr) || inactive) {
      stopHeartbeatTimer();
      return;
    }
    if (overlayState.heartbeatTimer && overlayState.heartbeatHr === hr) return;
    window.clearTimeout(overlayState.heartbeatTimer);
    overlayState.heartbeatTimer = null;
    overlayState.heartbeatHr = hr;
    triggerHeartbeatBeat();
  }

  function renderOverlay(status) {
    const shadow = ensureOverlay();
    const widget = shadow.querySelector(".widget");
    const result = status?.result || {};
    const display = currentDisplayHr(result);
    const confidence = Number.isFinite(result.confidence) ? clamp(result.confidence, 0, 1) : 0;
    const arc = confidence * 360;

    shadow.querySelector(".bpm").textContent = display.hr == null ? "--" : String(Math.round(display.hr));
    shadow.querySelector(".sqi").textContent = Number.isFinite(result.SQI) ? result.SQI.toFixed(2) : "--";
    shadow.querySelector(".frames").textContent = String(status?.frame_count || 0);
    shadow.querySelector(".status").textContent = status?.error || statusLabel(result.status || status?.state, result.reason);
    shadow.querySelector(".pulse").style.setProperty("--heart-arc", `${arc}deg`);

    widget.classList.toggle("has-output", Number.isFinite(result.bpm));
    widget.classList.toggle("preview", !Number.isFinite(result.bpm) && Number.isFinite(display.hr));
    widget.classList.toggle("holding", Number.isFinite(display.hr) && !display.fresh);
    widget.classList.toggle("error", Boolean(status?.error) || ["failed", "no_face"].includes(result.status));
    updateHeartbeat(display.hr, result);
  }

  async function pollOverlayStatus() {
    const status = await sendBackground({ type: "status" });
    if (status?.ok) {
      renderOverlay(status);
      if (status.result?.status === "stopped" || status.state === "idle") {
        stopOverlayPolling();
        hideOverlay();
      }
      return;
    }
    renderOverlay({
      ok: false,
      state: "failed",
      frame_count: 0,
      error: status?.error || "插件后台未响应",
      result: { status: "failed" },
    });
  }

  function startOverlayPolling() {
    window.clearInterval(overlayState.statusTimer);
    pollOverlayStatus();
    overlayState.statusTimer = window.setInterval(pollOverlayStatus, STATUS_POLL_MS);
  }

  function stopOverlayPolling() {
    window.clearInterval(overlayState.statusTimer);
    overlayState.statusTimer = null;
  }

  chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
    if (message?.type === "video-rect") {
      const video = bestVideo();
      if (!video) {
        sendResponse({ ok: false, error: "当前页面没有可检测的可见视频。" });
        return true;
      }
      sendResponse({ ok: true, video });
      return true;
    }
    if (message?.type === "browser-heart-started") {
      showOverlay();
      renderOverlay(message.status);
      startOverlayPolling();
      sendResponse({ ok: true });
      return true;
    }
    if (message?.type === "browser-heart-status") {
      showOverlay();
      renderOverlay(message.status);
      sendResponse({ ok: true });
      return true;
    }
    if (message?.type === "browser-heart-stopped") {
      stopOverlayPolling();
      hideOverlay();
      sendResponse({ ok: true });
      return true;
    }
    return false;
  });
}
