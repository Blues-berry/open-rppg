const video = document.getElementById("tabVideo");
const canvas = document.getElementById("frameCanvas");
const ctx = canvas.getContext("2d", { willReadFrequently: false });
const DEFAULT_CAPTURE_FPS = 30;

let mediaStream = null;
let captureTimer = null;
let captureConfig = null;
let frameSeq = 0;
let sending = false;

function stopCapture() {
  window.clearInterval(captureTimer);
  captureTimer = null;
  captureConfig = null;
  frameSeq = 0;
  sending = false;
  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
  }
  mediaStream = null;
  video.srcObject = null;
}

function cropRect(config) {
  const rect = config.video.rect;
  const viewport = config.video.viewport;
  const videoWidth = video.videoWidth || viewport.width;
  const videoHeight = video.videoHeight || viewport.height;
  const scaleX = videoWidth / Math.max(1, viewport.width);
  const scaleY = videoHeight / Math.max(1, viewport.height);
  return {
    sx: Math.max(0, Math.round(rect.x * scaleX)),
    sy: Math.max(0, Math.round(rect.y * scaleY)),
    sw: Math.max(1, Math.round(rect.width * scaleX)),
    sh: Math.max(1, Math.round(rect.height * scaleY)),
  };
}

async function sendFrame() {
  if (!captureConfig || sending || !video.videoWidth || !video.videoHeight) return;
  sending = true;
  try {
    const crop = cropRect(captureConfig);
    const targetWidth = 360;
    const targetHeight = Math.max(1, Math.round((crop.sh / crop.sw) * targetWidth));
    canvas.width = targetWidth;
    canvas.height = targetHeight;
    ctx.drawImage(video, crop.sx, crop.sy, crop.sw, crop.sh, 0, 0, targetWidth, targetHeight);
    const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.78));
    if (!blob) return;
    const fps = captureConfig.targetFps || DEFAULT_CAPTURE_FPS;
    const ts = frameSeq / fps;
    frameSeq += 1;
    await fetch(`${captureConfig.serviceBase}/session/${captureConfig.sessionId}/frame?ts=${ts}`, {
      method: "POST",
      headers: { "Content-Type": "image/jpeg" },
      body: blob,
    });
  } catch (error) {
    // Popup polls the local service for surfaced errors; keep capture loop alive for transient failures.
  } finally {
    sending = false;
  }
}

async function startCapture(config) {
  stopCapture();
  captureConfig = config;
  mediaStream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: {
      mandatory: {
        chromeMediaSource: "tab",
        chromeMediaSourceId: config.streamId,
      },
    },
  });
  video.srcObject = mediaStream;
  await video.play();
  const fps = captureConfig.targetFps || DEFAULT_CAPTURE_FPS;
  captureTimer = window.setInterval(sendFrame, 1000 / fps);
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  (async () => {
    if (message?.type === "offscreen-start") {
      await startCapture(message);
      sendResponse({ ok: true });
    } else if (message?.type === "offscreen-stop") {
      stopCapture();
      sendResponse({ ok: true });
    }
  })().catch((error) => sendResponse({ ok: false, error: error.message }));
  return true;
});
