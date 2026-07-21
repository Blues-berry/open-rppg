import { FaceLandmarker, FilesetResolver } from "../assets/vendor/mediapipe/vision_bundle.mjs";

const WASM_ROOT = new URL("../assets/vendor/mediapipe/", import.meta.url).href;
const MODEL_PATH = new URL("../assets/models/face_landmarker.task", import.meta.url).href;
const FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 152, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109];

export class FaceTracker {
  static async create() {
    const vision = await FilesetResolver.forVisionTasks(WASM_ROOT);
    let landmarker;
    try { landmarker = await FaceLandmarker.createFromOptions(vision, { baseOptions: { modelAssetPath: MODEL_PATH, delegate: "GPU" }, runningMode: "VIDEO", numFaces: 2, minFaceDetectionConfidence: 0.58, minFacePresenceConfidence: 0.58, minTrackingConfidence: 0.58 }); }
    catch { landmarker = await FaceLandmarker.createFromOptions(vision, { baseOptions: { modelAssetPath: MODEL_PATH, delegate: "CPU" }, runningMode: "VIDEO", numFaces: 2, minFaceDetectionConfidence: 0.58, minFacePresenceConfidence: 0.58, minTrackingConfidence: 0.58 }); }
    return new FaceTracker(landmarker);
  }

  constructor(landmarker) { this.landmarker = landmarker; this.previous = null; this.lastValid = null; this.lastValidAt = 0; }
  close() { this.landmarker?.close(); }
  detect(video, timestamp) {
    const result = this.landmarker.detectForVideo(video, timestamp);
    const faces = result.faceLandmarks || [];
    if (faces.length > 1) return { valid: false, reason: "multiple" };
    if (!faces.length) return this.reuseRecent(timestamp);
    const landmarks = faces[0]; const bounds = boundsFor(landmarks);
    if (bounds.width * bounds.height < 0.075) return this.reuseRecent(timestamp, "small");
    const eyeDistance = distance(landmarks[33], landmarks[263]);
    const yaw = eyeDistance ? Math.abs(landmarks[1].x - (landmarks[33].x + landmarks[263].x) / 2) / eyeDistance : 1;
    if (yaw > 0.32) return this.reuseRecent(timestamp, "pose");
    const center = { x: bounds.x + bounds.width / 2, y: bounds.y + bounds.height / 2 };
    const movement = this.previous ? Math.hypot(center.x - this.previous.x, center.y - this.previous.y) / Math.max(bounds.width, bounds.height) : 0;
    this.previous = center;
    const faceQuality = Math.min(1, (bounds.width * bounds.height) / 0.2) * clamp(1 - yaw / 0.32) * (center.x > 0.08 && center.x < 0.92 && center.y > 0.06 && center.y < 0.94 ? 1 : 0.55);
    const detected = { valid: true, landmarks, outline: FACE_OVAL.map((index) => landmarks[index]), bounds, rois: buildRois(bounds), motionQuality: clamp(1 - movement * 4.8), faceQuality: clamp(faceQuality), yaw };
    this.lastValid = detected; this.lastValidAt = timestamp;
    return detected;
  }
  reuseRecent(timestamp, reason = "missing") { return this.lastValid && timestamp - this.lastValidAt < 360 ? { ...this.lastValid, stale: true } : { valid: false, reason }; }
}

function buildRois(bounds) {
  const { x, y, width: w, height: h } = bounds;
  return [rect(x + w * 0.31, y + h * 0.18, w * 0.38, h * 0.2), rect(x + w * 0.14, y + h * 0.49, w * 0.25, h * 0.2), rect(x + w * 0.61, y + h * 0.49, w * 0.25, h * 0.2)];
}
function rect(x, y, width, height) { return [{ x, y }, { x: x + width, y }, { x: x + width * 0.94, y: y + height }, { x: x + width * 0.06, y: y + height }]; }
function boundsFor(landmarks) { const xs = landmarks.map((p) => p.x), ys = landmarks.map((p) => p.y); const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys); return { x: x0, y: y0, width: x1 - x0, height: y1 - y0 }; }
function distance(a, b) { return Math.hypot(a.x - b.x, a.y - b.y); }
function clamp(value) { return Math.min(1, Math.max(0, Number.isFinite(value) ? value : 0)); }
