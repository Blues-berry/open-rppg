import { FaceTracker } from "../../../modules/face-tracker.js?v=20260722-spectrum-v5";
import { FacePhysEngine } from "../../../modules/facephys-engine.js?v=20260722-spectrum-v5";

const clamp = (value, min = 0, max = 1) => Math.min(max, Math.max(min, Number.isFinite(value) ? value : min));

export class LocalFacePhysCapture {
  constructor(onUpdate, onError) {
    this.onUpdate = onUpdate;
    this.onError = onError;
    this.stream = null;
    this.tracker = null;
    this.engine = null;
    this.running = false;
    this.frameRequest = null;
    this.canvas = document.createElement("canvas");
    this.canvas.width = 36;
    this.canvas.height = 36;
    this.context = this.canvas.getContext("2d", { willReadFrequently: true });
    this.video = document.createElement("video");
    this.video.autoplay = true;
    this.video.muted = true;
    this.video.playsInline = true;
    this.face = null;
    this.lastFaceAt = 0;
    this.lastSampleAt = 0;
    this.lastEngineAt = 0;
    this.lastFrameAt = 0;
    this.fps = 0;
    this.frames = 0;
    this.waveform = [];
    this.duration = 0;
    this.samples = 0;
    this.analysis = { bpm: null, sqi: 0, analysisRevision: 0 };
    this.settings = {};
  }

  async start() {
    if (this.running) return this.snapshot();
    this.stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30, min: 20 } }, audio: false });
    this.video.srcObject = this.stream;
    await this.video.play();
    this.tracker = await FaceTracker.create();
    this.engine = new FacePhysEngine((frame) => this.onFrame(frame), (analysis) => this.onAnalysis(analysis), (error) => this.fail(error));
    await this.engine.initialize();
    this.running = true;
    this.emit();
    this.frameRequest = requestAnimationFrame((now) => this.process(now));
    return this.snapshot();
  }

  stop() {
    this.running = false;
    if (this.frameRequest) cancelAnimationFrame(this.frameRequest);
    this.frameRequest = null;
    this.stream?.getTracks().forEach((track) => track.stop());
    this.stream = null;
    this.video.srcObject = null;
    this.tracker?.close(); this.tracker = null;
    this.engine?.destroy(); this.engine = null;
    this.reset();
  }

  reset() {
    this.engine?.reset();
    this.waveform = []; this.duration = 0; this.samples = 0;
    this.analysis = { bpm: null, sqi: 0, analysisRevision: 0 };
    this.emit();
  }

  process(now) {
    if (!this.running) return;
    if (this.lastFrameAt) this.fps = 1000 / Math.max(1, now - this.lastFrameAt);
    this.lastFrameAt = now;
    if (now - this.lastFaceAt > 90) {
      this.lastFaceAt = now;
      try { this.face = this.tracker.detect(this.video, now); } catch { this.face = null; }
    }
    if (this.face?.valid && now - this.lastSampleAt > 33) {
      this.lastSampleAt = now;
      const sample = this.sample(this.face.bounds);
      if (sample) {
        const dt = this.lastEngineAt ? (now - this.lastEngineAt) / 1000 : 1 / 30;
        this.lastEngineAt = now;
        this.engine.submit(sample, dt, now);
      }
    }
    this.emit();
    this.frameRequest = requestAnimationFrame((time) => this.process(time));
  }

  sample(bounds) {
    const width = this.video.videoWidth;
    const height = this.video.videoHeight;
    if (!width || !height || !bounds) return null;
    const x = clamp(bounds.x) * width;
    const y = clamp(bounds.y - bounds.height * .2) * height;
    const sampleWidth = Math.min(width - x, bounds.width * width);
    const sampleHeight = Math.min(height - y, bounds.height * 1.2 * height);
    if (sampleWidth < 24 || sampleHeight < 24) return null;
    this.context.drawImage(this.video, x, y, sampleWidth, sampleHeight, 0, 0, 36, 36);
    const pixels = this.context.getImageData(0, 0, 36, 36).data;
    const frame = new Float32Array(36 * 36 * 3);
    for (let i = 0, output = 0; i < pixels.length; i += 4) { frame[output++] = pixels[i] / 255; frame[output++] = pixels[i + 1] / 255; frame[output++] = pixels[i + 2] / 255; }
    return frame;
  }

  onFrame(frame) {
    this.duration = frame.duration || 0; this.samples = frame.samples || 0;
    this.waveform.push(frame.value || 0); if (this.waveform.length > 180) this.waveform.shift();
  }

  onAnalysis(analysis) { if (analysis.analysisRevision > this.analysis.analysisRevision) this.analysis = analysis; }
  fail(error) { if (this.running) { this.stop(); this.onError?.(error); } }
  emit() { this.onUpdate?.(this.snapshot()); }

  snapshot(settings = this.settings) {
    const hasFace = Boolean(this.face?.valid);
    const sqi = Number(this.analysis.sqi || 0);
    const bpm = Number.isFinite(this.analysis.bpm) ? this.analysis.bpm : null;
    const ready = this.duration >= 15 && sqi >= .2 && bpm;
    const now = Date.now() / 1000;
    return {
      local: true,
      capture: { state: this.running ? "running" : "idle", device_index: "browser", width: this.video.videoWidth || 1280, height: this.video.videoHeight || 720, input_fps: this.running ? this.fps : 0, target_fps: 30, read_ms: 0, frames_read: this.frames += this.running ? 1 : 0, dropped_frames: 0 },
      model: { ready: this.running, model: "FacePhys · local", hr: bpm, SQI: sqi, input_fps: this.running ? 30 : 0, has_face: hasFace, no_face_count: hasFace ? 0 : 1, hr_window_seconds: 15, perf: { update_ms: 0, metric_ms: 0 }, waveform: { bvp: this.waveform, ts: this.waveform.map((_, index) => now - this.waveform.length / 30 + index / 30) } },
      output: { bpm: ready ? bpm : null, confidence: sqi, status: ready ? "stable" : this.running ? "warming" : "idle", reason: ready ? "local_facephys" : hasFace ? "building_local_window" : "waiting_for_face" },
      settings: { pulse: settings.pulse !== false, light_enabled: Boolean(settings.light_enabled), brightness: Number(settings.brightness || 72), temperature: Number(settings.temperature || 4800), light_x: Number(settings.light_x || 50), light_y: Number(settings.light_y || 38), light_z: Number(settings.light_z || 45), light_range: Number(settings.light_range || 58), light_angle_enabled: Boolean(settings.light_angle_enabled), light_angle: Number(settings.light_angle || 0), light_revision: Date.now() },
      agent: { configured: false, history: [], latest: {} }, highlights: { recording: { enabled: false, state: "unavailable" }, items: [] },
    };
  }
}
