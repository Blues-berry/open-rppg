import { FaceTracker } from "../../../modules/face-tracker.js?v=20260722-spectrum-v5";
import { FacePhysEngine } from "../../../modules/facephys-engine.js?v=20260722-spectrum-v5";
import { evaluateGate } from "../../../modules/quality-gate.js?v=20260722-spectrum-v5";

const clamp = (value, min = 0, max = 1) => Math.min(max, Math.max(min, Number.isFinite(value) ? value : min));

export class LocalFacePhysCapture {
  constructor(video, onUpdate, onError, onStream) {
    this.video = video;
    this.onUpdate = onUpdate;
    this.onError = onError;
    this.onStream = onStream;
    this.stream = null;
    this.tracker = null;
    this.engine = null;
    this.running = false;
    this.frameRequest = null;
    this.canvas = document.createElement("canvas");
    this.canvas.width = 36;
    this.canvas.height = 36;
    this.context = this.canvas.getContext("2d", { willReadFrequently: true });
    this.video.autoplay = true;
    this.video.muted = true;
    this.video.playsInline = true;
    this.face = null;
    this.lastFaceAt = 0;
    this.lastSampleAt = 0;
    this.lastEngineAt = 0;
    this.lastFrameAt = 0;
    this.lastEmitAt = 0;
    this.fps = 0;
    this.fpsSamples = [];
    this.frames = 0;
    this.waveform = [];
    this.duration = 0;
    this.samples = 0;
    this.analysis = { bpm: null, sqi: 0, analysisRevision: 0 };
    this.metrics = { brightness: 0, light: 0, motion: 0, face: 0, signal: 0, sqi: 0 };
    this.settings = {};
  }

  async start() {
    if (this.running) return this.snapshot();
    if (!this.stream) {
      this.stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30, min: 20 } }, audio: false });
      this.video.srcObject = this.stream;
      await this.video.play();
      this.onStream?.(this.stream);
    }
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
    this.fpsSamples = []; this.fps = 0; this.lastFrameAt = 0;
    this.metrics = { brightness: 0, light: 0, motion: 0, face: 0, signal: 0, sqi: 0 };
    this.emit();
  }

  process(now) {
    if (!this.running) return;
    if (this.lastFrameAt) {
      this.fpsSamples.push(Math.max(1, now - this.lastFrameAt));
      if (this.fpsSamples.length > 30) this.fpsSamples.shift();
      const averageFrameMs = this.fpsSamples.reduce((sum, value) => sum + value, 0) / this.fpsSamples.length;
      this.fps = 1000 / averageFrameMs;
    }
    this.lastFrameAt = now;
    this.frames += 1;
    if (now - this.lastFaceAt > 90) {
      this.lastFaceAt = now;
      try { this.face = this.tracker.detect(this.video, now); } catch { this.face = null; }
    }
    if (this.face?.valid && now - this.lastSampleAt > 33) {
      this.lastSampleAt = now;
      const sample = this.sample(this.face.bounds);
      if (sample) {
        const brightness = sample.brightness;
        const light = brightness < .16 || brightness > .9 ? .08 : clamp(1 - Math.abs(brightness - .52) / .43);
        this.metrics = {
          brightness,
          light,
          motion: clamp(this.face.motionQuality),
          face: clamp(this.face.faceQuality),
          signal: clamp(this.analysis.signal ?? this.analysis.sqi),
          sqi: clamp(this.analysis.sqi),
        };
        const dt = this.lastEngineAt ? (now - this.lastEngineAt) / 1000 : 1 / 30;
        this.lastEngineAt = now;
        this.engine.submit(sample.frame, dt, now);
      }
    }
    if (now - this.lastEmitAt >= 120) {
      this.lastEmitAt = now;
      this.emit();
    }
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
    let luminance = 0;
    for (let i = 0, output = 0; i < pixels.length; i += 4) {
      const red = pixels[i] / 255; const green = pixels[i + 1] / 255; const blue = pixels[i + 2] / 255;
      frame[output++] = red; frame[output++] = green; frame[output++] = blue;
      luminance += red * .2126 + green * .7152 + blue * .0722;
    }
    return { frame, brightness: luminance / (36 * 36) };
  }

  onFrame(frame) {
    this.duration = frame.duration || 0; this.samples = frame.samples || 0;
    this.waveform.push(frame.value || 0); if (this.waveform.length > 180) this.waveform.shift();
  }

  onAnalysis(analysis) {
    if (analysis.analysisRevision <= this.analysis.analysisRevision) return;
    this.analysis = analysis;
    this.metrics.signal = clamp(analysis.signal ?? analysis.sqi);
    this.metrics.sqi = clamp(analysis.sqi);
    this.emit();
  }
  fail(error) { if (this.running) { this.stop(); this.onError?.(error); } }
  emit() { this.onUpdate?.(this.snapshot()); }

  snapshot(settings = this.settings) {
    const hasFace = Boolean(this.face?.valid);
    const sqi = clamp(this.analysis.sqi);
    const candidateBpm = Number(this.analysis.bpm);
    // FacePhys estimates outside this physiological range are transient detector artefacts.
    const bpm = Number.isFinite(candidateBpm) && candidateBpm >= 38 && candidateBpm <= 220 ? candidateBpm : null;
    const metrics = { ...this.metrics, fps: this.fps, signal: clamp(this.analysis.signal ?? sqi), sqi };
    const gate = evaluateGate(metrics, { bpm, duration: this.duration, sqi }, 15);
    const ready = gate.accepted;
    const now = Date.now() / 1000;
    return {
      local: true,
      capture: { state: this.running ? "running" : this.stream ? "starting" : "idle", device_index: "browser", width: this.video.videoWidth || 1280, height: this.video.videoHeight || 720, input_fps: this.running ? this.fps : 0, target_fps: 30, read_ms: 0, frames_read: this.frames, dropped_frames: 0 },
      model: { ready: this.running, model: "FacePhys · local", hr: bpm, SQI: sqi, input_fps: this.running ? this.fps : 0, has_face: hasFace, no_face_count: hasFace ? 0 : 1, hr_window_seconds: 15, metrics, perf: { update_ms: 0, metric_ms: 0 }, waveform: { bvp: this.waveform, ts: this.waveform.map((_, index) => now - this.waveform.length / 30 + index / 30) } },
      output: { bpm: ready ? bpm : null, confidence: sqi, status: ready ? "stable" : this.running ? "warming" : "idle", reason: ready ? "local_facephys" : gate.code === "face" ? "waiting_for_face" : gate.code },
      settings: { pulse: settings.pulse !== false, light_enabled: Boolean(settings.light_enabled), brightness: Number(settings.brightness || 72), temperature: Number(settings.temperature || 4800), light_x: Number(settings.light_x || 50), light_y: Number(settings.light_y || 38), light_z: Number(settings.light_z || 45), light_range: Number(settings.light_range || 58), light_angle_enabled: Boolean(settings.light_angle_enabled), light_angle: Number(settings.light_angle || 0), light_revision: Date.now() },
      agent: { configured: false, history: [], latest: {} }, highlights: { recording: { enabled: false, state: "unavailable" }, items: [] },
    };
  }
}
