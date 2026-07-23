const BUFFER_SIZE = 450, FAST_INTERVAL = 125, NORMAL_INTERVAL = 160, LOW_POWER_INTERVAL = 250;

export class FacePhysEngine {
  constructor(onFrame, onAnalysis, onError) {
    this.onFrame = onFrame; this.onAnalysis = onAnalysis; this.onError = onError;
    this.inferenceWorker = null; this.analysisWorker = null; this.ready = false;
    this.inferencePending = false; this.analysisPending = false;
    this.values = new Float32Array(BUFFER_SIZE); this.intervals = new Float32Array(BUFFER_SIZE);
    this.cursor = 0; this.count = 0; this.analysisRevision = 0; this.epoch = 0; this.lastAnalysisAt = 0;
    this.frameTimes = []; this.lowPower = isLowPowerDevice(); this.analysisInterval = this.lowPower ? LOW_POWER_INTERVAL : NORMAL_INTERVAL;
  }

  async initialize() {
    this.destroy(); this.resetBuffers();
    this.inferenceWorker = new Worker(new URL("./facephys-worker.js?v=20260722-spectrum-v5", import.meta.url));
    this.analysisWorker = new Worker(new URL("./facephys-analysis-worker.js?v=20260722-spectrum-v5", import.meta.url));
    return new Promise((resolve, reject) => {
      const ready = new Set(); let settled = false;
      const timer = setTimeout(() => fail(new Error("FacePhys 初始化超时")), 45000);
      const finish = () => { if (ready.size < 2 || settled) return; settled = true; clearTimeout(timer); this.ready = true; resolve(); };
      const fail = (error) => { if (settled) return; settled = true; clearTimeout(timer); this.onError?.(error); reject(error); };
      this.inferenceWorker.onmessage = ({ data }) => {
        if (data?.type === "ready") { ready.add("inference"); finish(); }
        else if (data?.type === "frame") { this.inferencePending = false; this.handleFrame(data.payload); }
        else if (data?.type === "error") failOrReport.call(this, data, fail);
      };
      this.analysisWorker.onmessage = ({ data }) => {
        if (data?.type === "ready") { ready.add("analysis"); finish(); }
        else if (data?.type === "analysis") { this.analysisPending = false; if (data.payload?.epoch !== this.epoch) return; this.adjustAnalysisInterval(data.payload); this.onAnalysis?.(data.payload); }
        else if (data?.type === "error") { this.analysisPending = false; failOrReport.call(this, data, fail); }
      };
      this.inferenceWorker.onerror = (event) => fail(event.error || new Error(event.message || "FacePhys 主 Worker 加载失败"));
      this.analysisWorker.onerror = (event) => fail(event.error || new Error(event.message || "FacePhys 分析 Worker 加载失败"));
      this.inferenceWorker.postMessage({ type: "init" }); this.analysisWorker.postMessage({ type: "init" });
    });
  }

  submit(frame, dt, timestamp) {
    if (!this.ready || this.inferencePending) return false;
    this.inferencePending = true;
    this.inferenceWorker.postMessage({ type: "run", payload: { frame, dt, timestamp } }, [frame.buffer]);
    return true;
  }

  handleFrame(payload) {
    const dt = Math.min(.12, Math.max(.015, Number(payload.dt) || 1 / 30));
    this.values[this.cursor] = Number(payload.value) || 0; this.intervals[this.cursor] = dt;
    this.cursor = (this.cursor + 1) % BUFFER_SIZE; this.count = Math.min(BUFFER_SIZE, this.count + 1);
    const timestamp = Number(payload.timestamp) || performance.now(); this.frameTimes.push(timestamp);
    if (this.frameTimes.length > 60) this.frameTimes.shift();
    this.onFrame?.(payload); this.scheduleAnalysis();
  }

  scheduleAnalysis() {
    const now = performance.now();
    if (this.count < BUFFER_SIZE || this.analysisPending || now - this.lastAnalysisAt < this.analysisInterval) return;
    const window = orderedCopy(this.values, this.cursor), orderedDt = orderedCopy(this.intervals, this.cursor);
    const sampleInterval = orderedDt.reduce((sum, value) => sum + value, 0) / BUFFER_SIZE;
    this.analysisPending = true; this.lastAnalysisAt = now; this.analysisRevision += 1;
    this.analysisWorker.postMessage({ type: "analyze", payload: { window, sampleInterval, analysisRevision: this.analysisRevision, epoch: this.epoch } }, [window.buffer]);
  }

  adjustAnalysisInterval({ analysisLatency = 0 }) {
    const fps = effectiveFps(this.frameTimes);
    this.analysisInterval = chooseAnalysisInterval(this.lowPower, analysisLatency, fps);
  }

  reset() {
    this.inferencePending = false; this.analysisPending = false; this.epoch += 1; this.resetBuffers();
    if (this.ready) { this.inferenceWorker?.postMessage({ type: "reset" }); this.analysisWorker?.postMessage({ type: "reset" }); }
  }

  resetBuffers() {
    this.values.fill(0); this.intervals.fill(0); this.cursor = 0; this.count = 0; this.analysisRevision = 0;
    this.lastAnalysisAt = 0; this.frameTimes = []; this.analysisInterval = this.lowPower ? LOW_POWER_INTERVAL : NORMAL_INTERVAL;
  }

  destroy() {
    this.ready = false; this.inferencePending = false; this.analysisPending = false;
    this.inferenceWorker?.terminate(); this.analysisWorker?.terminate(); this.inferenceWorker = null; this.analysisWorker = null;
  }
}

function orderedCopy(buffer, cursor) { const ordered = new Float32Array(buffer.length); ordered.set(buffer.subarray(cursor)); ordered.set(buffer.subarray(0, cursor), buffer.length - cursor); return ordered; }
function effectiveFps(times) { if (times.length < 2) return 0; const elapsed = times[times.length - 1] - times[0]; return elapsed > 0 ? (times.length - 1) * 1000 / elapsed : 0; }
function chooseAnalysisInterval(lowPower, latency, fps) { if (lowPower || latency > 100 || (fps && fps < 20)) return LOW_POWER_INTERVAL; if (latency < 55 && fps >= 25) return FAST_INTERVAL; return NORMAL_INTERVAL; }
function isLowPowerDevice() { return (navigator.hardwareConcurrency && navigator.hardwareConcurrency <= 4) || (navigator.deviceMemory && navigator.deviceMemory <= 4) || matchMedia("(max-width: 700px)").matches; }
function failOrReport(data, fail) { const error = new Error(`${data.source || "FacePhys"}: ${data.message || "运行失败"}`); if (!this.ready) fail(error); else this.onError?.(error); }

export const __test__ = { orderedCopy, effectiveFps, chooseAnalysisInterval, constants: { BUFFER_SIZE, FAST_INTERVAL, NORMAL_INTERVAL, LOW_POWER_INTERVAL } };
