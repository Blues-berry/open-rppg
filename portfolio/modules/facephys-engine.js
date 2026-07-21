export class FacePhysEngine {
  constructor(onResult, onError) { this.onResult = onResult; this.onError = onError; this.worker = null; this.ready = false; this.pending = false; }
  async initialize() {
    this.worker?.terminate(); this.ready = false; this.pending = false;
    this.worker = new Worker(new URL("./facephys-worker.js", import.meta.url), { type: "module" });
    return new Promise((resolve, reject) => {
      let settled = false;
      const finishError = (error) => { if (settled) return; settled = true; clearTimeout(timer); this.pending = false; this.onError?.(error); reject(error); };
      const timer = setTimeout(() => finishError(new Error("FacePhys 初始化超时")), 30000);
      this.worker.onmessage = (event) => { const { type, payload, message } = event.data || {}; if (type === "ready") { if (settled) return; settled = true; clearTimeout(timer); this.ready = true; resolve(); } else if (type === "result") { this.pending = false; this.onResult?.(payload); } else if (type === "error") { const error = new Error(message || "FacePhys 运行失败"); this.pending = false; if (!this.ready) finishError(error); else this.onError?.(error); } };
      this.worker.onerror = (event) => { clearTimeout(timer); this.onError?.(event.error || new Error(event.message)); reject(event.error || new Error(event.message)); };
      this.worker.postMessage({ type: "init" });
    });
  }
  submit(frame, dt, timestamp) { if (!this.ready || this.pending) return false; this.pending = true; this.worker.postMessage({ type: "run", payload: { frame, dt, timestamp } }, [frame.buffer]); return true; }
  reset() { this.pending = false; if (this.ready) this.worker?.postMessage({ type: "reset" }); }
  destroy() { this.ready = false; this.pending = false; this.worker?.terminate(); this.worker = null; }
}
