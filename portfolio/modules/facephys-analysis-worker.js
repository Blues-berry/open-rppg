const BUFFER_SIZE = 450, INPUT_SHAPE = [1, BUFFER_SIZE];
let LiteRT, Tensor, sqiModel, psdModel;
const siteRoot = new URL("../", self.location.href);
const asset = (name) => new URL(`assets/models/facephys/${name}`, siteRoot).href;
const wasmRoot = new URL("assets/vendor/litert/", siteRoot).href;
const litertModuleUrl = new URL("assets/vendor/litert/index.js", siteRoot).href;

self.onmessage = async ({ data }) => {
  try {
    if (data.type === "init") await init();
    else if (data.type === "analyze") analyze(data.payload);
  } catch (error) { self.postMessage({ type: "error", source: "analysis", message: error?.message || String(error) }); }
};

async function init() {
  LiteRT = await import(litertModuleUrl); Tensor = LiteRT.Tensor;
  await loadLocalLiteRt();
  const [sqi, psd] = await Promise.all([fetch(asset("sqi_model.tflite")), fetch(asset("psd_model.tflite"))]);
  if (!sqi.ok || !psd.ok) throw new Error("FacePhys 频谱模型资源加载失败");
  sqiModel = await LiteRT.loadAndCompile(new Uint8Array(await sqi.arrayBuffer()), { accelerator: "wasm" });
  psdModel = await LiteRT.loadAndCompile(new Uint8Array(await psd.arrayBuffer()), { accelerator: "wasm" });
  self.postMessage({ type: "ready", source: "analysis" });
}

async function loadLocalLiteRt() {
  const nativeFetch = self.fetch.bind(self);
  self.fetch = (input, options) => {
    const requested = typeof input === "string" ? input : input?.url || String(input);
    if (/\.wasm(?:$|[?#])/.test(requested)) {
      const fileName = new URL(requested, self.location.href).pathname.split("/").pop();
      return nativeFetch(new URL(fileName, wasmRoot), options);
    }
    return nativeFetch(input, options);
  };
  try { await LiteRT.loadLiteRt(wasmRoot); } finally { self.fetch = nativeFetch; }
}

function analyze({ window, sampleInterval, analysisRevision, epoch }) {
  if (!sqiModel || !psdModel || window.length !== BUFFER_SIZE) return;
  const startedAt = performance.now(), input = new Tensor(window, INPUT_SHAPE);
  const sqiOutput = sqiModel.run([input]);
  const sqi = sqiOutput[0]?.toTypedArray?.()[0] || 0; sqiOutput.forEach((tensor) => tensor?.delete?.());
  const psdOutput = psdModel.run([input]);
  const rawHr = psdOutput[0]?.toTypedArray?.()[0] || 0;
  const freq = psdOutput[1]?.toTypedArray?.() || [], psd = psdOutput[2]?.toTypedArray?.() || [];
  const bpm = rawHr / 30 / sampleInterval;
  const spectrum = Array.from(freq, (value, index) => ({ bpm: value * 60 / sampleInterval, power: psd[index] || 0 }));
  psdOutput.forEach((tensor) => tensor?.delete?.()); input.delete();
  self.postMessage({ type: "analysis", payload: { analysisRevision, epoch, sqi, bpm, spectrum, analysisLatency: performance.now() - startedAt, sampleInterval } });
}
