const INPUT_COUNT = 48, IMG_IDX = 1, DT_IDX = 0, IMG_SHAPE = [1, 1, 36, 36, 3];
const STATE_MAP = [{inIdx:2,outIdx:1},{inIdx:3,outIdx:12},{inIdx:14,outIdx:23},{inIdx:25,outIdx:34},{inIdx:36,outIdx:42},{inIdx:43,outIdx:43},{inIdx:44,outIdx:44},{inIdx:45,outIdx:45},{inIdx:46,outIdx:46},{inIdx:47,outIdx:2},{inIdx:4,outIdx:3},{inIdx:5,outIdx:4},{inIdx:6,outIdx:5},{inIdx:7,outIdx:6},{inIdx:8,outIdx:7},{inIdx:9,outIdx:8},{inIdx:10,outIdx:9},{inIdx:11,outIdx:10},{inIdx:12,outIdx:11},{inIdx:13,outIdx:13},{inIdx:15,outIdx:14},{inIdx:16,outIdx:15},{inIdx:17,outIdx:16},{inIdx:18,outIdx:17},{inIdx:19,outIdx:18},{inIdx:20,outIdx:19},{inIdx:21,outIdx:20},{inIdx:22,outIdx:21},{inIdx:23,outIdx:22},{inIdx:24,outIdx:24},{inIdx:26,outIdx:25},{inIdx:27,outIdx:26},{inIdx:28,outIdx:27},{inIdx:29,outIdx:28},{inIdx:30,outIdx:29},{inIdx:31,outIdx:30},{inIdx:32,outIdx:31},{inIdx:33,outIdx:32},{inIdx:34,outIdx:33},{inIdx:35,outIdx:35},{inIdx:37,outIdx:36},{inIdx:38,outIdx:37},{inIdx:39,outIdx:38},{inIdx:40,outIdx:39},{inIdx:41,outIdx:40},{inIdx:42,outIdx:41}];
let LiteRT, Tensor, model, inputs = [], initialState = {}, samples = 0, elapsed = 0;
const siteRoot = new URL("../", self.location.href);
const asset = (name) => new URL(`assets/models/facephys/${name}`, siteRoot).href;
const wasmRoot = new URL("assets/vendor/litert/", siteRoot).href;
const litertModuleUrl = new URL("assets/vendor/litert/index.js", siteRoot).href;

self.onmessage = async ({ data }) => {
  try {
    if (data.type === "init") await init();
    else if (data.type === "run") run(data.payload);
    else if (data.type === "reset") reset();
  } catch (error) { self.postMessage({ type: "error", source: "inference", message: error?.message || String(error) }); }
};

async function init() {
  LiteRT = await import(litertModuleUrl); Tensor = LiteRT.Tensor;
  await loadLocalLiteRt();
  const [main, state] = await Promise.all([fetch(asset("model.tflite")), fetch(asset("state.gz"))]);
  if (!main.ok || !state.ok) throw new Error("FacePhys 主模型资源加载失败");
  model = await LiteRT.loadAndCompile(new Uint8Array(await main.arrayBuffer()), { accelerator: "wasm" });
  const stream = state.body.pipeThrough(new DecompressionStream("gzip"));
  initialState = await new Response(stream).json();
  reset(); self.postMessage({ type: "ready", source: "inference" });
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

function reset() {
  inputs.forEach((tensor) => tensor?.delete?.()); inputs = new Array(INPUT_COUNT);
  const meta = model.getInputDetails();
  for (let i = 0; i < INPUT_COUNT; i += 1) {
    const detail = meta[i], size = detail.shape.reduce((a, b) => a * b, 1); let data;
    if (i === IMG_IDX) data = new Float32Array(size);
    else if (i === DT_IDX) data = new Float32Array([1 / 30]);
    else { const raw = initialState[detail.name]; data = raw ? new Float32Array(raw.flat(Infinity)) : new Float32Array(size); }
    inputs[i] = new Tensor(data, detail.shape);
  }
  samples = 0; elapsed = 0;
}

function run({ frame, dt, timestamp }) {
  if (!model) return;
  const safeDt = Math.min(.12, Math.max(.015, Number(dt) || 1 / 30));
  inputs[IMG_IDX].delete(); inputs[DT_IDX].delete();
  inputs[IMG_IDX] = new Tensor(frame, IMG_SHAPE); inputs[DT_IDX] = new Tensor(new Float32Array([safeDt]), [1]);
  const output = model.run(inputs), used = new Set();
  for (const map of STATE_MAP) {
    const tensor = output[map.outIdx];
    if (tensor) { inputs[map.inIdx]?.delete(); inputs[map.inIdx] = tensor; used.add(map.outIdx); }
  }
  const value = output[0]?.toTypedArray?.()[0] || 0;
  output.forEach((tensor, index) => { if (!used.has(index)) tensor?.delete?.(); });
  samples += 1; elapsed += safeDt;
  self.postMessage({ type: "frame", payload: { value, dt: safeDt, timestamp, samples, duration: elapsed } });
}
