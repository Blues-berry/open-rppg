export const QUALITY_GATES = Object.freeze({ face: 0.6, light: 0.55, motion: 0.55, signal: 0.42, sqi: 0.42 });

export function qualityLevel(score) {
  if (score >= 0.65) return "pass";
  if (score >= 0.45) return "caution";
  return "fail";
}

export function evaluateGate(metrics, result, warmupSeconds) {
  if (metrics.face < QUALITY_GATES.face) return gate("face", "ROI 可用性不足", "请靠近镜头并保持整张脸位于取景框中央。", result.bpm);
  if (metrics.light < QUALITY_GATES.light) return gate("light", "光照需调整", "避免逆光、过暗或明显过曝，让面部均匀受光。", result.bpm);
  if (metrics.motion < QUALITY_GATES.motion) return gate("motion", "请保持静止", "请减少头部移动，保持正对镜头数秒。", result.bpm);
  if (result.duration < warmupSeconds) return gate("window", "建立信号中", `正在累积色彩窗口，还需 ${Math.max(0, warmupSeconds - result.duration).toFixed(1)} 秒。`);
  if (metrics.signal < QUALITY_GATES.signal || metrics.sqi < QUALITY_GATES.sqi || !result.bpm) return gate("signal", "频谱质量不足", "已暂停心率输出；请保持自然光照和相对静止。", result.bpm);
  return { accepted: true, code: "ready", title: "稳定监测", hint: "已检测到稳定的本地色彩周期变化；请保持自然呼吸与相对静止。" };
}

function gate(code, title, hint, candidateBpm = 0) { return { accepted: false, code, title, hint, candidateBpm }; }
