export const QUALITY_GATES = Object.freeze({ face: 0.6, light: 0.55, motion: 0.55, signal: 0.42, sqi: 0.42 });

export function qualityLevel(score) {
  if (score >= 0.65) return "pass";
  if (score >= 0.45) return "caution";
  return "fail";
}

export function evaluateGate(metrics, result, warmupSeconds) {
  if (metrics.face < QUALITY_GATES.face) return gate("face", "人脸区域不足", "请稍微靠近镜头，并让完整面部保持在取景框中央。", result.bpm);
  if (metrics.light < QUALITY_GATES.light) return gate("light", "面部光线不均", "请避开逆光、暗光与过曝，让面部获得柔和均匀的光线。", result.bpm);
  if (metrics.motion < QUALITY_GATES.motion) return gate("motion", "画面移动过多", "请正对镜头并短暂保持稳定，减少头部和设备移动。", result.bpm);
  if (result.duration < warmupSeconds) return gate("window", "正在建立信号", `保持当前状态，还需 ${Math.max(0, warmupSeconds - result.duration).toFixed(1)} 秒。`);
  if (metrics.signal < QUALITY_GATES.signal || metrics.sqi < QUALITY_GATES.sqi || !result.bpm) return gate("signal", "脉搏信号较弱", "请保持面部光线稳定，并减少动作后继续等待。", result.bpm);
  return { accepted: true, code: "ready", title: "信号稳定", hint: "脉搏信号已达到显示条件，请继续自然呼吸并保持放松。" };
}

function gate(code, title, hint, candidateBpm = 0) { return { accepted: false, code, title, hint, candidateBpm }; }
