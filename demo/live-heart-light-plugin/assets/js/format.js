export function clamp(value, min = 0, max = 1) {
  return Math.min(max, Math.max(min, value));
}

export function formatMs(value) {
  return Number.isFinite(value) ? `${value.toFixed(value >= 10 ? 0 : 1)}ms` : "--";
}

export function formatFps(value) {
  return Number.isFinite(value) ? `${value.toFixed(value >= 10 ? 0 : 1)}fps` : "--";
}

export function formatCount(value) {
  return Number.isFinite(value) ? String(Math.round(value)) : "--";
}

export function formatPercent(value) {
  return `${Math.round(clamp(Number(value || 0)) * 100)}%`;
}

export function formatBpm(value) {
  return Number.isFinite(value) ? String(Math.round(value)) : "--";
}

export function formatElapsed(seconds) {
  if (!Number.isFinite(seconds)) return "--:--";
  const total = Math.max(0, Math.round(seconds));
  const minutes = Math.floor(total / 60);
  const secs = total % 60;
  return `${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
}

export function basename(path) {
  if (!path) return "--";
  return String(path).split(/[\\/]/).filter(Boolean).pop() || String(path);
}

export function formatReason(reason) {
  const reasonMap = {
    ready: "已输出",
    low_sqi_preview: "低置信预览",
    low_sqi: "SQI 偏低",
    no_face: "未检测到稳定人脸",
    no_recent_input: "等待新帧",
    building_window: "建立窗口",
    model_loading: "模型加载中",
    capture_idle: "采集待机",
  };
  return reasonMap[reason] || reason || "--";
}

export function formatOutputStatus(status, hasBpm) {
  if (hasBpm) return "直播中";
  const statusMap = {
    preview: "预览",
    low_sqi: "锁定",
    no_face: "等待人脸",
    warming: "校准中",
    waiting: "待机",
  };
  return statusMap[status] || "锁定";
}
