const OVERLAY_STATE_URL = "/api/overlay/state";

const ui = {
  widget: document.getElementById("overlayWidget"),
  pulse: document.getElementById("overlayPulse"),
  bpm: document.getElementById("overlayBpm"),
  status: document.getElementById("overlayStatus"),
  agentSubtitle: document.getElementById("agentSubtitle"),
};

function clamp(value, min = 0, max = 1) {
  return Math.min(max, Math.max(min, value));
}

function statusText(output) {
  if (!output || output.status === "waiting") return "等待采样";
  if (output.status === "stable" && output.bpm) return "Open-rppg 模型";
  if (output.status === "preview") return "低置信预览";
  if (output.status === "low_sqi") return "SQI 偏低";
  if (output.status === "no_face") return "等待人脸";
  if (output.status === "warming") return "正在校准";
  return "等待采样";
}

async function readState() {
  const response = await fetch(OVERLAY_STATE_URL, { cache: "no-store" });
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json();
}

function renderState(state) {
  const output = state.output || {};
  const settings = state.settings || {};
  const agent = state.agent || {};
  const bpm = output.bpm;
  const confidence = clamp(Number(output.confidence || 0));
  const arc = confidence * 360;

  ui.bpm.textContent = bpm || "--";
  ui.status.textContent = statusText(output);
  renderAgentSubtitle(agent);

  ui.widget.classList.toggle("pulsing", Boolean(settings.pulse && bpm && confidence > 0.45));
  ui.widget.style.opacity = bpm ? "1" : "0.72";

  const interval = bpm ? Math.max(430, Math.min(1200, 60000 / bpm)) : 900;
  ui.widget.style.setProperty("--beat-ms", `${interval}ms`);
  ui.widget.style.setProperty("--heart-arc", `${arc}deg`);
}

function renderAgentSubtitle(agent) {
  if (!ui.agentSubtitle) return;
  const latest = agent.latest || {};
  const visible = Boolean(latest.visible && latest.subtitle);
  ui.agentSubtitle.textContent = visible ? latest.subtitle : "";
  ui.agentSubtitle.hidden = !visible;
  ui.agentSubtitle.classList.toggle("visible", visible);
}

async function render() {
  try {
    renderState(await readState());
  } catch (error) {
    ui.bpm.textContent = "--";
    ui.status.textContent = "服务离线";
    renderAgentSubtitle({});
    ui.widget.classList.remove("pulsing");
    ui.widget.style.opacity = "0.62";
    ui.widget.style.setProperty("--heart-arc", "0deg");
  }
}

window.setInterval(render, 500);
render();
