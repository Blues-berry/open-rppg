function number(value, fallback) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function settingsFrom(input = {}) {
  return {
    pulse: input.pulse !== false,
    light_enabled: Boolean(input.light_enabled),
    brightness: number(input.brightness, 72),
    temperature: number(input.temperature, 4800),
    light_x: number(input.light_x, 50),
    light_y: number(input.light_y, 38),
    light_z: number(input.light_z, 45),
    light_range: number(input.light_range, 58),
    light_angle_enabled: Boolean(input.light_angle_enabled),
    light_angle: number(input.light_angle, 0),
    light_revision: Date.now(),
  };
}

function snapshot({ running = false, settings = {} } = {}) {
  const now = Date.now() / 1000;
  const bpm = 70 + Math.round(Math.sin(now * 0.8) * 3);
  const sqi = Number((0.77 + Math.sin(now * 0.45) * 0.06).toFixed(2));
  const bvp = Array.from({ length: 120 }, (_, index) => Math.sin(index * 0.33 + now * 2.1) * .8 + Math.sin(index * .09) * .18);
  const normalized = settingsFrom(settings);
  return {
    simulated: true,
    capture: { state: running ? "running" : "idle", device_index: "browser-preview", width: 1280, height: 720, input_fps: running ? 29.8 : 0, target_fps: 30, read_ms: running ? 4.2 : 0, frames_read: running ? Math.round(now * 30) : 0, dropped_frames: 0 },
    model: { ready: true, model: "Simulated FacePhys", hr: running ? bpm : null, SQI: running ? sqi : 0, input_fps: running ? 15 : 0, has_face: running, no_face_count: 0, hr_window_seconds: 10, perf: { update_ms: 13, metric_ms: 4 }, waveform: { bvp, ts: bvp.map((_, index) => now - 8 + index * (8 / bvp.length)) } },
    output: { bpm: running ? bpm : null, confidence: running ? sqi : 0, status: running ? "stable" : "idle", reason: "browser_preview_simulation" },
    settings: normalized,
    agent: { configured: false, history: [], latest: {} },
    highlights: { recording: { enabled: false, state: "unavailable" }, items: [] },
  };
}

module.exports = function handler(request, response) {
  if (request.method !== "POST") return response.status(405).json({ error: "POST only" });
  const body = request.body || {};
  const action = body.action || "state";
  if (!["state", "start", "stop", "reset", "settings"].includes(action)) return response.status(400).json({ error: "Unsupported simulation action" });
  const running = action === "start" ? true : action === "stop" || action === "reset" ? false : Boolean(body.running);
  return response.status(200).json({ ok: true, simulated: true, snapshot: snapshot({ running, settings: body.settings }) });
}
