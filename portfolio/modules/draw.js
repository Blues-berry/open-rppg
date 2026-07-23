export function drawFaceOverlay(canvas, video, face) {
  const rect = canvas.getBoundingClientRect(); const dpr = window.devicePixelRatio || 1; const width = Math.max(1, rect.width), height = Math.max(1, rect.height);
  if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) { canvas.width = Math.round(width * dpr); canvas.height = Math.round(height * dpr); }
  const ctx = canvas.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, width, height);
  if (!face?.valid || !video.videoWidth) return;
  const scale = Math.max(width / video.videoWidth, height / video.videoHeight); const offsetX = (width - video.videoWidth * scale) / 2, offsetY = (height - video.videoHeight * scale) / 2; const map = (point) => ({ x: point.x * video.videoWidth * scale + offsetX, y: point.y * video.videoHeight * scale + offsetY });
  ctx.save(); ctx.lineWidth = 1.2; ctx.strokeStyle = "rgba(109, 255, 214, .72)"; ctx.shadowColor = "#54f7c5"; ctx.shadowBlur = 8; path(ctx, face.outline.map(map)); ctx.stroke();
  face.rois.forEach((roi, index) => { const points = roi.map(map); ctx.fillStyle = index === 0 ? "rgba(112, 255, 206, .12)" : "rgba(99, 194, 255, .1)"; ctx.strokeStyle = index === 0 ? "rgba(130, 255, 214, .92)" : "rgba(114, 211, 255, .8)"; path(ctx, points); ctx.fill(); ctx.stroke(); const p = points[0]; ctx.fillStyle = "#cffff1"; ctx.font = "10px ui-monospace, monospace"; ctx.fillText(["FOREHEAD", "LEFT CHEEK", "RIGHT CHEEK"][index], p.x + 4, p.y - 7); });
  ctx.fillStyle = "#d7fff4"; ctx.shadowBlur = 4; face.outline.filter((_, index) => index % 4 === 0).forEach((point) => { const p = map(point); ctx.beginPath(); ctx.arc(p.x, p.y, 1.7, 0, Math.PI * 2); ctx.fill(); }); ctx.restore();
}

export function drawWaveform(canvas, values) { const { ctx, width, height } = setup(canvas); grid(ctx, width, height); if (!values?.length) return; const max = Math.max(...values.map((v) => Math.abs(v)), 1); ctx.beginPath(); values.forEach((value, index) => { const x = (index / Math.max(1, values.length - 1)) * width, y = height / 2 - (value / max) * height * 0.34; index ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }); ctx.strokeStyle = "#67f3c2"; ctx.lineWidth = 1.7; ctx.shadowColor = "#49e8b2"; ctx.shadowBlur = 7; ctx.stroke(); }
export function drawSpectrum(canvas, points, bpm = 0) { drawSpectrumFrame(canvas, sanitizeSpectrum(points), bpm); }

export class SpectrumAnimator {
  constructor(canvas, duration = 180) {
    this.canvas = canvas; this.duration = duration; this.current = []; this.target = [];
    this.from = []; this.currentBpm = 0; this.fromBpm = 0; this.targetBpm = 0;
    this.scale = 1e-9; this.frame = null; this.startedAt = 0;
  }

  update(points, bpm = 0) {
    const target = sanitizeSpectrum(points); if (!target.length) return false;
    const now = performance.now(); this.capture(now);
    const bpms = target.map((point) => point.bpm);
    this.from = this.current.length ? resampleSpectrum(this.current, bpms) : target.map((point) => ({ bpm: point.bpm, power: 0 }));
    this.target = target; this.fromBpm = this.currentBpm || bpm; this.targetBpm = Number(bpm) || 0; this.startedAt = now;
    const targetMax = Math.max(...target.map((point) => point.power), 1e-9);
    this.scale = targetMax >= this.scale ? targetMax : Math.max(targetMax, this.scale * .9);
    if (this.frame) cancelAnimationFrame(this.frame);
    this.frame = requestAnimationFrame((time) => this.tick(time)); return true;
  }

  tick(now) {
    const progress = Math.min(1, Math.max(0, (now - this.startedAt) / this.duration));
    const eased = 1 - Math.pow(1 - progress, 3);
    this.current = this.target.map((point, index) => ({ bpm: point.bpm, power: mix(this.from[index]?.power || 0, point.power, eased) }));
    this.currentBpm = mix(this.fromBpm, this.targetBpm, eased);
    drawSpectrumFrame(this.canvas, this.current, this.currentBpm, this.scale);
    if (progress < 1) this.frame = requestAnimationFrame((time) => this.tick(time)); else this.frame = null;
  }

  capture(now) {
    if (!this.frame || !this.target.length) return;
    const progress = Math.min(1, Math.max(0, (now - this.startedAt) / this.duration));
    const eased = 1 - Math.pow(1 - progress, 3);
    this.current = this.target.map((point, index) => ({ bpm: point.bpm, power: mix(this.from[index]?.power || 0, point.power, eased) }));
    this.currentBpm = mix(this.fromBpm, this.targetBpm, eased);
  }

  redraw() { drawSpectrumFrame(this.canvas, this.current, this.currentBpm, this.scale); }
  clear() { if (this.frame) cancelAnimationFrame(this.frame); this.frame = null; this.current = []; this.target = []; this.from = []; this.currentBpm = 0; this.targetBpm = 0; this.scale = 1e-9; drawSpectrumFrame(this.canvas, [], 0, this.scale); }
}

function drawSpectrumFrame(canvas, points, bpm = 0, scale = 0) {
  const { ctx, width, height } = setup(canvas); grid(ctx, width, height); if (!points.length) return;
  const max = Math.max(scale, ...points.map((point) => point.power), 1e-9);
  const barWidth = Math.max(1, width / Math.max(1, points.length) - 1);
  ctx.fillStyle = "rgba(103, 243, 194, .48)";
  points.forEach((point) => { const x = ((point.bpm - 42) / 138) * width, barHeight = Math.max(0, point.power / max) * (height - 14); if (x >= 0 && x <= width) ctx.fillRect(x, height - barHeight, barWidth, barHeight); });
  if (bpm >= 42 && bpm <= 180) { const x = ((bpm - 42) / 138) * width; ctx.strokeStyle = "#f3d56b"; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, height); ctx.stroke(); }
}

function sanitizeSpectrum(points) { return (Array.isArray(points) ? points : []).map((point) => ({ bpm: Number(point?.bpm), power: Math.max(0, Number(point?.power) || 0) })).filter((point) => Number.isFinite(point.bpm) && point.bpm >= 42 && point.bpm <= 180).sort((a, b) => a.bpm - b.bpm); }
function resampleSpectrum(points, bpms) { return bpms.map((bpm) => ({ bpm, power: samplePower(points, bpm) })); }
function samplePower(points, bpm) { if (!points.length) return 0; if (bpm <= points[0].bpm) return points[0].power; for (let i = 1; i < points.length; i += 1) { if (bpm <= points[i].bpm) { const left = points[i - 1], right = points[i], span = right.bpm - left.bpm || 1; return mix(left.power, right.power, (bpm - left.bpm) / span); } } return points[points.length - 1].power; }
function mix(from, to, amount) { return from + (to - from) * amount; }
export const __test__ = { sanitizeSpectrum, resampleSpectrum, samplePower, mix };
function setup(canvas) {
  const rect = canvas.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
  const width = Math.max(1, rect.width), height = Math.max(1, rect.height);
  const pixelWidth = Math.round(width * dpr), pixelHeight = Math.round(height * dpr);
  // Reassigning canvas.width/height clears the backing store. The spectrum is
  // refreshed frequently, so resizing it on every frame caused visible flashes.
  if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
    canvas.width = pixelWidth; canvas.height = pixelHeight;
  }
  const ctx = canvas.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, width, height);
  return { ctx, width, height };
}
function grid(ctx, width, height) { ctx.strokeStyle = "rgba(175, 255, 225, .08)"; ctx.lineWidth = 1; for (let i = 1; i < 4; i += 1) { const y = (height / 4) * i; ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke(); } }
function path(ctx, points) { ctx.beginPath(); points.forEach((point, index) => index ? ctx.lineTo(point.x, point.y) : ctx.moveTo(point.x, point.y)); ctx.closePath(); }
