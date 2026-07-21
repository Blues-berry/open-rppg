const WINDOW_SECONDS = 20;
const WARMUP_SECONDS = 12;
const TARGET_RATE = 30;

export class SignalProcessor {
  constructor() { this.samples = []; this.lastBpm = 0; }
  get count() { return this.samples.length; }
  get durationSeconds() { return this.samples.length > 1 ? (this.samples.at(-1).t - this.samples[0].t) / 1000 : 0; }
  reset() { this.samples.length = 0; this.lastBpm = 0; }
  push(sample) { if (!Number.isFinite(sample.t)) return; this.samples.push(sample); const cutoff = sample.t - WINDOW_SECONDS * 1000; while (this.samples.length && this.samples[0].t < cutoff) this.samples.shift(); }
  analyze() {
    const duration = this.durationSeconds;
    if (duration < 2 || this.samples.length < 45) return empty(duration);
    const rgb = resample(this.samples, TARGET_RATE); const pos = posSignal(rgb); const wave = normalize(bandpassApprox(pos, TARGET_RATE)); const spectrum = frequencyScan(wave, TARGET_RATE);
    const best = choosePulsePeak(spectrum);
    const noise = spectrum.filter((point) => Math.abs(point.bpm - best.bpm) > 5).map((point) => point.power); const noiseFloor = noise.reduce((sum, value) => sum + value, 0) / Math.max(1, noise.length);
    const snr = best.power / Math.max(noiseFloor, 1e-9); const continuity = this.lastBpm ? Math.exp(-Math.abs(best.bpm - this.lastBpm) / 18) : 0.7; const sampleQuality = Math.min(1, rgb.length / (WARMUP_SECONDS * TARGET_RATE));
    const signalQuality = clamp((Math.log1p(snr) / Math.log(9)) * 0.7 + sampleQuality * 0.3); const sqi = clamp(signalQuality * 0.76 + continuity * 0.24);
    const bpm = duration >= WARMUP_SECONDS && signalQuality >= 0.34 ? best.bpm : 0;
    if (bpm) this.lastBpm = this.lastBpm ? this.lastBpm * 0.65 + bpm * 0.35 : bpm;
    return { duration, waveform: wave.slice(-360), spectrum, bpm: bpm ? this.lastBpm : 0, signalQuality, sqi };
  }
}

function empty(duration) { return { duration, waveform: [], spectrum: [], bpm: 0, signalQuality: 0, sqi: 0 }; }
function choosePulsePeak(points) { const peak = points.reduce((winner, point) => point.power > winner.power ? point : winner, { bpm: 0, power: 0 }); if (peak.bpm < 86) return peak; const half = points.find((point) => point.bpm === Math.round(peak.bpm / 2)); return half && half.power >= peak.power * 0.35 ? half : peak; }
function resample(samples, rate) { const start = samples[0].t, end = samples.at(-1).t, step = 1000 / rate, out = []; let pointer = 0; for (let t = start; t <= end; t += step) { while (pointer < samples.length - 2 && samples[pointer + 1].t < t) pointer += 1; const a = samples[pointer], b = samples[Math.min(pointer + 1, samples.length - 1)], ratio = b.t === a.t ? 0 : (t - a.t) / (b.t - a.t); out.push({ r: a.r + (b.r - a.r) * ratio, g: a.g + (b.g - a.g) * ratio, b: a.b + (b.b - a.b) * ratio }); } return out; }
function posSignal(rgb) { const means = ["r", "g", "b"].map((key) => rgb.reduce((sum, sample) => sum + sample[key], 0) / rgb.length); const s1 = rgb.map((sample) => sample.g / means[1] - sample.b / means[2]); const s2 = rgb.map((sample) => sample.g / means[1] + sample.b / means[2] - 2 * sample.r / means[0]); const alpha = std(s1) / Math.max(std(s2), 1e-6); return s1.map((value, index) => value - alpha * s2[index]); }
function bandpassApprox(values, rate) { const trend = movingAverage(values, Math.max(3, Math.round(rate * 0.9))); const high = values.map((value, index) => value - trend[index]); return movingAverage(high, Math.max(2, Math.round(rate * 0.1))); }
function movingAverage(values, span) { return values.map((_, index) => { const lo = Math.max(0, index - Math.floor(span / 2)), hi = Math.min(values.length, index + Math.ceil(span / 2)); let sum = 0; for (let i = lo; i < hi; i += 1) sum += values[i]; return sum / Math.max(1, hi - lo); }); }
function frequencyScan(values, rate) { const n = values.length; return Array.from({ length: 139 }, (_, index) => { const bpm = index + 42, frequency = bpm / 60; let re = 0, im = 0; for (let i = 0; i < n; i += 1) { const window = 0.54 - 0.46 * Math.cos((2 * Math.PI * i) / Math.max(1, n - 1)); const angle = (2 * Math.PI * frequency * i) / rate; re += values[i] * window * Math.cos(angle); im -= values[i] * window * Math.sin(angle); } return { bpm, power: re * re + im * im }; }); }
function normalize(values) { const mean = values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length); const deviation = std(values) || 1; return values.map((value) => (value - mean) / deviation); }
function std(values) { const mean = values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length); return Math.sqrt(values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / Math.max(1, values.length)); }
function clamp(value) { return Math.min(1, Math.max(0, Number.isFinite(value) ? value : 0)); }
