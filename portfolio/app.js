(() => {
  const camera = document.querySelector('#camera');
  const canvas = document.querySelector('#frameCanvas');
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  const wave = document.querySelector('#waveCanvas');
  const waveCtx = wave.getContext('2d');
  const startButton = document.querySelector('#startButton');
  const empty = document.querySelector('#cameraEmpty');
  const roi = document.querySelector('#roi');
  const bpmValue = document.querySelector('#bpmValue');
  const qualityValue = document.querySelector('#qualityValue');
  const qualityBar = document.querySelector('#qualityBar');
  const windowValue = document.querySelector('#windowValue');
  const statusChip = document.querySelector('#statusChip');
  const hint = document.querySelector('#demoHint');
  const cameraStatus = document.querySelector('#cameraStatus');
  const SAMPLE_RATE = 30;
  const WINDOW_SECONDS = 20;
  const samples = [];
  let stream, timer, lastAnalysis = 0;

  const setState = (state, message) => { statusChip.textContent = state; hint.textContent = message; };
  const stop = () => {
    clearInterval(timer); timer = null;
    if (stream) stream.getTracks().forEach(track => track.stop());
    stream = null; camera.srcObject = null; samples.length = 0;
    empty.hidden = false; roi.hidden = true; startButton.textContent = '开启摄像头 →'; startButton.classList.remove('stop');
    bpmValue.textContent = '--'; qualityValue.textContent = '--'; qualityBar.style.width = '0%'; windowValue.textContent = '窗口 0.0 / 20s'; cameraStatus.textContent = 'STANDBY';
    setState('等待开始', '请保持脸部居中、光线稳定'); drawWave([]);
  };
  const start = async () => {
    try {
      setState('正在请求权限', '浏览器将询问摄像头权限');
      stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: SAMPLE_RATE } }, audio: false });
      camera.srcObject = stream; await camera.play();
      empty.hidden = true; roi.hidden = false; startButton.textContent = '停止采集 ×'; startButton.classList.add('stop'); cameraStatus.textContent = 'LOCAL LIVE';
      setState('采集中', '正在建立 20 秒色彩信号窗口');
      timer = setInterval(sample, 1000 / SAMPLE_RATE);
    } catch (error) {
      setState('无法开启摄像头', error.name === 'NotAllowedError' ? '请在浏览器设置中允许此站点使用摄像头' : '请确认设备存在可用摄像头');
      cameraStatus.textContent = 'CAMERA ERROR';
    }
  };
  const sample = () => {
    if (!camera.videoWidth) return;
    const w = 80, h = 80; canvas.width = w; canvas.height = h;
    const side = Math.min(camera.videoWidth, camera.videoHeight) * .42;
    const sx = (camera.videoWidth - side) / 2, sy = camera.videoHeight * .25;
    ctx.drawImage(camera, sx, sy, side, side, 0, 0, w, h);
    const pixels = ctx.getImageData(0, 0, w, h).data;
    let sum = 0, count = 0;
    for (let y = 14; y < 68; y++) for (let x = 12; x < 68; x++) { const i = (y * w + x) * 4; const r = pixels[i], g = pixels[i + 1], b = pixels[i + 2]; if (g > 22 && r > 18 && b > 12 && Math.max(r,g,b) - Math.min(r,g,b) > 4) { sum += g; count++; } }
    if (!count) return;
    samples.push({ t: performance.now(), v: sum / count });
    const cutoff = performance.now() - WINDOW_SECONDS * 1000;
    while (samples.length && samples[0].t < cutoff) samples.shift();
    windowValue.textContent = `窗口 ${Math.min(WINDOW_SECONDS, samples.length / SAMPLE_RATE).toFixed(1)} / ${WINDOW_SECONDS}s`;
    drawWave(samples.map(s => s.v));
    if (samples.length > SAMPLE_RATE * 10 && performance.now() - lastAnalysis > 1000) { lastAnalysis = performance.now(); analyze(); }
  };
  const analyze = () => {
    const values = samples.map(s => s.v); const n = values.length;
    const mean = values.reduce((a,b) => a + b, 0) / n;
    const detrended = values.map((v,i) => v - mean - ((values[n - 1] - values[0]) / n) * (i - n / 2));
    const filtered = detrended.map((_, i) => { let total = 0, count = 0; for (let k = Math.max(0, i - 2); k <= Math.min(n - 1, i + 2); k++) { total += detrended[k]; count++; } return detrended[i] - total / count; });
    const hz = SAMPLE_RATE; let best = { power: 0, bpm: 0 }, totalPower = 0;
    for (let bpm = 42; bpm <= 180; bpm++) { const f = bpm / 60; let re = 0, im = 0; for (let i = 0; i < n; i++) { const a = 2 * Math.PI * f * i / hz; re += filtered[i] * Math.cos(a); im -= filtered[i] * Math.sin(a); } const power = re * re + im * im; totalPower += power; if (power > best.power) best = { power, bpm }; }
    const quality = Math.min(.96, Math.max(0, (best.power / Math.max(totalPower, 1)) * 9));
    qualityValue.textContent = quality.toFixed(2); qualityBar.style.width = `${quality * 100}%`;
    if (quality >= .16) { bpmValue.textContent = best.bpm; setState(quality > .35 ? '信号可用' : '低置信度', quality > .35 ? '检测到稳定的周期性颜色变化' : '请保持静止并改善光线'); }
    else { bpmValue.textContent = '--'; setState('建立信号中', '请保持面部居中，至少等待 10 秒'); }
  };
  const drawWave = values => {
    const rect = wave.getBoundingClientRect(), dpr = window.devicePixelRatio || 1; wave.width = rect.width * dpr; wave.height = rect.height * dpr; waveCtx.scale(dpr, dpr); const w = rect.width, h = rect.height;
    waveCtx.clearRect(0, 0, w, h); waveCtx.strokeStyle = 'rgba(230,230,223,.1)'; waveCtx.lineWidth = 1; for(let y=1;y<4;y++){waveCtx.beginPath();waveCtx.moveTo(0,h*y/4);waveCtx.lineTo(w,h*y/4);waveCtx.stroke()}
    if (values.length < 2) return; const mean = values.reduce((a,b)=>a+b,0)/values.length; const max = Math.max(...values.map(v=>Math.abs(v-mean)), 1);
    waveCtx.beginPath(); values.forEach((v,i) => { const x = i/(Math.max(values.length-1,1))*w; const y = h/2 - ((v-mean)/max)*h*.36; i ? waveCtx.lineTo(x,y) : waveCtx.moveTo(x,y); }); waveCtx.strokeStyle='#d2ff3f'; waveCtx.lineWidth=1.5; waveCtx.stroke();
  };
  startButton.addEventListener('click', () => stream ? stop() : start()); window.addEventListener('resize', () => drawWave(samples.map(s => s.v))); drawWave([]);
})();
