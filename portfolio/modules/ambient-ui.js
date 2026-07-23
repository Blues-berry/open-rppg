import { Renderer, Program, Mesh, Triangle } from "../assets/vendor/ogl/ogl.mjs";

// Line Waves shader adapted from React Bits for this static, non-React site.
const VERTEX_SHADER = `
attribute vec2 uv;
attribute vec2 position;
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = vec4(position, 0, 1);
}
`;

const FRAGMENT_SHADER = `
precision highp float;

uniform float uTime;
uniform vec3 uResolution;
uniform float uSpeed;
uniform float uInnerLines;
uniform float uOuterLines;
uniform float uWarpIntensity;
uniform float uRotation;
uniform float uEdgeFadeWidth;
uniform float uColorCycleSpeed;
uniform float uBrightness;
uniform vec3 uColor1;
uniform vec3 uColor2;
uniform vec3 uColor3;
uniform vec2 uMouse;
uniform float uMouseInfluence;
uniform bool uEnableMouse;

#define HALF_PI 1.5707963

float hashF(float n) {
  return fract(sin(n * 127.1) * 43758.5453123);
}

float smoothNoise(float x) {
  float i = floor(x);
  float f = fract(x);
  float u = f * f * (3.0 - 2.0 * f);
  return mix(hashF(i), hashF(i + 1.0), u);
}

float displaceA(float coord, float t) {
  float result = sin(coord * 2.123) * 0.2;
  result += sin(coord * 3.234 + t * 4.345) * 0.1;
  result += sin(coord * 0.589 + t * 0.934) * 0.5;
  return result;
}

float displaceB(float coord, float t) {
  float result = sin(coord * 1.345) * 0.3;
  result += sin(coord * 2.734 + t * 3.345) * 0.2;
  result += sin(coord * 0.189 + t * 0.934) * 0.3;
  return result;
}

vec2 rotate2D(vec2 point, float angle) {
  float c = cos(angle);
  float s = sin(angle);
  return vec2(point.x * c - point.y * s, point.x * s + point.y * c);
}

void main() {
  vec2 coords = gl_FragCoord.xy / uResolution.xy;
  coords = rotate2D(coords * 2.0 - 1.0, uRotation);

  float halfT = uTime * uSpeed * 0.5;
  float fullT = uTime * uSpeed;
  float mouseWarp = 0.0;

  if (uEnableMouse) {
    vec2 mousePosition = rotate2D(uMouse * 2.0 - 1.0, uRotation);
    float mouseDistance = length(coords - mousePosition);
    mouseWarp = uMouseInfluence * exp(-mouseDistance * mouseDistance * 4.0);
  }

  float warpAx = coords.x + displaceA(coords.y, halfT) * uWarpIntensity + mouseWarp;
  float warpAy = coords.y - displaceA(coords.x * cos(fullT) * 1.235, halfT) * uWarpIntensity;
  float warpBx = coords.x + displaceB(coords.y, halfT) * uWarpIntensity + mouseWarp;
  float warpBy = coords.y - displaceB(coords.x * sin(fullT) * 1.235, halfT) * uWarpIntensity;

  vec2 fieldA = vec2(warpAx, warpAy);
  vec2 fieldB = vec2(warpBx, warpBy);
  vec2 blended = mix(fieldA, fieldB, mix(fieldA, fieldB, 0.5));

  float fadeTop = smoothstep(uEdgeFadeWidth, uEdgeFadeWidth + 0.4, blended.y);
  float fadeBottom = smoothstep(-uEdgeFadeWidth, -(uEdgeFadeWidth + 0.4), blended.y);
  float verticalMask = 1.0 - max(fadeTop, fadeBottom);

  float tileCount = mix(uOuterLines, uInnerLines, verticalMask);
  float scaledY = blended.y * tileCount;
  float noiseY = smoothNoise(abs(scaledY));
  float ridge = pow(step(abs(noiseY - blended.x) * 2.0, HALF_PI) * cos(2.0 * (noiseY - blended.x)), 5.0);

  float lines = 0.0;
  for (float i = 1.0; i < 3.0; i += 1.0) {
    lines += pow(max(fract(scaledY), fract(-scaledY)), i * 2.0);
  }

  float pattern = verticalMask * lines;
  float cycleT = fullT * uColorCycleSpeed;
  float red = (pattern + lines * ridge) * (cos(blended.y + cycleT * 0.234) * 0.5 + 1.0);
  float green = (pattern + verticalMask * ridge) * (sin(blended.x + cycleT * 1.745) * 0.5 + 1.0);
  float blue = (pattern + lines * ridge) * (cos(blended.x + cycleT * 0.534) * 0.5 + 1.0);

  vec3 color = (red * uColor1 + green * uColor2 + blue * uColor3) * uBrightness;
  gl_FragColor = vec4(color, clamp(length(color), 0.0, 1.0));
}
`;

const clamp = (value, min = 0, max = 1) => Math.min(max, Math.max(min, value));

function hexToVec3(hex) {
  const value = hex.replace("#", "");
  return [
    Number.parseInt(value.slice(0, 2), 16) / 255,
    Number.parseInt(value.slice(2, 4), 16) / 255,
    Number.parseInt(value.slice(4, 6), 16) / 255,
  ];
}

function addMediaListener(query, listener) {
  if (query.addEventListener) query.addEventListener("change", listener);
  else query.addListener(listener);
  return () => {
    if (query.removeEventListener) query.removeEventListener("change", listener);
    else query.removeListener(listener);
  };
}

class LineWavesController {
  constructor(container, hero, reducedMotion, finePointer) {
    this.container = container;
    this.hero = hero;
    this.reducedMotion = reducedMotion;
    this.finePointer = finePointer;
    this.captureActive = false;
    this.heroVisible = true;
    this.destroyed = false;
    this.frame = 0;
    this.lastFrameAt = 0;
    this.elapsed = 0;
    this.currentMouse = [0.5, 0.5];
    this.targetMouse = [0.5, 0.5];
    this.frameInterval = matchMedia("(max-width: 700px)").matches ? 50 : 1000 / 30;

    const dpr = Math.min(window.devicePixelRatio || 1, 1.25);
    this.renderer = new Renderer({ alpha: true, premultipliedAlpha: false, dpr });
    this.gl = this.renderer.gl;
    this.gl.clearColor(0, 0, 0, 0);
    this.gl.canvas.setAttribute("aria-hidden", "true");

    const geometry = new Triangle(this.gl);
    this.program = new Program(this.gl, {
      vertex: VERTEX_SHADER,
      fragment: FRAGMENT_SHADER,
      uniforms: {
        uTime: { value: 0 },
        uResolution: { value: [1, 1, 1] },
        uSpeed: { value: 0.16 },
        uInnerLines: { value: 24 },
        uOuterLines: { value: 30 },
        uWarpIntensity: { value: 0.72 },
        uRotation: { value: (-35 * Math.PI) / 180 },
        uEdgeFadeWidth: { value: -0.05 },
        uColorCycleSpeed: { value: 0.24 },
        uBrightness: { value: 0.15 },
        uColor1: { value: hexToVec3("#65efc1") },
        uColor2: { value: hexToVec3("#78cfff") },
        uColor3: { value: hexToVec3("#edf8f5") },
        uMouse: { value: new Float32Array([0.5, 0.5]) },
        uMouseInfluence: { value: 0.45 },
        uEnableMouse: { value: finePointer.matches },
      },
    });
    this.mesh = new Mesh(this.gl, { geometry, program: this.program });
    this.container.appendChild(this.gl.canvas);

    this.handlePointerMove = this.handlePointerMove.bind(this);
    this.handlePointerLeave = this.handlePointerLeave.bind(this);
    this.handleVisibility = this.sync.bind(this);
    this.handleMotionChange = this.sync.bind(this);
    this.handlePointerChange = this.updatePointerMode.bind(this);
    this.tick = this.tick.bind(this);

    this.hero.addEventListener("pointermove", this.handlePointerMove, { passive: true });
    this.hero.addEventListener("pointerleave", this.handlePointerLeave, { passive: true });
    document.addEventListener("visibilitychange", this.handleVisibility);
    this.removeMotionListener = addMediaListener(reducedMotion, this.handleMotionChange);
    this.removePointerListener = addMediaListener(finePointer, this.handlePointerChange);

    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(this.container);
    this.intersectionObserver = new IntersectionObserver(
      ([entry]) => {
        this.heroVisible = Boolean(entry?.isIntersecting);
        this.sync();
      },
      { rootMargin: "140px 0px" },
    );
    this.intersectionObserver.observe(this.hero);

    this.resize();
    this.updatePointerMode();
    this.sync();
  }

  shouldRun() {
    return !this.destroyed && !this.captureActive && this.heroVisible && !document.hidden && !this.reducedMotion.matches;
  }

  sync() {
    this.container.classList.toggle("is-static", this.reducedMotion.matches);
    if (this.shouldRun() && !this.frame) {
      this.lastFrameAt = 0;
      this.frame = requestAnimationFrame(this.tick);
    } else if (!this.shouldRun() && this.frame) {
      cancelAnimationFrame(this.frame);
      this.frame = 0;
      this.lastFrameAt = 0;
    }
  }

  resize() {
    const width = Math.max(1, Math.round(this.container.clientWidth));
    const height = Math.max(1, Math.round(this.container.clientHeight));
    this.renderer.setSize(width, height);
    this.program.uniforms.uResolution.value = [
      this.gl.canvas.width,
      this.gl.canvas.height,
      this.gl.canvas.width / Math.max(1, this.gl.canvas.height),
    ];
  }

  updatePointerMode() {
    this.program.uniforms.uEnableMouse.value = this.finePointer.matches && !this.captureActive;
    if (!this.program.uniforms.uEnableMouse.value) this.handlePointerLeave();
  }

  handlePointerMove(event) {
    if (!this.finePointer.matches || this.captureActive) return;
    const rect = this.container.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    this.targetMouse = [
      clamp((event.clientX - rect.left) / rect.width),
      1 - clamp((event.clientY - rect.top) / rect.height),
    ];
  }

  handlePointerLeave() {
    this.targetMouse = [0.5, 0.5];
  }

  tick(time) {
    this.frame = 0;
    if (!this.shouldRun()) return;
    if (this.lastFrameAt && time - this.lastFrameAt < this.frameInterval) {
      this.frame = requestAnimationFrame(this.tick);
      return;
    }

    const delta = this.lastFrameAt ? Math.min((time - this.lastFrameAt) / 1000, 0.1) : 0;
    this.lastFrameAt = time;
    this.elapsed += delta;
    this.program.uniforms.uTime.value = this.elapsed;

    const mouseEnabled = this.finePointer.matches && !this.captureActive;
    this.currentMouse[0] += 0.045 * (this.targetMouse[0] - this.currentMouse[0]);
    this.currentMouse[1] += 0.045 * (this.targetMouse[1] - this.currentMouse[1]);
    this.program.uniforms.uMouse.value[0] = mouseEnabled ? this.currentMouse[0] : 0.5;
    this.program.uniforms.uMouse.value[1] = mouseEnabled ? this.currentMouse[1] : 0.5;
    this.renderer.render({ scene: this.mesh });
    this.frame = requestAnimationFrame(this.tick);
  }

  setCaptureActive(active) {
    this.captureActive = Boolean(active);
    this.updatePointerMode();
    this.sync();
  }

  getDebugState() {
    return {
      running: Boolean(this.frame),
      captureActive: this.captureActive,
      heroVisible: this.heroVisible,
      reducedMotion: this.reducedMotion.matches,
      frameInterval: this.frameInterval,
    };
  }

  destroy() {
    this.destroyed = true;
    if (this.frame) cancelAnimationFrame(this.frame);
    this.frame = 0;
    this.resizeObserver.disconnect();
    this.intersectionObserver.disconnect();
    this.hero.removeEventListener("pointermove", this.handlePointerMove);
    this.hero.removeEventListener("pointerleave", this.handlePointerLeave);
    document.removeEventListener("visibilitychange", this.handleVisibility);
    this.removeMotionListener();
    this.removePointerListener();
    this.gl.canvas.remove();
    this.gl.getExtension("WEBGL_lose_context")?.loseContext();
  }
}

function initPipelineHover(pipeline, finePointer, reducedMotion) {
  const marker = pipeline?.querySelector(".pipeline-focus");
  const cards = pipeline ? [...pipeline.querySelectorAll("article")] : [];
  if (!pipeline || !marker || !cards.length) return { destroy() {} };

  let activeCard = null;
  const showCard = (card) => {
    if (!card) return;
    activeCard = card;
    const parent = pipeline.getBoundingClientRect();
    const bounds = card.getBoundingClientRect();
    marker.style.setProperty("--focus-x", `${bounds.left - parent.left}px`);
    marker.style.setProperty("--focus-y", `${bounds.top - parent.top}px`);
    marker.style.setProperty("--focus-w", `${bounds.width}px`);
    marker.style.setProperty("--focus-h", `${bounds.height}px`);
    marker.classList.add("is-visible");
  };
  const hide = () => {
    if (pipeline.contains(document.activeElement)) return;
    activeCard = null;
    marker.classList.remove("is-visible");
  };
  const handlePointerOver = (event) => {
    if (!finePointer.matches || reducedMotion.matches) return;
    showCard(event.target.closest("article"));
  };
  const handleFocusIn = (event) => showCard(event.target.closest("article"));
  const handleFocusOut = () => requestAnimationFrame(hide);
  const handleResize = () => activeCard && showCard(activeCard);

  pipeline.addEventListener("pointerover", handlePointerOver, { passive: true });
  pipeline.addEventListener("pointerleave", hide, { passive: true });
  pipeline.addEventListener("focusin", handleFocusIn);
  pipeline.addEventListener("focusout", handleFocusOut);
  window.addEventListener("resize", handleResize);

  return {
    destroy() {
      pipeline.removeEventListener("pointerover", handlePointerOver);
      pipeline.removeEventListener("pointerleave", hide);
      pipeline.removeEventListener("focusin", handleFocusIn);
      pipeline.removeEventListener("focusout", handleFocusOut);
      window.removeEventListener("resize", handleResize);
    },
  };
}

function initBorderGlow(lab, finePointer, reducedMotion) {
  if (!lab) return { setCaptureActive() {}, destroy() {} };
  let captureActive = false;

  const reset = () => {
    lab.style.setProperty("--glow-x", "50%");
    lab.style.setProperty("--glow-y", "50%");
    lab.classList.remove("has-pointer-glow");
  };
  const handlePointerMove = (event) => {
    if (!finePointer.matches || reducedMotion.matches || captureActive) return;
    const bounds = lab.getBoundingClientRect();
    lab.style.setProperty("--glow-x", `${event.clientX - bounds.left}px`);
    lab.style.setProperty("--glow-y", `${event.clientY - bounds.top}px`);
    lab.classList.add("has-pointer-glow");
  };

  lab.addEventListener("pointermove", handlePointerMove, { passive: true });
  lab.addEventListener("pointerleave", reset, { passive: true });
  reset();

  return {
    setCaptureActive(active) {
      captureActive = Boolean(active);
      lab.classList.toggle("is-capturing", captureActive);
      if (captureActive) reset();
    },
    destroy() {
      lab.removeEventListener("pointermove", handlePointerMove);
      lab.removeEventListener("pointerleave", reset);
    },
  };
}

function initReveals(reducedMotion) {
  if (reducedMotion.matches || !("IntersectionObserver" in window)) return { destroy() {} };
  const selectors = [
    ".hero-copy",
    ".hero-art",
    ".metrics",
    ".section-heading",
    ".pipeline article",
    ".materials .deck-card",
    ".workflow-card",
  ];
  const items = [...document.querySelectorAll(selectors.join(","))];
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        const item = entry.target;
        const order = Number(item.dataset.revealOrder || 0);
        item.animate(
          [
            { opacity: 0.01, transform: "translateY(18px)" },
            { opacity: 1, transform: "translateY(0)" },
          ],
          {
            duration: 620,
            delay: Math.min(order * 55, 165),
            easing: "cubic-bezier(.2,.72,.2,1)",
            fill: "both",
          },
        );
        observer.unobserve(item);
      });
    },
    { threshold: 0.12, rootMargin: "0px 0px -5% 0px" },
  );

  items.forEach((item, index) => {
    item.dataset.revealOrder = String(index % 4);
    observer.observe(item);
  });
  return { destroy: () => observer.disconnect() };
}

export function initAmbientUi() {
  const hero = document.querySelector(".hero");
  const lineWavesElement = document.getElementById("lineWaves");
  const lab = document.getElementById("pulseLab");
  const pipeline = document.querySelector(".pipeline");
  const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");
  const finePointer = matchMedia("(pointer: fine)");
  const controllers = [];
  let lineWaves = null;

  if (hero && lineWavesElement && !reducedMotion.matches) {
    try {
      lineWaves = new LineWavesController(lineWavesElement, hero, reducedMotion, finePointer);
      lineWavesElement.dataset.status = "ready";
      controllers.push(lineWaves);
    } catch (error) {
      lineWavesElement.dataset.status = "fallback";
      console.warn("Line Waves unavailable; using static hero background.", error);
    }
  } else if (lineWavesElement) {
    lineWavesElement.dataset.status = "static";
  }

  const pipelineHover = initPipelineHover(pipeline, finePointer, reducedMotion);
  const borderGlow = initBorderGlow(lab, finePointer, reducedMotion);
  const reveals = initReveals(reducedMotion);
  controllers.push(pipelineHover, borderGlow, reveals);

  let captureActive = false;
  const setCaptureActive = (active) => {
    captureActive = Boolean(active);
    document.documentElement.classList.toggle("is-capturing", captureActive);
    lineWaves?.setCaptureActive(captureActive);
    borderGlow.setCaptureActive(captureActive);
  };

  return {
    setCaptureActive,
    getDebugState() {
      return {
        captureActive,
        lineWaves: lineWaves?.getDebugState() || { running: false, fallback: true },
      };
    },
    destroy() {
      document.documentElement.classList.remove("is-capturing");
      controllers.reverse().forEach((controller) => controller.destroy());
    },
  };
}
