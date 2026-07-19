import { UI_IDS } from "./config.js?v=20260719-live-v4";
import { clamp } from "./format.js?v=20260719-live-v4";

const missingUiIds = [];
const ui = Object.fromEntries(
  Object.entries(UI_IDS).map(([key, id]) => {
    const element = document.getElementById(id);
    if (!element) missingUiIds.push(id);
    return [key, element];
  }),
);

export function setText(element, value) {
  if (element) element.textContent = value == null ? "" : String(value);
}

export function setHidden(element, hidden) {
  if (!element) return;
  element.hidden = Boolean(hidden);
  element.setAttribute("aria-hidden", hidden ? "true" : "false");
}

export function setDisabled(element, disabled) {
  if (element) element.disabled = Boolean(disabled);
}

export function setChecked(element, checked) {
  if (element) element.checked = Boolean(checked);
}

export function setValue(element, value) {
  if (element && value != null) element.value = value;
}

export function setMeter(element, value) {
  if (element) element.value = Math.round(clamp(Number(value || 0)) * 100);
}

export function setClassName(element, className) {
  if (element) element.className = className;
}

export function setStyleProp(element, property, value) {
  if (element) element.style.setProperty(property, value);
}

export function toggleClass(element, className, enabled) {
  if (element) element.classList.toggle(className, Boolean(enabled));
}

export function setDatasetFlag(element, key, enabled) {
  if (!element) return;
  if (enabled) element.dataset[key] = "true";
  else delete element.dataset[key];
}

export function on(element, eventName, handler) {
  if (element) element.addEventListener(eventName, handler);
}

export function numberValue(element, fallback = 0) {
  const value = Number(element?.value);
  return Number.isFinite(value) ? value : fallback;
}

export function checkedValue(element, fallback = false) {
  return element ? Boolean(element.checked) : fallback;
}

export function setStatePill(label, kind) {
  setText(ui.captureState, label);
  setClassName(ui.captureState, `state-pill ${kind}`);
}

export function showBanner(element, message, kind = "warn") {
  if (!element) return;
  element.dataset.kind = kind;
  setText(element, message);
  setHidden(element, false);
}

export function hideBanner(element) {
  setHidden(element, true);
}

export function validateDom() {
  if (Array.isArray(missingUiIds) && missingUiIds.length === 0) {
    hideBanner(ui.frontendWarning);
    return true;
  }
  const message = `前端渲染错误：缺少页面节点 ${missingUiIds.join(", ")}。请刷新静态文件或检查 index.html/app.js 是否同步。`;
  showBanner(ui.frontendWarning, message, "bad");
  setStatePill("FRONTEND ERR", "bad");
  setText(ui.heartTitle, "前端渲染错误");
  setText(ui.heartDescription, message);
  console.error(message);
  return false;
}

export { missingUiIds, ui };
