from __future__ import annotations

import json
import threading
import time
from collections import deque
from datetime import datetime
from urllib import error as urllib_error
from urllib import request as urllib_request

from .config import (
    AGENT_AUTO_COOLDOWN_SECONDS,
    AGENT_BPM_DELTA_TRIGGER,
    AGENT_DEFAULT_MODEL,
    AGENT_DEFAULT_PROTOCOL,
    AGENT_DEFAULT_VERSION,
    AGENT_HISTORY_LIMIT,
    AGENT_HTTP_TIMEOUT_SECONDS,
    AGENT_LOCAL_CONFIG_PATH,
    AGENT_LOG_DIR,
    AGENT_MAX_REPLY_CHARS,
    AGENT_MAX_SUBTITLE_CHARS,
    AGENT_MAX_USER_TEXT_CHARS,
    AGENT_RECENT_HEART_SECONDS,
    AGENT_SUBTITLE_TTL_SECONDS,
    OUTPUT_SQI_THRESHOLD,
    PREVIEW_SQI_THRESHOLD,
)
from .utils import agent_config_value, load_agent_config, safe_float, save_agent_config

class CompatibleClient:
    def __init__(self):
        config = load_agent_config()
        self.protocol = str(config.get("protocol") or AGENT_DEFAULT_PROTOCOL).strip().lower()
        if self.protocol not in {"anthropic", "openai"}:
            self.protocol = AGENT_DEFAULT_PROTOCOL
        self.base_url = agent_config_value(config, "ANTHROPIC_BASE_URL", "base_url").rstrip("/")
        self.auth_token = agent_config_value(config, "ANTHROPIC_AUTH_TOKEN", "auth_token")
        self.api_key = agent_config_value(config, "ANTHROPIC_API_KEY", "api_key")
        self.model = agent_config_value(config, "ANTHROPIC_MODEL", "model") or AGENT_DEFAULT_MODEL
        self.version = agent_config_value(config, "ANTHROPIC_VERSION", "version") or AGENT_DEFAULT_VERSION

    @property
    def configured(self):
        return bool(self.base_url and (self.auth_token or self.api_key))

    def public_status(self):
        return {
            "configured": self.configured,
            "protocol": self.protocol,
            "model": self.model,
            "has_api_key": bool(self.auth_token or self.api_key),
        }

    def configure(self, settings: dict):
        protocol = str(settings.get("protocol") or "").strip().lower()
        base_url = str(settings.get("base_url") or "").strip().rstrip("/")
        model = str(settings.get("model") or "").strip()
        supplied_key = str(settings.get("api_key") or "").strip()
        if protocol not in {"anthropic", "openai"}:
            raise ValueError("protocol must be anthropic or openai")
        if not base_url:
            raise ValueError("base_url is required")
        if not model:
            raise ValueError("model is required")
        api_key = supplied_key or self.api_key or self.auth_token
        if not api_key:
            raise ValueError("api_key is required")
        self.protocol = protocol
        self.base_url = base_url
        self.api_key = api_key
        self.auth_token = ""
        self.model = model
        save_agent_config({
            "protocol": self.protocol,
            "base_url": self.base_url,
            "api_key": self.api_key,
            "auth_token": "",
            "model": self.model,
            "version": self.version,
        })

    def create_message(self, system: str, messages: list[dict], max_tokens: int = 520):
        if not self.configured:
            raise RuntimeError("agent model is not configured")

        if self.protocol == "openai":
            payload = {
                "model": self.model,
                "max_tokens": max_tokens,
                "messages": [{"role": "system", "content": system}, *messages],
            }
        else:
            payload = {"model": self.model, "max_tokens": max_tokens, "system": system, "messages": messages}
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib_request.Request(
            self._messages_url(),
            data=data,
            method="POST",
            headers=self._headers(),
        )
        try:
            with urllib_request.urlopen(request, timeout=AGENT_HTTP_TIMEOUT_SECONDS) as response:
                response_data = response.read().decode("utf-8")
        except urllib_error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"agent API {exc.code}: {detail}") from exc
        except urllib_error.URLError as exc:
            raise RuntimeError(f"agent API unavailable: {exc.reason}") from exc

        try:
            parsed = json.loads(response_data)
        except json.JSONDecodeError as exc:
            raise RuntimeError("agent API returned non-JSON response") from exc
        text = self._extract_text(parsed)
        if not text:
            raise RuntimeError("agent API returned an empty response")
        return text

    def _messages_url(self):
        if self.protocol == "openai":
            if self.base_url.endswith("/v1/chat/completions"):
                return self.base_url
            if self.base_url.endswith("/v1"):
                return f"{self.base_url}/chat/completions"
            return f"{self.base_url}/v1/chat/completions"
        if self.base_url.endswith("/v1/messages"):
            return self.base_url
        if self.base_url.endswith("/v1"):
            return f"{self.base_url}/messages"
        return f"{self.base_url}/v1/messages"

    def _headers(self):
        headers = {"Content-Type": "application/json"}
        if self.protocol == "openai":
            headers["Authorization"] = f"Bearer {self.api_key or self.auth_token}"
        else:
            headers["anthropic-version"] = self.version
        if self.protocol != "openai" and self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        elif self.api_key:
            headers["x-api-key"] = self.api_key
        return headers

    @staticmethod
    def _extract_text(parsed):
        if not isinstance(parsed, dict):
            return ""
        content = parsed.get("content") if isinstance(parsed, dict) else None
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            text = "".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ).strip()
            if text:
                return text

        choices = parsed.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0] or {}
            message = first.get("message") or {}
            text = message.get("content") or first.get("text")
            if isinstance(text, str):
                return text.strip()
        text = parsed.get("text")
        return text.strip() if isinstance(text, str) else ""


class AgentWorker:
    def __init__(self):
        self.client = CompatibleClient()
        self.mode = "local"
        self._lock = threading.RLock()
        self._thread = None
        self._turn_seq = 0
        self.busy = False
        self.error = None
        self.history = deque(maxlen=AGENT_HISTORY_LIMIT)
        self.latest = None
        self.heart_samples = deque()
        self.last_auto_at = 0.0
        self.last_observed_status = None
        self.last_observed_sqi_bucket = None
        self.last_observed_light_revision = None
        self.last_observed_light_enabled = None
        self.last_commented_bpm = None
        self.last_completed_at = None

    def status(self):
        now = time.time()
        with self._lock:
            latest = dict(self.latest or {})
            visible = bool(latest.get("subtitle")) and safe_float(latest.get("expires_at")) is not None
            visible = visible and latest["expires_at"] > now
            latest["visible"] = visible
            public_config = self.client.public_status()
            return {
                **public_config,
                "mode": self.mode,
                "local_auto_enabled": True,
                "busy": self.busy,
                "error": self.error,
                "history": list(self.history),
                "latest": latest,
                "last_auto_at": safe_float(self.last_auto_at),
                "last_completed_at": self.last_completed_at,
                "auto_cooldown_seconds": AGENT_AUTO_COOLDOWN_SECONDS,
                "subtitle_ttl_seconds": AGENT_SUBTITLE_TTL_SECONDS,
                "log_path": str(self._current_log_path()),
            }

    def reset(self):
        with self._lock:
            if self.busy:
                return self.status(), 409
            self._turn_seq += 1
            self.history.clear()
            self.latest = None
            self.error = None
        self._write_log("reset", {})
        return self.status(), 200

    def enable_api(self, settings: dict):
        with self._lock:
            if self.busy:
                return self.status(), 409
            try:
                self.client.configure(settings)
            except ValueError as exc:
                return {**self.status(), "error": str(exc)}, 400
            self.mode = "api"
            self.error = None
        self._write_log("api_enabled", {"protocol": self.client.protocol, "model": self.client.model})
        return self.status(), 200

    def disable_api(self):
        with self._lock:
            self._turn_seq += 1
            self.mode = "local"
            self.busy = False
            self.error = None
        self._write_log("api_disabled", {})
        return self.status(), 200

    def submit_user_message(self, text: str, snapshot: dict):
        text = self._clean_user_text(text)
        if not text:
            return {**self.status(), "error": "empty message"}, 400
        if self.mode != "api" or not self.client.configured:
            self._record_error("外部 API 已关闭；本地模式只自动播报状态事件。", "manual", snapshot)
            return {**self.status(), "error": "外部 API 已关闭；请先开启 API 模式后使用自由对话。"}, 409

        now_ms = int(time.time() * 1000)
        with self._lock:
            if self.busy:
                return self.status(), 409
            vitals = self._vitals_summary(snapshot)
            self._remember_heart_sample_locked(vitals)
            heart_summary = self._heart_summary_locked()
            history_for_model = list(self.history)
            self.history.append({
                "role": "user",
                "event": "manual",
                "text": text,
                "created_at": now_ms,
            })
            turn_id = self._begin_turn_locked("manual", text, "manual", snapshot, history_for_model, heart_summary)

        self._write_log("user_message", {
            "text": text,
            "trigger": "manual",
            "turn_id": turn_id,
            "vitals": self._vitals_summary(snapshot),
        })
        return self.status(), 202

    def observe(self, snapshot: dict):
        now = time.time()
        with self._lock:
            vitals = self._vitals_summary(snapshot)
            self._remember_heart_sample_locked(vitals, now)
            trigger = self._auto_trigger_locked(vitals, now)
            if not trigger:
                return
            heart_summary = self._heart_summary_locked(now)
            history_for_model = list(self.history)
            if self.mode != "api":
                self._emit_local_reply_locked(trigger, snapshot, heart_summary, now)
                return
            turn_id = self._begin_turn_locked("auto", "", trigger, snapshot, history_for_model, heart_summary)

        self._write_log("auto_trigger", {
            "trigger": trigger,
            "turn_id": turn_id,
            "vitals": self._vitals_summary(snapshot),
        })

    def _begin_turn_locked(self, kind: str, text: str, trigger: str, snapshot: dict, history_for_model: list, heart_summary: dict):
        self.busy = True
        self.error = None
        self._turn_seq += 1
        turn_id = self._turn_seq
        self._thread = threading.Thread(
            target=self._run_turn,
            args=(turn_id, kind, text, trigger, snapshot, history_for_model, heart_summary),
            daemon=True,
        )
        self._thread.start()
        return turn_id

    def _run_turn(self, turn_id: int, kind: str, text: str, trigger: str, snapshot: dict, history_for_model: list, heart_summary: dict):
        try:
            messages = self._build_messages(kind, text, trigger, snapshot, history_for_model, heart_summary)
            raw_text = self.client.create_message(self._system_prompt(), messages)
            reply, subtitle = self._parse_agent_response(raw_text)
            now = time.time()
            entry = {
                "role": "assistant",
                "event": kind,
                "text": reply,
                "subtitle": subtitle,
                "trigger": trigger,
                "created_at": int(now * 1000),
            }
            latest = {
                "text": reply,
                "subtitle": subtitle,
                "trigger": trigger,
                "kind": kind,
                "created_at": int(now * 1000),
                "expires_at": now + AGENT_SUBTITLE_TTL_SECONDS,
            }
            with self._lock:
                if turn_id == self._turn_seq:
                    self.history.append(entry)
                    self.latest = latest
                    self.error = None
                    self.last_completed_at = now
                self.busy = False
            self._write_log("agent_reply", {
                "trigger": trigger,
                "kind": kind,
                "text": reply,
                "subtitle": subtitle,
                "model": self.client.model,
                "turn_id": turn_id,
                "vitals": self._vitals_summary(snapshot),
            })
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            with self._lock:
                if turn_id == self._turn_seq:
                    self.error = message
                self.busy = False
            self._write_log("error", {
                "trigger": trigger,
                "kind": kind,
                "error": message,
                "model": self.client.model,
                "turn_id": turn_id,
                "vitals": self._vitals_summary(snapshot),
            })

    def _record_error(self, message: str, trigger: str, snapshot: dict):
        with self._lock:
            self.error = message
        self._write_log("error", {
            "trigger": trigger,
            "error": message,
            "model": self.client.model,
            "vitals": self._vitals_summary(snapshot),
        })

    def _auto_trigger_locked(self, vitals: dict, now: float):
        status = vitals.get("output_status") or "waiting"
        sqi_bucket = self._sqi_bucket(vitals.get("sqi"))
        current_bpm = vitals.get("bpm") if vitals.get("bpm") is not None else vitals.get("model_hr")
        lighting = vitals.get("lighting") or {}
        light_revision = lighting.get("revision")
        light_enabled = lighting.get("enabled")

        reasons = []
        if self.last_observed_status is None:
            if status == "stable":
                reasons.append("first_stable")
        elif status != self.last_observed_status and status in {"stable", "preview", "low_sqi"}:
            reasons.append(f"status_{self.last_observed_status}_to_{status}")

        if (
            self.last_observed_sqi_bucket is not None
            and sqi_bucket != self.last_observed_sqi_bucket
            and sqi_bucket in {"stable", "preview", "low"}
        ):
            reasons.append(f"sqi_{self.last_observed_sqi_bucket}_to_{sqi_bucket}")

        if current_bpm is not None:
            if self.last_commented_bpm is None and status == "stable":
                reasons.append("first_bpm")
            elif self.last_commented_bpm is not None:
                delta = current_bpm - self.last_commented_bpm
                if abs(delta) >= AGENT_BPM_DELTA_TRIGGER:
                    reasons.append(f"bpm_delta_{delta:+.0f}")

        if light_revision is not None:
            if self.last_observed_light_revision is None:
                if light_enabled:
                    reasons.append("light_enabled")
            elif light_revision != self.last_observed_light_revision:
                if light_enabled != self.last_observed_light_enabled:
                    reasons.append("light_enabled_changed")
                else:
                    reasons.append("light_settings_changed")

        self.last_observed_status = status
        self.last_observed_sqi_bucket = sqi_bucket
        self.last_observed_light_revision = light_revision
        self.last_observed_light_enabled = light_enabled

        if not reasons:
            return None
        if self.busy:
            return None
        if now - self.last_auto_at < AGENT_AUTO_COOLDOWN_SECONDS:
            return None
        if vitals.get("capture_state") != "running":
            return None
        is_light_trigger = any(reason.startswith("light_") for reason in reasons)
        if not is_light_trigger and (not vitals.get("model_ready") or not vitals.get("has_face")):
            return None

        self.last_auto_at = now
        if current_bpm is not None:
            self.last_commented_bpm = current_bpm
        return ",".join(reasons[:2])

    def _emit_local_reply_locked(self, trigger: str, snapshot: dict, heart_summary: dict, now: float):
        vitals = self._vitals_summary(snapshot)
        reply, subtitle = self._local_reply(trigger, vitals, heart_summary)
        entry = {"role": "assistant", "event": "auto", "text": reply, "subtitle": subtitle, "trigger": trigger, "created_at": int(now * 1000)}
        self.history.append(entry)
        self.latest = {"text": reply, "subtitle": subtitle, "trigger": trigger, "kind": "auto", "created_at": int(now * 1000), "expires_at": now + AGENT_SUBTITLE_TTL_SECONDS}
        self.error = None
        self.last_completed_at = now
        self.last_auto_at = now
        self._write_log("local_reply", {"trigger": trigger, "text": reply, "subtitle": subtitle, "model": "local-broadcast", "vitals": vitals})

    @staticmethod
    def _local_reply(trigger: str, vitals: dict, heart_summary: dict):
        bpm = vitals.get("bpm") if vitals.get("bpm") is not None else vitals.get("model_hr")
        bpm_text = f"{round(bpm)} bpm" if isinstance(bpm, (int, float)) else "--"
        status = vitals.get("output_status") or "waiting"
        lighting = vitals.get("lighting") or {}
        if "light_" in trigger:
            if lighting.get("enabled"):
                return ("补光参数已经更新，画面会更均匀一些。保持脸部正对镜头，等信号稳定后我们继续看节奏～", "补光已更新 · 等待信号稳定")
            return ("补光已关闭，环境光不足时信号可能会变弱；需要更稳定的采集时可以随时再打开。", "补光已关闭 · 注意环境光")
        if status == "stable":
            trend = heart_summary.get("trend")
            trend_text = "上升" if trend == "rising" else "下降" if trend == "falling" else "平稳"
            return (f"✨ 采集稳定了！当前心率 {bpm_text}，信号质量不错。近段时间节奏{trend_text}，直播状态正好～", f"采集稳定 · {bpm_text} · 信号清晰")
        if status in {"preview", "low_sqi"}:
            return (f"现在参考心率约 {bpm_text}，不过画面信号还在调整。试试保持脸部居中、减少头动，让光线更均匀一些～", "信号优化中 · 调整角度或补光")
        if status == "no_face":
            return ("暂时没有检测到稳定人脸，把脸部回到镜头中央后，采集会自动继续。", "等待人脸回到镜头")
        if "bpm_delta_" in trigger:
            return (f"心率节奏有明显变化，目前约 {bpm_text}。放松呼吸、保持自然节奏，等信号再稳定一点～", f"心率变化 · 当前 {bpm_text}")
        return ("正在建立心率信号窗口，保持自然面对镜头，稍等片刻就会有更稳定的互动提示。", "正在建立心率信号")

    def _remember_heart_sample_locked(self, vitals: dict, now: float | None = None):
        now = now or time.time()
        bpm = vitals.get("bpm") if vitals.get("bpm") is not None else vitals.get("model_hr")
        sqi = vitals.get("sqi")
        if bpm is not None or sqi is not None:
            self.heart_samples.append({
                "t": now,
                "bpm": bpm,
                "sqi": sqi,
                "status": vitals.get("output_status"),
            })
        cutoff = now - AGENT_RECENT_HEART_SECONDS
        while self.heart_samples and self.heart_samples[0]["t"] < cutoff:
            self.heart_samples.popleft()

    def _heart_summary_locked(self, now: float | None = None):
        now = now or time.time()
        cutoff = now - AGENT_RECENT_HEART_SECONDS
        samples = [sample for sample in self.heart_samples if sample["t"] >= cutoff]
        bpms = [sample["bpm"] for sample in samples if sample.get("bpm") is not None]
        sqis = [sample["sqi"] for sample in samples if sample.get("sqi") is not None]
        if not bpms:
            return {"sample_count": len(samples), "window_seconds": AGENT_RECENT_HEART_SECONDS}
        delta = bpms[-1] - bpms[0] if len(bpms) >= 2 else 0.0
        trend = "rising" if delta >= 3 else "falling" if delta <= -3 else "steady"
        return {
            "sample_count": len(samples),
            "window_seconds": AGENT_RECENT_HEART_SECONDS,
            "latest_bpm": round(bpms[-1], 1),
            "avg_bpm": round(sum(bpms) / len(bpms), 1),
            "min_bpm": round(min(bpms), 1),
            "max_bpm": round(max(bpms), 1),
            "trend": trend,
            "trend_delta": round(delta, 1),
            "avg_sqi": round(sum(sqis) / len(sqis), 3) if sqis else None,
        }

    def _build_messages(self, kind: str, text: str, trigger: str, snapshot: dict, history_for_model: list, heart_summary: dict):
        prompt = self._user_prompt(kind, text, trigger, snapshot, heart_summary)
        messages = []
        for item in history_for_model[-10:]:
            role = item.get("role")
            body = self._compact_text(item.get("text"), 900)
            if role not in {"user", "assistant"} or not body:
                continue
            if messages and messages[-1]["role"] == role:
                messages[-1]["content"] = f"{messages[-1]['content']}\n\n{body}"
            else:
                messages.append({"role": role, "content": body})
        while messages and messages[0]["role"] != "user":
            messages.pop(0)
        if messages and messages[-1]["role"] == "user":
            messages[-1]["content"] = f"{messages[-1]['content']}\n\n{prompt}"
        else:
            messages.append({"role": "user", "content": prompt})
        return messages

    def _user_prompt(self, kind: str, text: str, trigger: str, snapshot: dict, heart_summary: dict):
        vitals = self._vitals_summary(snapshot)
        mode_line = (
            "这是主播手动发起的自由对话，请像直播互动搭档一样自然回应，同时给出可上屏的短字幕。"
            if kind == "manual"
            else "这是系统根据心率、采集或补光状态自动触发的直播反馈，请输出适合上屏的短句。"
        )
        user_line = f"\n用户消息：{text}" if text else ""
        return (
            f"{mode_line}\n"
            f"触发原因：{trigger}\n"
            f"当前直播上下文：{json.dumps(vitals, ensure_ascii=False)}\n"
            f"近 {AGENT_RECENT_HEART_SECONDS} 秒摘要：{json.dumps(heart_summary, ensure_ascii=False)}"
            f"{user_line}\n"
            "要求：可以自由聊直播氛围、光线、补光参数、背景效果、心率互动和状态反馈。"
            "只基于这些数值和对话内容，不要声称看到了摄像头画面或背景细节。"
            "不要诊断、不要说病名、不要替代医疗建议。"
            "请返回 JSON 对象，字段为 reply 和 subtitle。subtitle 要短，适合 OBS 字幕，最多两行。"
        )

    @staticmethod
    def _system_prompt():
        return (
            "你是直播互动导演 Agent，运行在 rPPG 心率与补光 Overlay 中。"
            "你的语气温暖、轻快、有现场感；默认使用中文。"
            "你可以根据心率/SQI、采集状态、补光开关、亮度、色温、灯位、光束范围和 Overlay 设置，"
            "给出直播互动、光线氛围、补光效果、背景观感和节奏反馈。"
            "你没有摄像头图像输入，不能声称自己亲眼看到了脸、背景或具体物体；只能说基于参数的建议或推测。"
            "心率和 SQI 只是互动展示信号，不是医疗监测。不要做诊断、不要推断疾病、不要给治疗建议；"
            "如状态看起来吃力，只温和建议休息、喝水、调整节奏。"
            "手动聊天可以稍完整，自动反馈必须短。"
            "你必须只输出 JSON：{\"reply\":\"...\",\"subtitle\":\"...\"}。"
        )

    def _parse_agent_response(self, raw_text: str):
        text = (raw_text or "").strip()
        parsed = None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if 0 <= start < end:
                try:
                    parsed = json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    parsed = None
        if isinstance(parsed, dict):
            reply = self._compact_text(parsed.get("reply") or parsed.get("text") or text, AGENT_MAX_REPLY_CHARS)
            subtitle = self._compact_text(parsed.get("subtitle") or reply, AGENT_MAX_SUBTITLE_CHARS)
            return reply, subtitle
        reply = self._compact_text(text, AGENT_MAX_REPLY_CHARS)
        return reply, self._compact_text(reply, AGENT_MAX_SUBTITLE_CHARS)

    @staticmethod
    def _vitals_summary(snapshot: dict):
        capture_status = snapshot.get("capture", {}) if isinstance(snapshot, dict) else {}
        model_status = snapshot.get("model", {}) if isinstance(snapshot, dict) else {}
        output = snapshot.get("output", {}) if isinstance(snapshot, dict) else {}
        settings_status = snapshot.get("settings", {}) if isinstance(snapshot, dict) else {}
        bpm = safe_float(output.get("bpm"))
        model_hr = safe_float(model_status.get("hr"))
        sqi = safe_float(model_status.get("SQI"))
        return {
            "bpm": bpm,
            "model_hr": model_hr,
            "sqi": sqi,
            "confidence": safe_float(output.get("confidence")),
            "output_status": output.get("status"),
            "output_reason": output.get("reason"),
            "capture_state": capture_status.get("state"),
            "capture_fps": safe_float(capture_status.get("input_fps")),
            "capture_width": capture_status.get("width"),
            "capture_height": capture_status.get("height"),
            "model_ready": bool(model_status.get("ready")),
            "has_face": bool(model_status.get("has_face")),
            "no_face_count": model_status.get("no_face_count", 0),
            "hr_window_seconds": model_status.get("hr_window_seconds"),
            "lighting": {
                "enabled": bool(settings_status.get("light_enabled")),
                "brightness": settings_status.get("brightness"),
                "temperature": settings_status.get("temperature"),
                "light_x": settings_status.get("light_x"),
                "light_y": settings_status.get("light_y"),
                "light_z": settings_status.get("light_z"),
                "range": settings_status.get("light_range"),
                "angle_enabled": bool(settings_status.get("light_angle_enabled")),
                "angle": settings_status.get("light_angle"),
                "revision": settings_status.get("light_revision"),
            },
            "overlay": {
                "pulse": bool(settings_status.get("pulse")),
            },
        }

    @staticmethod
    def _sqi_bucket(sqi):
        sqi = safe_float(sqi)
        if sqi is None:
            return "unknown"
        if sqi >= OUTPUT_SQI_THRESHOLD:
            return "stable"
        if sqi >= PREVIEW_SQI_THRESHOLD:
            return "preview"
        return "low"

    @staticmethod
    def _clean_user_text(text: str):
        return AgentWorker._compact_text(text, AGENT_MAX_USER_TEXT_CHARS)

    @staticmethod
    def _compact_text(text, limit: int):
        value = "" if text is None else str(text)
        value = " ".join(value.replace("\r", "\n").split())
        if len(value) <= limit:
            return value
        return value[: max(0, limit - 1)].rstrip() + "…"

    def _write_log(self, event: str, payload: dict):
        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "event": event,
            **payload,
        }
        try:
            AGENT_LOG_DIR.mkdir(parents=True, exist_ok=True)
            with self._current_log_path().open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass

    @staticmethod
    def _current_log_path():
        return AGENT_LOG_DIR / f"{datetime.now().strftime('%Y-%m-%d')}.jsonl"

