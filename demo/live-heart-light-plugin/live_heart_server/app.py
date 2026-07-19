from __future__ import annotations

import time

import cv2

from .agent import AgentWorker
from .capture import CaptureWorker
from .config import (
    DEVICE_SCAN_MAX,
    HIGHLIGHT_CLIP_SECONDS,
    HIGHLIGHT_HALF_SECONDS,
    HIGHLIGHT_MAX_CLIP_SECONDS,
    HIGHLIGHT_MIN_CLIP_SECONDS,
    OUTPUT_SQI_THRESHOLD,
    OUTPUT_TTL_SECONDS,
    PREVIEW_SQI_THRESHOLD,
)
from .highlights import HighlightTracker
from .recording import RecordingManager
from .runtime import OpenRppgRuntime
from .settings import OverlaySettings
from .utils import camera_api, safe_float
from .video import VideoAnalysisWorker


class LiveHeartApp:
    def __init__(self):
        self.runtime = OpenRppgRuntime()
        self.settings = OverlaySettings()
        self.recording = RecordingManager()
        self.highlight_tracker = HighlightTracker()
        self.capture = CaptureWorker(self.runtime, self.settings, self.recording, self.highlight_tracker)
        self.video_analysis = VideoAnalysisWorker()
        self.agent = AgentWorker()

    def highlights_state(self):
        state = self.highlight_tracker.status(self.recording)
        return {
            "recording": self.recording.status(),
            "items": state.get("items", []),
            "export": self.recording.export_status(),
            "clip_seconds": HIGHLIGHT_CLIP_SECONDS,
            "half_seconds": HIGHLIGHT_HALF_SECONDS,
            "dynamic_clip": True,
            "min_clip_seconds": HIGHLIGHT_MIN_CLIP_SECONDS,
            "max_clip_seconds": HIGHLIGHT_MAX_CLIP_SECONDS,
        }

    def overlay_state(self, include_agent: bool = True, observe_agent: bool = True):
        capture_status = self.capture.status()
        model_status = self.runtime.status()
        result = model_status.get("result", {})
        hr = result.get("hr")
        sqi = result.get("SQI")
        now = time.time()
        recent_input = (
            model_status.get("last_input_at") is not None
            and now - model_status["last_input_at"] <= OUTPUT_TTL_SECONDS
        )
        capture_running = capture_status["state"] == "running"
        has_hr = isinstance(hr, (int, float)) and 30 <= hr <= 180
        has_sqi = isinstance(sqi, (int, float))
        has_face = bool(model_status.get("has_face"))

        bpm = None
        confidence = safe_float(sqi) or 0.0
        status = "waiting"
        reason = "capture_idle"
        source = "none"

        if not capture_running:
            reason = capture_status.get("error") or "capture_idle"
        elif not model_status.get("ready"):
            status = "warming"
            reason = model_status.get("state") or "model_loading"
            source = "open-rppg"
        elif not recent_input:
            status = "warming"
            reason = "no_recent_input"
            source = "open-rppg"
        elif not has_face:
            status = "no_face"
            reason = "no_face"
            source = "open-rppg"
        elif has_hr and has_sqi and sqi >= OUTPUT_SQI_THRESHOLD:
            bpm = round(hr)
            status = "stable"
            reason = "ready"
            source = "open-rppg"
        elif has_hr and has_sqi and sqi >= PREVIEW_SQI_THRESHOLD:
            status = "preview"
            reason = "low_sqi_preview"
            source = "open-rppg"
        elif has_hr and has_sqi:
            status = "low_sqi"
            reason = "low_sqi"
            source = "open-rppg"
        else:
            status = "warming"
            reason = "building_window"
            source = "open-rppg"

        state = {
            "capture": capture_status,
            "model": {
                "state": model_status.get("state"),
                "ready": model_status.get("ready"),
                "model": model_status.get("model"),
                "frame_count": model_status.get("frame_count"),
                "hr_window_seconds": model_status.get("hr_window_seconds"),
                "metric_seq": model_status.get("metric_seq"),
                "metric_window": model_status.get("metric_window", {}),
                "metric_captured_at": model_status.get("metric_captured_at"),
                "input_fps": model_status.get("input_fps"),
                "last_input_at": model_status.get("last_input_at"),
                "hr": safe_float(hr),
                "SQI": safe_float(sqi),
                "latency": safe_float(result.get("latency")),
                "box": model_status.get("box"),
                "has_face": has_face,
                "no_face_count": model_status.get("no_face_count", 0),
                "error": model_status.get("error"),
                "perf": model_status.get("perf", {}),
                "internal": model_status.get("internal", {}),
                "waveform": model_status.get("waveform", {}),
            },
            "output": {
                "bpm": bpm,
                "confidence": safe_float(confidence) or 0.0,
                "status": status,
                "reason": reason,
                "source": source,
                "updated_at": int(now * 1000),
            },
            "settings": self.settings.status(),
        }
        self.highlight_tracker.observe(capture_status, model_status, state["output"])
        state["highlights"] = self.highlights_state()
        if observe_agent:
            self.agent.observe(state)
        if include_agent:
            state["agent"] = self.agent.status()
        return state

    def list_devices(self):
        devices = []
        for index in range(DEVICE_SCAN_MAX + 1):
            cap = cv2.VideoCapture(index, camera_api())
            available = bool(cap.isOpened())
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) if available else None
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) if available else None
            fps = safe_float(cap.get(cv2.CAP_PROP_FPS)) if available else None
            cap.release()
            devices.append({
                "device_index": index,
                "available": available,
                "width": width,
                "height": height,
                "fps": fps,
            })
        return {"devices": devices}

    def shutdown(self):
        self.capture.stop(reset_model=False)
        self.runtime.stop()
