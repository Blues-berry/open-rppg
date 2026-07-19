from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from datetime import datetime

import cv2
import numpy as np

from .config import (
    CAPTURE_NO_FRAME_TIMEOUT_SECONDS,
    DEFAULT_CAPTURE,
    INPUT_FPS_WINDOW_SECONDS,
    MJPEG_CACHE_SECONDS,
    MODEL_INPUT_TARGET_FPS,
    PREVIEW_JPEG_QUALITY,
)
from .light import apply_virtual_fill_light, draw_virtual_fill_light_marker
from .runtime import OpenRppgRuntime
from .utils import as_bool, camera_api, clamp_int, safe_float

class CaptureWorker:
    def __init__(self, runtime: OpenRppgRuntime, settings, recording, highlight_tracker):
        self.runtime = runtime
        self.settings = settings
        self.recording = recording
        self.highlight_tracker = highlight_tracker
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._model_frame_event = threading.Event()
        self._thread = None
        self._model_thread = None
        self._last_frame_bgr = None
        self._pending_model_bgr = None
        self._pending_model_ts = None
        self._last_model_queue_mono = 0.0
        self._preview_cache = {}
        self._recent_frame_times = deque()
        self.state = "idle"
        self.error = None
        self.device_index = DEFAULT_CAPTURE["device_index"]
        self.width = DEFAULT_CAPTURE["width"]
        self.height = DEFAULT_CAPTURE["height"]
        self.fps = DEFAULT_CAPTURE["fps"]
        self.actual_width = None
        self.actual_height = None
        self.actual_fps = None
        self.frames_read = 0
        self.dropped_frames = 0
        self.last_frame_at = None
        self.started_at = None
        self.stopped_at = None
        self.last_read_ms = None
        self.last_model_state = None
        self.session_id = None

    def start(self, config: dict | None = None):
        config = config or {}
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return self.status()

            self.device_index = self._as_int(config.get("device_index"), DEFAULT_CAPTURE["device_index"])
            self.width = self._as_int(config.get("width"), DEFAULT_CAPTURE["width"])
            self.height = self._as_int(config.get("height"), DEFAULT_CAPTURE["height"])
            self.fps = self._as_int(config.get("fps"), DEFAULT_CAPTURE["fps"])
            self.error = None
            self.state = "starting"
            self.frames_read = 0
            self.dropped_frames = 0
            self.last_frame_at = None
            self.started_at = time.time()
            self.session_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
            self.stopped_at = None
            self.last_read_ms = None
            self.last_model_state = None
            self.actual_width = None
            self.actual_height = None
            self.actual_fps = None
            self._last_frame_bgr = None
            self._pending_model_bgr = None
            self._pending_model_ts = None
            self._last_model_queue_mono = 0.0
            self._preview_cache.clear()
            self._recent_frame_times.clear()
            self._stop_event.clear()
            self._model_frame_event.clear()
            self.highlight_tracker.start_session(self.session_id, self.started_at)
            self.recording.begin_capture_session(self.session_id, self.started_at)
            self.runtime.start_async()
            self._model_thread = threading.Thread(target=self._run_model_input, daemon=True)
            self._model_thread.start()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
            return self.status()

    def stop(self, reset_model: bool = True):
        thread = None
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                self.state = "stopping"
                self._stop_event.set()
                thread = self._thread
            else:
                self.state = "idle"
                self.stopped_at = time.time()

        if thread is not None:
            thread.join(timeout=4.0)
        model_thread = None
        with self._lock:
            if self._model_thread is not None and self._model_thread.is_alive():
                self._model_frame_event.set()
                model_thread = self._model_thread
        if model_thread is not None:
            model_thread.join(timeout=2.0)

        with self._lock:
            if self.state == "stopping":
                self.state = "idle"
            self.stopped_at = time.time()
            self._model_thread = None
            self._pending_model_bgr = None
            self._pending_model_ts = None

        if reset_model:
            self.runtime.reset_session("capture_stop")
        return self.status()

    def status(self):
        with self._lock:
            return {
                "state": self.state,
                "error": self.error,
                "device_index": self.device_index,
                "width": self.actual_width or self.width,
                "height": self.actual_height or self.height,
                "fps": self.actual_fps or self.fps,
                "target_fps": self.fps,
                "input_fps": safe_float(self._input_fps_locked()),
                "frames_read": self.frames_read,
                "dropped_frames": self.dropped_frames,
                "last_frame_at": self.last_frame_at,
                "started_at": self.started_at,
                "stopped_at": self.stopped_at,
                "session_id": self.session_id,
                "read_ms": safe_float(self.last_read_ms),
                "model_state": self.last_model_state,
            }

    def snapshot_jpeg(self, light_settings: dict | None = None):
        cache_key = self._preview_cache_key("snapshot", light_settings)
        cached = self._cached_jpeg(cache_key)
        if cached is not None:
            return cached

        frame, _has_live_frame = self._current_frame()
        if frame is None:
            frame = self._placeholder_frame(self.state, self.error or "Waiting for capture")
        elif _has_live_frame:
            frame = apply_virtual_fill_light(frame, light_settings)
        encoded = self._encode_jpeg(frame)
        self._store_cached_jpeg(cache_key, encoded)
        return encoded

    def light_preview_jpeg(self, light_settings: dict | None = None):
        cache_key = self._preview_cache_key("light-preview", light_settings)
        cached = self._cached_jpeg(cache_key)
        if cached is not None:
            return cached

        frame, _has_live_frame = self._current_frame()
        if frame is None:
            frame = self._placeholder_frame(self.state, self.error or "Waiting for capture")

        preview_settings = dict(light_settings or {})
        preview_settings["light_enabled"] = True
        simulated = apply_virtual_fill_light(frame.copy(), preview_settings)
        simulated = draw_virtual_fill_light_marker(simulated, preview_settings)
        encoded = self._encode_jpeg(simulated)
        self._store_cached_jpeg(cache_key, encoded)
        return encoded

    def _current_frame(self):
        with self._lock:
            frame = None if self._last_frame_bgr is None else self._last_frame_bgr.copy()
            has_live_frame = frame is not None
            return frame, has_live_frame

    @staticmethod
    def _encode_jpeg(frame: np.ndarray):
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), PREVIEW_JPEG_QUALITY])
        if not ok:
            return b""
        return encoded.tobytes()

    @staticmethod
    def _preview_cache_key(kind: str, light_settings: dict | None):
        light_settings = light_settings or {}
        return (
            kind,
            bool(light_settings.get("light_enabled")),
            clamp_int(light_settings.get("brightness"), 20, 100, 72),
            clamp_int(light_settings.get("temperature"), 2700, 6500, 4800),
            clamp_int(light_settings.get("light_x"), 0, 100, 50),
            clamp_int(light_settings.get("light_y"), 0, 100, 38),
            clamp_int(light_settings.get("light_z"), 0, 100, 45),
            clamp_int(light_settings.get("light_range"), 15, 120, 58),
            as_bool(light_settings.get("light_angle_enabled"), False),
            clamp_int(light_settings.get("light_angle"), -75, 75, 0),
            int(light_settings.get("light_revision") or 0),
        )

    def _cached_jpeg(self, cache_key):
        now = time.monotonic()
        with self._lock:
            cached = self._preview_cache.get(cache_key)
            if cached and now - cached[0] <= MJPEG_CACHE_SECONDS:
                return cached[1]
        return None

    def _store_cached_jpeg(self, cache_key, encoded: bytes):
        now = time.monotonic()
        with self._lock:
            self._preview_cache[cache_key] = (now, encoded)
            if len(self._preview_cache) > 8:
                oldest_key = min(self._preview_cache, key=lambda key: self._preview_cache[key][0])
                self._preview_cache.pop(oldest_key, None)

    def _run(self):
        cap = None
        capture_base_mono = None
        try:
            cap = cv2.VideoCapture(self.device_index, camera_api())
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)

            if not cap.isOpened():
                self._set_error("camera_error", f"Cannot open camera device {self.device_index}")
                return

            self._set_running(cap)
            last_frame_mono = None
            capture_base_mono = time.monotonic()
            min_period = 1.0 / max(1, self.fps)

            while not self._stop_event.is_set():
                loop_started = time.perf_counter()
                ok, bgr = cap.read()
                now_mono = time.monotonic()
                if not ok or bgr is None:
                    self._mark_drop()
                    if last_frame_mono is None or now_mono - last_frame_mono > CAPTURE_NO_FRAME_TIMEOUT_SECONDS:
                        self._set_error("camera_error", "Camera stopped returning frames")
                        return
                    time.sleep(0.02)
                    continue

                last_frame_mono = now_mono
                read_ms = (time.perf_counter() - loop_started) * 1000
                self._mark_frame(bgr, read_ms)
                session_elapsed = max(0.0, now_mono - capture_base_mono) if capture_base_mono is not None else 0.0
                try:
                    if self.recording.wants_frame():
                        record_frame = apply_virtual_fill_light(bgr.copy(), self.settings.status())
                        self.recording.write_frame(record_frame, session_elapsed, self.actual_fps or self.fps)
                except Exception:
                    pass

                if self.runtime.is_ready():
                    self._queue_model_frame(bgr, session_elapsed, now_mono)
                else:
                    self.runtime.start_async()
                    self._set_model_state(self.runtime.state)

                elapsed = time.perf_counter() - loop_started
                if elapsed < min_period:
                    time.sleep(min_period - elapsed)
        except Exception as exc:  # pragma: no cover - surfaced to UI
            self._set_error("camera_error", f"{type(exc).__name__}: {exc}")
        finally:
            if cap is not None:
                cap.release()
            self.recording.end_capture_session(self.highlight_tracker.snapshot())
            with self._lock:
                if self.state in {"running", "starting", "stopping"}:
                    self.state = "idle" if self._stop_event.is_set() else self.state
                self.stopped_at = time.time()

    def _queue_model_frame(self, bgr: np.ndarray, session_elapsed: float, now_mono: float):
        min_period = 1.0 / max(1, MODEL_INPUT_TARGET_FPS)
        with self._lock:
            if now_mono - self._last_model_queue_mono < min_period:
                return
            self._last_model_queue_mono = now_mono
            self._pending_model_bgr = bgr.copy()
            self._pending_model_ts = float(session_elapsed)
            self._model_frame_event.set()

    def _take_model_frame(self):
        with self._lock:
            if self._pending_model_bgr is None:
                self._model_frame_event.clear()
                return None, None
            bgr = self._pending_model_bgr
            ts = self._pending_model_ts
            self._pending_model_bgr = None
            self._pending_model_ts = None
            if self._pending_model_bgr is None:
                self._model_frame_event.clear()
            return bgr, ts

    def _run_model_input(self):
        while not self._stop_event.is_set():
            self._model_frame_event.wait(timeout=0.25)
            if self._stop_event.is_set():
                break
            bgr, ts = self._take_model_frame()
            if bgr is None:
                continue
            if not self.runtime.is_ready():
                self.runtime.start_async()
                self._set_model_state(self.runtime.state)
                continue
            try:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                status = self.runtime.submit_capture_frame(rgb, ts)
                self._set_model_state(status.get("state"))
            except Exception as exc:
                self._set_model_state(f"{type(exc).__name__}: {exc}")

    def _set_running(self, cap):
        with self._lock:
            self.state = "running"
            self.error = None
            self.actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or self.width)
            self.actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or self.height)
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            self.actual_fps = safe_float(actual_fps) or self.fps

    def _set_error(self, state: str, message: str):
        with self._lock:
            self.state = state
            self.error = message
            self.stopped_at = time.time()

    def _mark_frame(self, bgr: np.ndarray, read_ms: float):
        now = time.time()
        with self._lock:
            self.frames_read += 1
            self.last_frame_at = now
            self.last_read_ms = read_ms
            self._last_frame_bgr = bgr.copy()
            self._recent_frame_times.append(now)
            cutoff = now - INPUT_FPS_WINDOW_SECONDS
            while self._recent_frame_times and self._recent_frame_times[0] < cutoff:
                self._recent_frame_times.popleft()

    def _mark_drop(self):
        with self._lock:
            self.dropped_frames += 1

    def _set_model_state(self, state: str | None):
        with self._lock:
            self.last_model_state = state

    def _input_fps_locked(self):
        if len(self._recent_frame_times) < 2:
            return None
        duration = self._recent_frame_times[-1] - self._recent_frame_times[0]
        if duration <= 0:
            return None
        return (len(self._recent_frame_times) - 1) / duration

    @staticmethod
    def _as_int(value, default):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _placeholder_frame(state: str, message: str):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame[:] = (18, 22, 25)
        cv2.putText(frame, "rPPG Camera Source", (64, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (240, 247, 247), 4)
        cv2.putText(frame, state.upper(), (64, 190), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (91, 193, 245), 3)
        cv2.putText(frame, message[:70], (64, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (181, 179, 168), 2)
        return frame


