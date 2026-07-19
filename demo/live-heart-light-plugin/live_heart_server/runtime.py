from __future__ import annotations

import threading
import time
from functools import partial

import cv2
import numpy as np
try:
    from scipy.signal import butter, filtfilt
except Exception:  # pragma: no cover - optional display-only smoothing
    butter = None
    filtfilt = None

from .config import (
    FACE_CHANNELS,
    HR_WINDOW_SECONDS,
    INPUT_FPS_WINDOW_SECONDS,
    MAX_CLIENT_TS_GAP_SECONDS,
    METRIC_INTERVAL_SECONDS,
    WAVEFORM_SAMPLE_LIMIT,
)
from .utils import safe_float
import rppg

class OpenRppgRuntime:
    def __init__(self, model_name: str = "FacePhys.rlap"):
        self.model_name = model_name
        self.model = None
        self.state = "idle"
        self.error = None
        self.frame_count = 0
        self.last_result = None
        self.last_waveform = {"bvp": [], "ts": [], "sample_count": 0, "source": "none"}
        self.last_metric_window = {"start": None, "end": None}
        self.last_metric_wall_at = None
        self.metric_seq = 0
        self.last_box = None
        self.last_metric_at = 0.0
        self.last_input_at = None
        self.last_client_ts = None
        self.last_reset_at = None
        self.reset_reason = None
        self.last_request_ms = None
        self.last_update_ms = None
        self.last_metric_ms = None
        self.request_count = 0
        self.recent_input_times = []
        self.input_mode = None
        self.last_face_at = None
        self.last_no_face_at = None
        self._last_null_count = 0
        self.started_at = None
        self._context = None
        self._start_lock = threading.Lock()
        self._frame_lock = threading.Lock()
        self._start_thread = None

    def start_async(self):
        with self._start_lock:
            if self.state in {"loading", "ready"}:
                return
            self.state = "loading"
            self.error = None
            self._start_thread = threading.Thread(target=self._start, daemon=True)
            self._start_thread.start()

    def is_ready(self):
        return self.state == "ready" and self.model is not None

    def _start(self):
        try:
            model = rppg.Model(self.model_name)
            model.face_detection_threads = 1
            model.face_resampling_threads = 1
            model.face_detect_per_n = 3
            self._context = model.__enter__()
            self.model = model
            self.started_at = time.time()
            self.state = "ready"
        except Exception as exc:  # pragma: no cover - surfaced to UI
            self.error = f"{type(exc).__name__}: {exc}"
            self.state = "failed"

    def stop(self):
        self._close_model()
        self.state = "stopped"

    def reset_session(self, reason: str = "manual_reset"):
        with self._start_lock:
            with self._frame_lock:
                if self.model is not None:
                    self._restart_ready_context(reason)
                else:
                    self._clear_stream_state(reason)
                    self.state = "idle"
                    self.started_at = None

    def _close_model(self):
        if self.model is not None and self._context is not None:
            try:
                self.model.__exit__(None, None, None)
            except Exception:
                pass
        self.model = None
        self._context = None

    def _clear_stream_state(self, reason: str | None):
        self.error = None
        self.frame_count = 0
        self.last_result = None
        self.last_waveform = {"bvp": [], "ts": [], "sample_count": 0, "source": "none"}
        self.last_metric_window = {"start": None, "end": None}
        self.last_metric_wall_at = None
        self.metric_seq = 0
        self.last_box = None
        self.last_metric_at = 0.0
        self.last_input_at = None
        self.last_client_ts = None
        self.last_request_ms = None
        self.last_update_ms = None
        self.last_metric_ms = None
        self.request_count = 0
        self.recent_input_times = []
        self.input_mode = None
        self.last_face_at = None
        self.last_no_face_at = None
        self._last_null_count = 0
        self.last_reset_at = time.time()
        self.reset_reason = reason

    def _restart_ready_context(self, reason: str | None):
        model = self.model
        if model is None:
            self._clear_stream_state(reason)
            self.state = "idle"
            return
        if self._context is not None:
            try:
                model.__exit__(None, None, None)
            except Exception:
                pass
        self._clear_stream_state(reason)
        self._context = model.__enter__()
        self.started_at = time.time()
        self.state = "ready"

    def _ready_or_status(self):
        if self.state == "idle":
            self.start_async()
        if self.state != "ready" or self.model is None:
            return False, self.status()
        return True, None

    def submit_capture_frame(self, rgb: np.ndarray, ts: float):
        request_started = time.perf_counter()
        ready, status = self._ready_or_status()
        if not ready:
            return status

        with self._frame_lock:
            try:
                if self._has_large_client_gap(ts):
                    self._restart_ready_context("capture_gap")
                update_started = time.perf_counter()
                self.model.update_frame(rgb, ts)
                self.last_client_ts = ts
                self.last_update_ms = (time.perf_counter() - update_started) * 1000
                self._mark_input("capture")
                self.last_box = self._box_to_list(getattr(self.model, "box", None))
                self._update_metric()
            except Exception as exc:  # pragma: no cover - surfaced to UI
                self.error = f"{type(exc).__name__}: {exc}"
                self.state = "failed"

        self._mark_request(request_started)
        return self.status()

    def submit_jpeg(self, payload: bytes, ts: float | None):
        request_started = time.perf_counter()
        ready, status = self._ready_or_status()
        if not ready:
            return status

        arr = np.frombuffer(payload, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return {"ok": False, "error": "invalid image", **self.status()}
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        with self._frame_lock:
            try:
                if self._has_large_client_gap(ts):
                    self._restart_ready_context("input_gap")
                update_started = time.perf_counter()
                self.model.update_frame(rgb, ts or time.time())
                self.last_client_ts = ts
                self.last_update_ms = (time.perf_counter() - update_started) * 1000
                self._mark_input("frame")
                self.last_box = self._box_to_list(getattr(self.model, "box", None))
                self._update_metric()
            except Exception as exc:  # pragma: no cover - surfaced to UI
                self.error = f"{type(exc).__name__}: {exc}"
                self.state = "failed"

        self._mark_request(request_started)
        return {"ok": True, **self.status()}

    def submit_face(self, payload: bytes, ts: float | None, width: int, height: int, hasface: bool):
        request_started = time.perf_counter()
        ready, status = self._ready_or_status()
        if not ready:
            return status

        face = None
        if hasface:
            expected = width * height * FACE_CHANNELS
            if len(payload) != expected:
                return {"ok": False, "error": f"expected {expected} RGB bytes", **self.status()}
            face = np.frombuffer(payload, dtype=np.uint8).reshape((height, width, FACE_CHANNELS))
            target_h, target_w = self.model.input[1:3]
            if (height, width) != (target_h, target_w):
                face = cv2.resize(face, (target_w, target_h), interpolation=cv2.INTER_AREA)
            face = partial(lambda x: x, face)

        with self._frame_lock:
            try:
                if self._has_large_client_gap(ts):
                    self._restart_ready_context("input_gap")
                update_started = time.perf_counter()
                self.model.update_face_resized(face, ts=ts or time.time(), hasface=hasface)
                self.last_client_ts = ts
                self.last_update_ms = (time.perf_counter() - update_started) * 1000
                self._mark_input("face" if hasface else "noface")
                self.last_box = None
                self._update_metric()
            except Exception as exc:  # pragma: no cover - surfaced to UI
                self.error = f"{type(exc).__name__}: {exc}"
                self.state = "failed"

        self._mark_request(request_started)
        return {"ok": True, **self.status()}

    def _has_large_client_gap(self, ts: float | None):
        if ts is None or self.last_client_ts is None or self.frame_count <= 0:
            return False
        return ts - self.last_client_ts > MAX_CLIENT_TS_GAP_SECONDS

    def _mark_request(self, request_started: float):
        self.last_request_ms = (time.perf_counter() - request_started) * 1000
        self.request_count += 1

    def _mark_input(self, mode: str):
        now = time.time()
        self.frame_count += 1
        self.input_mode = mode
        self.last_input_at = now
        self.recent_input_times.append(now)
        cutoff = now - INPUT_FPS_WINDOW_SECONDS
        while self.recent_input_times and self.recent_input_times[0] < cutoff:
            self.recent_input_times.pop(0)

    def _update_metric(self):
        now = time.time()
        if now - self.last_metric_at <= METRIC_INTERVAL_SECONDS:
            return
        self.last_metric_at = now
        metric_started = time.perf_counter()
        result = None
        waveform = {"bvp": [], "ts": [], "sample_count": 0, "source": "none"}
        try:
            result = self.model.hr(start=-HR_WINDOW_SECONDS, return_hrv=False)
        except Exception:
            result = None
        try:
            waveform = self._build_waveform_snapshot()
        except Exception:
            waveform = {"bvp": [], "ts": [], "sample_count": 0, "source": "waveform_error"}
        self.last_result = result
        self.last_waveform = waveform
        self.last_metric_window = {
            "start": safe_float(waveform.get("window_start")),
            "end": safe_float(waveform.get("window_end")),
        }
        self.last_metric_wall_at = now
        self.metric_seq += 1
        self.last_metric_ms = (time.perf_counter() - metric_started) * 1000

    def status(self):
        with self._frame_lock:
            result = self.last_result or {}
            hr = result.get("hr") if isinstance(result, dict) else None
            sqi = result.get("SQI") if isinstance(result, dict) else None
            latency = result.get("latency") if isinstance(result, dict) else None
            internal = self._internal_status()
            waveform = dict(self.last_waveform or {})
            has_face = self._refresh_face_state(internal)
            return {
                "model": self.model_name,
                "state": self.state,
                "ready": self.state == "ready",
                "error": self.error,
                "frame_count": self.frame_count,
                "hr_window_seconds": HR_WINDOW_SECONDS,
                "metric_seq": self.metric_seq,
                "metric_window": dict(self.last_metric_window or {}),
                "metric_captured_at": self.last_metric_wall_at,
                "started_at": self.started_at,
                "box": self.last_box,
                "has_face": has_face,
                "last_face_at": self.last_face_at,
                "last_no_face_at": self.last_no_face_at,
                "no_face_count": internal["statistic"].get("null", 0),
                "input_mode": self.input_mode,
                "input_fps": safe_float(self._input_fps()),
                "last_input_at": self.last_input_at,
                "last_reset_at": self.last_reset_at,
                "reset_reason": self.reset_reason,
                "perf": {
                    "request_ms": safe_float(self.last_request_ms),
                    "update_ms": safe_float(self.last_update_ms),
                    "metric_ms": safe_float(self.last_metric_ms),
                    "request_count": self.request_count,
                },
                "internal": internal,
                "waveform": waveform,
                "result": {
                    "hr": safe_float(hr),
                    "SQI": safe_float(sqi),
                    "latency": safe_float(latency),
                },
            }

    def _internal_status(self):
        model = self.model
        if model is None:
            return {
                "n_frame": 0,
                "n_signal": 0,
                "face_buffer": 0,
                "signal_keys": [],
                "statistic": {},
            }
        signal_buff = getattr(model, "signal_buff", {}) or {}
        statistic = getattr(model, "statistic", {}) or {}
        return {
            "n_frame": int(getattr(model, "n_frame", 0) or 0),
            "n_signal": int(getattr(model, "n_signal", 0) or 0),
            "face_buffer": len(getattr(model, "face_buff", []) or []),
            "signal_keys": list(signal_buff.keys()),
            "statistic": {key: int(value or 0) for key, value in statistic.items()},
        }

    def _build_waveform_snapshot(self):
        model = self.model
        if model is None:
            return {"bvp": [], "ts": [], "sample_count": 0, "source": "model.bvp(raw=True)"}

        bvp = []
        ts = []
        source = "model.bvp(raw=True)+bandpass"
        try:
            bvp, ts = model.bvp(start=-HR_WINDOW_SECONDS, raw=True)
        except Exception:
            bvp, ts = [], []

        values, timestamps = self._paired_signal_tail(bvp, ts, WAVEFORM_SAMPLE_LIMIT)
        if values.size == 0:
            signal_buff = getattr(model, "signal_buff", {}) or {}
            raw_bvp = signal_buff.get("bvp") if isinstance(signal_buff, dict) else None
            fallback_ts = self._signal_timestamps(model, raw_bvp)
            values, timestamps = self._paired_signal_tail(raw_bvp, fallback_ts, WAVEFORM_SAMPLE_LIMIT)
            source = "signal_buff.bvp(raw)+bandpass" if values.size else source

        if values.size:
            values = self._display_waveform(values, getattr(model, "fps", 30))

        window_start = safe_float(timestamps[0]) if timestamps.size else None
        window_end = safe_float(timestamps[-1]) if timestamps.size else None
        return {
            "bvp": [round(float(value), 6) for value in values],
            "ts": [round(float(value), 4) for value in timestamps],
            "sample_count": int(values.size),
            "window_start": window_start,
            "window_end": window_end,
            "source": source,
        }

    @staticmethod
    def _paired_signal_tail(values, timestamps, limit: int):
        if values is None:
            return np.array([], dtype=np.float32), np.array([], dtype=np.float64)
        try:
            arr = np.asarray(values, dtype=np.float32).reshape(-1)
            if timestamps is None:
                return np.array([], dtype=np.float32), np.array([], dtype=np.float64)
            ts_arr = np.asarray(timestamps, dtype=np.float64).reshape(-1)
            if ts_arr.size == 0:
                return np.array([], dtype=np.float32), np.array([], dtype=np.float64)
            count = min(arr.size, ts_arr.size)
            arr = arr[:count]
            ts_arr = ts_arr[:count]
            finite = np.isfinite(arr) & np.isfinite(ts_arr)
            arr = arr[finite]
            ts_arr = ts_arr[finite]
        except Exception:
            return np.array([], dtype=np.float32), np.array([], dtype=np.float64)
        if arr.size == 0:
            return np.array([], dtype=np.float32), np.array([], dtype=np.float64)
        order = np.argsort(ts_arr, kind="mergesort")
        arr = arr[order]
        ts_arr = ts_arr[order]
        if ts_arr.size > 1:
            unique = np.concatenate(([True], np.diff(ts_arr) > 1e-6))
            arr = arr[unique]
            ts_arr = ts_arr[unique]
        return arr[-limit:], ts_arr[-limit:]

    @staticmethod
    def _signal_timestamps(model, values):
        if values is None:
            return []
        try:
            count = int(np.asarray(values).reshape(-1).size)
            ts = np.asarray(getattr(model, "ts", [])[:count], dtype=np.float64)
        except Exception:
            return []
        if ts.size == 0:
            return []
        return ts - ts[0]

    @staticmethod
    def _display_waveform(values: np.ndarray, fps) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32).reshape(-1)
        if arr.size < 2:
            return arr
        arr = arr - float(np.mean(arr))
        if butter is None or filtfilt is None:
            return arr
        sample_rate = safe_float(fps) or 30.0
        nyquist = sample_rate / 2.0
        low = 0.5 / nyquist
        high = min(3.0 / nyquist, 0.98)
        if arr.size < 24 or not 0 < low < high < 1:
            return arr
        try:
            b, a = butter(3, [low, high], btype="bandpass")
            return filtfilt(b, a, arr).astype(np.float32)
        except Exception:
            return arr

    def _refresh_face_state(self, internal: dict):
        model = self.model
        if model is None:
            return False
        now = time.time()
        self.last_box = self._box_to_list(getattr(model, "box", None))
        null_count = int((internal.get("statistic") or {}).get("null", 0) or 0)
        if null_count > self._last_null_count:
            self.last_no_face_at = now
            self._last_null_count = null_count
        if self.last_box is not None and (
            self.last_no_face_at is None or now - self.last_no_face_at > 1.0
        ):
            self.last_face_at = now
        if self.last_face_at is None:
            return False
        if now - self.last_face_at > 2.0:
            return False
        if self.last_no_face_at is not None and self.last_no_face_at > self.last_face_at:
            return False
        try:
            hasface_until = float(getattr(model, "hasface", 0) or 0)
            now = float(getattr(model, "now", 0) or 0)
            return hasface_until > now or self.last_box is not None
        except Exception:
            return self.last_box is not None

    def _input_fps(self):
        if len(self.recent_input_times) < 2:
            return None
        duration = self.recent_input_times[-1] - self.recent_input_times[0]
        if duration <= 0:
            return None
        return (len(self.recent_input_times) - 1) / duration

    @staticmethod
    def _box_to_list(box):
        if box is None:
            return None
        try:
            return np.asarray(box).astype(int).tolist()
        except Exception:
            return None


