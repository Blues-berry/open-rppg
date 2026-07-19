"""Local browser-video heart-rate service for the Chrome extension prototype.

Run from the repository root:

    python demo/browser_video_heart_server.py

Then load demo/browser-video-heart-extension as an unpacked Chrome extension.
"""

from __future__ import annotations

import json
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import rppg


HOST = "127.0.0.1"
PORT = 8030
MODEL_NAME = "FacePhys.rlap"
OUTPUT_SQI_THRESHOLD = 0.38
PREVIEW_SQI_THRESHOLD = 0.2
HR_WINDOW_SECONDS = 10
METRIC_INTERVAL_SECONDS = 0.25
INPUT_TTL_SECONDS = 3.0
SESSION_TTL_SECONDS = 120.0
MAX_FRAME_BYTES = 4 * 1024 * 1024


def safe_float(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(value) or np.isinf(value):
        return None
    return value


class BrowserVideoSession:
    def __init__(self, session_id: str, model_name: str = MODEL_NAME):
        self.session_id = session_id
        self.model_name = model_name
        self.lock = threading.RLock()
        self.model = None
        self.context = None
        self.state = "loading"
        self.error = None
        self.created_at = time.time()
        self.updated_at = self.created_at
        self.frame_count = 0
        self.last_input_at = None
        self.last_client_ts = None
        self.last_metric_at = 0.0
        self.last_metric_ms = None
        self.last_update_ms = None
        self.last_result = None
        self.last_box = None
        self._start_model()

    def _start_model(self):
        try:
            self.model = rppg.Model(self.model_name)
            self.model.face_detection_threads = 1
            self.model.face_resampling_threads = 1
            self.model.face_detect_per_n = 3
            self.context = self.model.__enter__()
            self.state = "ready"
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.state = "failed"

    def close(self):
        with self.lock:
            if self.context is not None and self.model is not None:
                try:
                    self.model.__exit__(None, None, None)
                except Exception:
                    pass
            self.context = None
            self.model = None
            self.state = "stopped"
            self.updated_at = time.time()

    def submit_jpeg(self, payload: bytes, ts: float | None):
        started = time.perf_counter()
        if len(payload) > MAX_FRAME_BYTES:
            return {"ok": False, "error": "frame too large", **self.status()}, 413

        arr = np.frombuffer(payload, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return {"ok": False, "error": "invalid image", **self.status()}, 400
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        with self.lock:
            if self.state != "ready" or self.model is None:
                return {"ok": False, "error": self.error or self.state, **self.status()}, 409
            try:
                model_ts = ts if ts is not None else time.time()
                self.model.update_frame(rgb, model_ts)
                self.frame_count += 1
                self.last_input_at = time.time()
                self.last_client_ts = model_ts
                self.updated_at = self.last_input_at
                self.last_box = self._box_to_list(getattr(self.model, "box", None))
                self.last_update_ms = (time.perf_counter() - started) * 1000
                self._update_metric()
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"
                self.state = "failed"
                return {"ok": False, "error": self.error, **self.status()}, 500

        return {"ok": True, **self.status()}, 200

    def _update_metric(self):
        now = time.time()
        if now - self.last_metric_at <= METRIC_INTERVAL_SECONDS:
            return
        self.last_metric_at = now
        started = time.perf_counter()
        try:
            self.last_result = self.model.hr(start=-HR_WINDOW_SECONDS, return_hrv=False)
        except Exception:
            self.last_result = None
        self.last_metric_ms = (time.perf_counter() - started) * 1000

    def status(self):
        with self.lock:
            result = self.last_result or {}
            hr = safe_float(result.get("hr") if isinstance(result, dict) else None)
            sqi = safe_float(result.get("SQI") if isinstance(result, dict) else None)
            has_hr = hr is not None and 30 <= hr <= 180
            has_sqi = sqi is not None
            has_recent_input = self.last_input_at is not None and time.time() - self.last_input_at <= INPUT_TTL_SECONDS
            has_face = bool(self.last_box)

            bpm = None
            output_state = "waiting"
            reason = "no_frame"
            if self.state == "failed":
                output_state = "failed"
                reason = self.error or "failed"
            elif self.state == "stopped":
                output_state = "stopped"
                reason = "stopped"
            elif not has_recent_input:
                output_state = "warming"
                reason = "no_recent_input"
            elif not has_face:
                output_state = "no_face"
                reason = "no_face"
            elif has_hr and has_sqi and sqi >= OUTPUT_SQI_THRESHOLD:
                bpm = round(hr)
                output_state = "stable"
                reason = "ready"
            elif has_hr and has_sqi and sqi >= PREVIEW_SQI_THRESHOLD:
                output_state = "preview"
                reason = "low_sqi_preview"
            elif has_hr and has_sqi:
                output_state = "low_sqi"
                reason = "low_sqi"
            else:
                output_state = "warming"
                reason = "building_window"

            return {
                "session_id": self.session_id,
                "state": self.state,
                "created_at": int(self.created_at * 1000),
                "updated_at": int(self.updated_at * 1000),
                "frame_count": self.frame_count,
                "hr_window_seconds": HR_WINDOW_SECONDS,
                "model": self.model_name,
                "has_face": has_face,
                "box": self.last_box,
                "input": {
                    "last_input_at": None if self.last_input_at is None else int(self.last_input_at * 1000),
                },
                "perf": {
                    "update_ms": safe_float(self.last_update_ms),
                    "metric_ms": safe_float(self.last_metric_ms),
                },
                "result": {
                    "hr": hr,
                    "SQI": sqi,
                    "bpm": bpm,
                    "confidence": sqi or 0.0,
                    "status": output_state,
                    "reason": reason,
                },
                "error": self.error,
            }

    @staticmethod
    def _box_to_list(box):
        if box is None:
            return None
        try:
            arr = np.asarray(box).astype(int)
            return arr.tolist()
        except Exception:
            return None


class SessionRegistry:
    def __init__(self):
        self.lock = threading.RLock()
        self.sessions: dict[str, BrowserVideoSession] = {}

    def create(self):
        self.cleanup()
        session_id = uuid.uuid4().hex
        session = BrowserVideoSession(session_id)
        with self.lock:
            self.sessions[session_id] = session
        return session

    def get(self, session_id: str):
        with self.lock:
            return self.sessions.get(session_id)

    def stop(self, session_id: str):
        session = self.get(session_id)
        if session is None:
            return None
        session.close()
        return session

    def cleanup(self):
        cutoff = time.time() - SESSION_TTL_SECONDS
        with self.lock:
            stale = [
                session_id
                for session_id, session in self.sessions.items()
                if session.updated_at < cutoff or session.state == "stopped"
            ]
            for session_id in stale:
                self.sessions.pop(session_id, None)

    def shutdown(self):
        with self.lock:
            sessions = list(self.sessions.values())
            self.sessions.clear()
        for session in sessions:
            session.close()


registry = SessionRegistry()


class BrowserVideoHandler(BaseHTTPRequestHandler):
    server_version = "BrowserVideoHeart/0.1"

    def do_OPTIONS(self):
        self.send_response(204)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        if path == "/api/browser-video/health":
            self._send_json({"ok": True, "service": "browser-video-heart", "model": MODEL_NAME})
            return

        match = self._session_path(path, "status")
        if match:
            session = registry.get(match)
            if session is None:
                self._send_json({"ok": False, "error": "session not found"}, status=404)
                return
            self._send_json({"ok": True, **session.status()})
            return

        self._send_json({"ok": False, "error": "not found"}, status=404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        query = parse_qs(parsed.query)

        if path == "/api/browser-video/session":
            session = registry.create()
            self._send_json({"ok": True, **session.status()}, status=201)
            return

        session_id = self._session_path(path, "frame")
        if session_id:
            session = registry.get(session_id)
            if session is None:
                self._send_json({"ok": False, "error": "session not found"}, status=404)
                return
            length = int(self.headers.get("Content-Length", "0") or 0)
            if length <= 0:
                self._send_json({"ok": False, "error": "empty frame"}, status=400)
                return
            ts = self._float_query(query, "ts")
            body, status = session.submit_jpeg(self.rfile.read(length), ts)
            self._send_json(body, status=status)
            return

        session_id = self._session_path(path, "stop")
        if session_id:
            session = registry.stop(session_id)
            if session is None:
                self._send_json({"ok": False, "error": "session not found"}, status=404)
                return
            self._send_json({"ok": True, **session.status()})
            return

        self._send_json({"ok": False, "error": "not found"}, status=404)

    @staticmethod
    def _session_path(path: str, action: str):
        parts = [part for part in path.split("/") if part]
        if len(parts) == 4 and parts[:2] == ["api", "browser-video"] and parts[2] == "session":
            return None
        if len(parts) == 5 and parts[:3] == ["api", "browser-video", "session"] and parts[4] == action:
            return parts[3]
        return None

    @staticmethod
    def _float_query(query, key: str):
        try:
            return float(query.get(key, [None])[0])
        except (TypeError, ValueError):
            return None

    def _send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "86400")

    def _send_json(self, body, status=200):
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):
        return


def main():
    server = ThreadingHTTPServer((HOST, PORT), BrowserVideoHandler)
    print(f"Browser video heart service: http://{HOST}:{PORT}/api/browser-video/health")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        registry.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
